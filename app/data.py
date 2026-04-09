"""
Data management module for VinFast Warranty AI Agent.

Handles loading mock data from JSON files, in-memory slot/booking management
with atomic state transitions (AVAILABLE -> PENDING -> CONFIRMED) and TTL support.
"""

import csv
import json
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional

# ─── Data directory ───────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOOKINGS_CSV_PATH = DATA_DIR / "bookings.csv"
USER_ID = "U_VIN_001"
TTL_SECONDS = 300
BOOKING_FIELDNAMES = [
    "booking_id",
    "user_id",
    "vehicle_id",
    "vin_number",
    "center_id",
    "center_name",
    "booking_date",
    "time_slot",
    "service_type",
    "ai_diagnosis_log",
    "note",
    "status",
    "created_at",
    "pending_expires_at",
    "ttl_seconds",
    "confirmed_at",
    "updated_at",
    "rescheduled_at",
]

# ─── In-memory stores ────────────────────────────────────────────
_vehicles: list[dict] = []
_warranty_policy: dict = {}
_service_centers: list[dict] = []
_time_slots: dict[str, dict] = {}  # slot_id -> slot object
_bookings: dict[str, dict] = {}    # booking_id -> booking object
_lock = threading.Lock()            # for atomic updates


# ─── Load static data from JSON ──────────────────────────────────
def _load_json(filename: str):
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_bookings_csv_exists():
    BOOKINGS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BOOKINGS_CSV_PATH.exists():
        return

    with open(BOOKINGS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOOKING_FIELDNAMES)
        writer.writeheader()


def _parse_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_optional(value):
    if value in ("", None):
        return None
    return value


def _booking_slot_id(booking: dict) -> str:
    return f"SLOT_{booking['center_id']}_{booking['booking_date']}_{booking['time_slot'].replace(':', '')}"


def _serialize_booking(booking: dict) -> dict:
    row = {}
    for field in BOOKING_FIELDNAMES:
        value = booking.get(field)
        row[field] = "" if value is None else value
    return row


def _load_bookings_from_csv() -> dict[str, dict]:
    _ensure_bookings_csv_exists()
    bookings: dict[str, dict] = {}

    with open(BOOKINGS_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            booking_id = row.get("booking_id")
            if not booking_id:
                continue

            booking = {field: _normalize_optional(row.get(field)) for field in BOOKING_FIELDNAMES}
            ttl_raw = booking.get("ttl_seconds")
            booking["ttl_seconds"] = int(ttl_raw) if ttl_raw not in (None, "") else None
            booking["status"] = str(booking.get("status") or "").upper()
            bookings[booking_id] = booking

    return bookings


def _save_bookings_to_csv_locked():
    _ensure_bookings_csv_exists()
    with open(BOOKINGS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOOKING_FIELDNAMES)
        writer.writeheader()
        for booking in sorted(_bookings.values(), key=lambda item: item.get("created_at") or ""):
            writer.writerow(_serialize_booking(booking))


def _clear_slot_assignment_internal(slot: dict):
    slot["status"] = "AVAILABLE"
    slot["pending_since"] = None
    slot["pending_booking_id"] = None


def _sync_slots_with_bookings_locked() -> bool:
    now = datetime.now()
    updated = False

    for booking in _bookings.values():
        status = str(booking.get("status") or "").upper()
        slot = _time_slots.get(_booking_slot_id(booking))

        if status == "PENDING":
            expires_at = _parse_datetime(booking.get("pending_expires_at"))
            if not slot or not expires_at or expires_at <= now:
                booking["status"] = "EXPIRED"
                booking["updated_at"] = now.isoformat()
                updated = True
                continue

            pending_started_at = expires_at.timestamp() - float(booking.get("ttl_seconds") or TTL_SECONDS)
            slot["status"] = "PENDING"
            slot["pending_since"] = pending_started_at
            slot["pending_booking_id"] = booking["booking_id"]
            continue

        if status == "CONFIRMED" and slot:
            slot["status"] = "CONFIRMED"
            slot["pending_since"] = None
            slot["pending_booking_id"] = booking["booking_id"]

    return updated


def init_data():
    """Load all mock data, restore persisted bookings, and generate time slots."""
    global _vehicles, _warranty_policy, _service_centers, _bookings
    _vehicles = _load_json("vehicles.json")
    _warranty_policy = _load_json("warranty_policy.json")
    _service_centers = _load_json("service_centers.json")
    _generate_time_slots()
    with _lock:
        _bookings = _load_bookings_from_csv()
        changed = _sync_slots_with_bookings_locked()
        if changed:
            _save_bookings_to_csv_locked()


def _parse_working_hours(working_hours: str) -> tuple[dt_time, dt_time]:
    """Extract opening/closing time from strings like '08:00 - 17:30 (Thứ 2 - Thứ 7)'."""
    matches = re.findall(r"(\d{2}:\d{2})", working_hours)
    if len(matches) >= 2:
        return (
            datetime.strptime(matches[0], "%H:%M").time(),
            datetime.strptime(matches[1], "%H:%M").time(),
        )
    return dt_time(8, 0), dt_time(17, 30)


def _center_operates_on_sunday(center: dict) -> bool:
    working_hours = center.get("working_hours", "")
    return "chủ nhật" in working_hours.lower() or "chu nhat" in working_hours.lower()


def _generate_slot_times_for_center(center: dict) -> list[str]:
    """Generate 30-minute appointment starts up to 30 minutes before closing."""
    opening_time, closing_time = _parse_working_hours(center.get("working_hours", ""))

    slot_times: list[str] = []
    cursor = datetime.combine(datetime.today().date(), opening_time)
    latest_start = datetime.combine(datetime.today().date(), closing_time) - timedelta(minutes=30)

    lunch_start = dt_time(11, 30)
    lunch_end = dt_time(13, 0)

    while cursor <= latest_start:
        current_time = cursor.time()
        if not (lunch_start <= current_time <= lunch_end):
            slot_times.append(cursor.strftime("%H:%M"))
        cursor += timedelta(minutes=30)

    return slot_times


def _generate_time_slots():
    """Generate available time slots for the next 7 days for each service center."""
    global _time_slots
    _time_slots = {}

    today = datetime.now().date()
    for center in _service_centers:
        slot_times = _generate_slot_times_for_center(center)
        for day_offset in range(1, 8):  # next 7 days
            date = today + timedelta(days=day_offset)
            # Skip Sunday unless the service center works on Sunday.
            if date.weekday() == 6 and not _center_operates_on_sunday(center):
                continue
            for t in slot_times:
                slot_id = f"SLOT_{center['id']}_{date.isoformat()}_{t.replace(':', '')}"
                _time_slots[slot_id] = {
                    "slot_id": slot_id,
                    "center_id": center["id"],
                    "center_name": center["name"],
                    "date": date.isoformat(),
                    "time": t,
                    "status": "AVAILABLE",       # AVAILABLE | PENDING | CONFIRMED
                    "pending_since": None,        # timestamp when moved to PENDING
                    "pending_booking_id": None,    # booking_id holding the slot
                }


# ─── Vehicle queries ─────────────────────────────────────────────
def get_all_vehicles() -> list[dict]:
    return _vehicles


def get_vehicle(vehicle_id: str) -> Optional[dict]:
    for v in _vehicles:
        if v["id"] == vehicle_id:
            return v
    return None


# ─── Warranty policy queries ─────────────────────────────────────
def get_warranty_policy() -> dict:
    return _warranty_policy


# ─── Service center queries ──────────────────────────────────────
def get_all_service_centers() -> list[dict]:
    return _service_centers


def get_service_centers_by_city(city: str) -> list[dict]:
    city_lower = city.lower().strip()
    results = []
    for sc in _service_centers:
        if city_lower in sc["city"].lower() or city_lower in sc["district"].lower():
            results.append(sc)
    return results


def get_service_center(center_id: str) -> Optional[dict]:
    for sc in _service_centers:
        if sc["id"] == center_id:
            return sc
    return None


# ─── Time Slot queries & state machine ───────────────────────────
def get_available_slots(center_id: str, date: Optional[str] = None) -> list[dict]:
    """Get available slots for a specific center, optionally filtered by date."""
    results = []
    for slot in _time_slots.values():
        if slot["center_id"] == center_id and slot["status"] == "AVAILABLE":
            if date is None or slot["date"] == date:
                results.append({
                    "slot_id": slot["slot_id"],
                    "date": slot["date"],
                    "time": slot["time"],
                    "status": slot["status"],
                })
    results.sort(key=lambda s: (s["date"], s["time"]))
    return results


def hold_slot(slot_id: str, vehicle_id: str, service_type: str,
              ai_diagnosis_log: str = "", note: str = "") -> Optional[dict]:
    """
    Atomic AVAILABLE -> PENDING transition.
    Creates a booking in PENDING state with TTL of 5 minutes.
    Returns the booking dict, or None if slot not available.
    """
    with _lock:
        slot = _time_slots.get(slot_id)
        if not slot or slot["status"] != "AVAILABLE":
            return None

        booking_id = f"BK_{uuid.uuid4().hex[:6].upper()}"
        now = time.time()
        now_iso = datetime.now().isoformat()

        slot["status"] = "PENDING"
        slot["pending_since"] = now
        slot["pending_booking_id"] = booking_id

        vehicle = get_vehicle(vehicle_id)

        booking = {
            "booking_id": booking_id,
            "user_id": "U_VIN_001",
            "vehicle_id": vehicle_id,
            "vin_number": vehicle["vin"] if vehicle else "N/A",
            "center_id": slot["center_id"],
            "center_name": slot["center_name"],
            "booking_date": slot["date"],
            "time_slot": slot["time"],
            "service_type": service_type,
            "ai_diagnosis_log": ai_diagnosis_log,
            "note": note,
            "status": "PENDING",
            "created_at": now_iso,
            "pending_expires_at": datetime.fromtimestamp(now + TTL_SECONDS).isoformat(),
            "ttl_seconds": TTL_SECONDS,
            "confirmed_at": None,
            "updated_at": now_iso,
            "rescheduled_at": None,
        }
        _bookings[booking_id] = booking
        _save_bookings_to_csv_locked()
        return dict(booking)


def confirm_booking(booking_id: str) -> Optional[dict]:
    """
    Atomic PENDING -> CONFIRMED transition.
    Returns updated booking or None if expired / not found.
    """
    with _lock:
        booking = _bookings.get(booking_id)
        if not booking or booking["status"] != "PENDING":
            return None

        slot = _time_slots.get(_booking_slot_id(booking))
        if slot and slot["status"] == "PENDING" and slot["pending_booking_id"] == booking_id:
            elapsed = time.time() - float(slot["pending_since"] or 0)
            if elapsed > TTL_SECONDS:
                _release_slot_internal(slot, booking)
                _save_bookings_to_csv_locked()
                return None

            slot["status"] = "CONFIRMED"
            slot["pending_since"] = None
            booking["status"] = "CONFIRMED"
            booking["pending_expires_at"] = None
            booking["ttl_seconds"] = None
            booking["confirmed_at"] = datetime.now().isoformat()
            booking["updated_at"] = booking["confirmed_at"]
            _save_bookings_to_csv_locked()
            return dict(booking)
        return None


def reschedule_booking(
    booking_id: str,
    center_id: str,
    slot_id: str,
    service_type: str | None = None,
    note: str | None = None,
) -> Optional[dict]:
    with _lock:
        booking = _bookings.get(booking_id)
        if not booking or booking.get("status") not in {"PENDING", "CONFIRMED"}:
            return None

        current_slot_id = _booking_slot_id(booking)
        new_slot = _time_slots.get(slot_id)
        if not new_slot or new_slot["center_id"] != center_id:
            return None

        now = time.time()
        now_iso = datetime.now().isoformat()

        if current_slot_id == slot_id:
            if service_type:
                booking["service_type"] = service_type
            if note is not None:
                booking["note"] = note
            booking["updated_at"] = now_iso
            booking["rescheduled_at"] = now_iso
            _save_bookings_to_csv_locked()
            return dict(booking)

        if new_slot["status"] != "AVAILABLE":
            return None

        old_slot = _time_slots.get(current_slot_id)
        if old_slot:
            _clear_slot_assignment_internal(old_slot)

        booking["center_id"] = new_slot["center_id"]
        booking["center_name"] = new_slot["center_name"]
        booking["booking_date"] = new_slot["date"]
        booking["time_slot"] = new_slot["time"]
        booking["service_type"] = service_type or booking.get("service_type") or "bao duong"
        if note is not None:
            booking["note"] = note
        booking["updated_at"] = now_iso
        booking["rescheduled_at"] = now_iso

        if booking["status"] == "CONFIRMED":
            new_slot["status"] = "CONFIRMED"
            new_slot["pending_since"] = None
            new_slot["pending_booking_id"] = booking_id
        else:
            new_slot["status"] = "PENDING"
            new_slot["pending_since"] = now
            new_slot["pending_booking_id"] = booking_id
            booking["pending_expires_at"] = datetime.fromtimestamp(now + TTL_SECONDS).isoformat()
            booking["ttl_seconds"] = TTL_SECONDS

        _save_bookings_to_csv_locked()
        return dict(booking)


def get_booking(booking_id: str) -> Optional[dict]:
    booking = _bookings.get(booking_id)
    return dict(booking) if booking else None


def get_user_bookings(user_id: str = "U_VIN_001", vehicle_id: str = None) -> list[dict]:
    """Get all bookings for a user, optionally filtered by vehicle_id."""
    results = []
    for booking in _bookings.values():
        if booking.get("user_id") == user_id:
            if vehicle_id and booking.get("vehicle_id") != vehicle_id:
                continue
            # Add TTL info for PENDING bookings
            entry = dict(booking)
            if entry["status"] == "PENDING":
                ttl = get_booking_ttl_remaining(entry["booking_id"])
                entry["ttl_remaining_seconds"] = ttl
            results.append(entry)
    results.sort(key=lambda b: b.get("updated_at") or b.get("created_at") or "", reverse=True)
    return results


def get_booking_ttl_remaining(booking_id: str) -> Optional[float]:
    """Returns remaining TTL in seconds, or None if not PENDING."""
    booking = _bookings.get(booking_id)
    if not booking or booking["status"] != "PENDING":
        return None
    slot = _time_slots.get(_booking_slot_id(booking))
    if not slot or slot["status"] != "PENDING" or slot["pending_booking_id"] != booking_id:
        return None
    elapsed = time.time() - float(slot["pending_since"] or 0)
    remaining = TTL_SECONDS - elapsed
    return max(0, remaining)


def _release_slot_internal(slot: dict, booking: dict):
    """Internal: release a slot back to AVAILABLE (must hold _lock)."""
    _clear_slot_assignment_internal(slot)
    booking["status"] = "EXPIRED"
    booking["updated_at"] = datetime.now().isoformat()


# ─── TTL Worker: clean up expired PENDING slots ──────────────────
def _ttl_worker():
    """Background thread that checks for expired PENDING slots every 10 seconds."""
    while True:
        time.sleep(10)
        now = time.time()
        changed = False
        with _lock:
            for slot in _time_slots.values():
                if slot["status"] == "PENDING" and slot["pending_since"]:
                    if now - float(slot["pending_since"]) > TTL_SECONDS:
                        bid = slot["pending_booking_id"]
                        if bid and bid in _bookings:
                            _release_slot_internal(slot, _bookings[bid])
                            changed = True
                        else:
                            _clear_slot_assignment_internal(slot)
                            changed = True
            if changed:
                _save_bookings_to_csv_locked()


def start_ttl_worker():
    """Start the background TTL cleanup worker as a daemon thread."""
    worker = threading.Thread(target=_ttl_worker, daemon=True)
    worker.start()
