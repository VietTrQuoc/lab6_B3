"""
Data management module for VinFast Warranty AI Agent.

Handles loading mock data from JSON files, in-memory slot/booking management
with atomic state transitions (AVAILABLE -> PENDING -> CONFIRMED) and TTL support.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─── Data directory ───────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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


def init_data():
    """Load all mock data and generate time slots."""
    global _vehicles, _warranty_policy, _service_centers
    _vehicles = _load_json("vehicles.json")
    _warranty_policy = _load_json("warranty_policy.json")
    _service_centers = _load_json("service_centers.json")
    _generate_time_slots()


def _generate_time_slots():
    """Generate available time slots for the next 7 days for each service center."""
    global _time_slots
    _time_slots = {}

    slot_times = [
        "08:00", "08:30", "09:00", "09:30", "10:00", "10:30",
        "11:00", "13:30", "14:00", "14:30", "15:00", "15:30",
        "16:00", "16:30"
    ]

    today = datetime.now().date()
    for center in _service_centers:
        for day_offset in range(1, 8):  # next 7 days
            date = today + timedelta(days=day_offset)
            # Skip Sunday (6 = Sunday)
            if date.weekday() == 6:
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

        # Atomic state transition
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
            "created_at": datetime.now().isoformat(),
            "pending_expires_at": datetime.fromtimestamp(now + 300).isoformat(),  # 5 min TTL
            "ttl_seconds": 300,
        }
        _bookings[booking_id] = booking
        return booking


def confirm_booking(booking_id: str) -> Optional[dict]:
    """
    Atomic PENDING -> CONFIRMED transition.
    Returns updated booking or None if expired / not found.
    """
    with _lock:
        booking = _bookings.get(booking_id)
        if not booking or booking["status"] != "PENDING":
            return None

        # Check TTL
        slot = _time_slots.get(
            f"SLOT_{booking['center_id']}_{booking['booking_date']}_{booking['time_slot'].replace(':', '')}"
        )
        if slot and slot["status"] == "PENDING" and slot["pending_booking_id"] == booking_id:
            elapsed = time.time() - slot["pending_since"]
            if elapsed > 300:
                # TTL expired – release
                _release_slot_internal(slot, booking)
                return None

            # Commit
            slot["status"] = "CONFIRMED"
            slot["pending_since"] = None
            booking["status"] = "CONFIRMED"
            booking["confirmed_at"] = datetime.now().isoformat()
            return booking
        return None


def get_booking(booking_id: str) -> Optional[dict]:
    return _bookings.get(booking_id)


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
    results.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    return results


def get_booking_ttl_remaining(booking_id: str) -> Optional[float]:
    """Returns remaining TTL in seconds, or None if not PENDING."""
    booking = _bookings.get(booking_id)
    if not booking or booking["status"] != "PENDING":
        return None
    slot_id = f"SLOT_{booking['center_id']}_{booking['booking_date']}_{booking['time_slot'].replace(':', '')}"
    slot = _time_slots.get(slot_id)
    if not slot or slot["status"] != "PENDING":
        return None
    elapsed = time.time() - slot["pending_since"]
    remaining = 300 - elapsed
    return max(0, remaining)


def _release_slot_internal(slot: dict, booking: dict):
    """Internal: release a slot back to AVAILABLE (must hold _lock)."""
    slot["status"] = "AVAILABLE"
    slot["pending_since"] = None
    slot["pending_booking_id"] = None
    booking["status"] = "EXPIRED"


# ─── TTL Worker: clean up expired PENDING slots ──────────────────
def _ttl_worker():
    """Background thread that checks for expired PENDING slots every 10 seconds."""
    while True:
        time.sleep(10)
        now = time.time()
        with _lock:
            for slot in _time_slots.values():
                if slot["status"] == "PENDING" and slot["pending_since"]:
                    if now - slot["pending_since"] > 300:
                        bid = slot["pending_booking_id"]
                        if bid and bid in _bookings:
                            _release_slot_internal(slot, _bookings[bid])


def start_ttl_worker():
    """Start the background TTL cleanup worker as a daemon thread."""
    worker = threading.Thread(target=_ttl_worker, daemon=True)
    worker.start()
