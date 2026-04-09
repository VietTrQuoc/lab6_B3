"""
LangGraph-based agent module for VinFast Warranty AI.

The graph keeps a strict agent -> tools -> agent loop so the backend only
returns after the final assistant answer is ready.
"""

import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import tools

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """Bạn là VinBot, chuyên viên tư vấn bảo hành xe máy điện VinFast.

Vai trò và phong cách:
- Xưng "em", gọi khách hàng là "anh/chị".
- Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng, có cấu trúc.
- Chỉ dùng dữ liệu từ tool hoặc dữ liệu đã nạp trong hệ thống.

Khả năng:
1. Tra cứu bảo hành xe và pin theo vehicle_id.
2. Giải thích chính sách bảo hành pin, linh kiện, lịch bảo dưỡng.
3. Chẩn đoán sơ bộ từ telemetry.
4. Tìm xưởng dịch vụ theo khu vực.
5. Đặt lịch kiểm tra, bảo dưỡng, sửa chữa.
6. Tra cứu lịch hẹn đã đặt.
7. Đổi lịch hẹn sang xưởng hoặc khung giờ khác.

Guardrails:
1. Không cam kết tài chính, quà tặng, đổi xe mới.
2. Không kết luận miễn phí cho lỗi vật lý khi chưa kiểm tra trực tiếp.
3. Không chốt giờ phục vụ như cam kết cứng.
4. Không xác nhận booking nếu backend chưa trả về trạng thái hợp lệ.
5. Nếu thiếu dữ liệu quan trọng, phải hỏi lại trước khi gọi tool.

Thông tin user hiện tại:
- User ID: U_VIN_001
- Tên: Nguyễn Văn An
- Đã đăng nhập, có thể truy cập tất cả xe của mình.

Khi khách chưa chọn xe cụ thể, hãy hỏi rõ xe nào trước khi tra cứu."""

TOPIC_KEYWORDS = {
    "bao hanh",
    "warranty",
    "chinh sach",
    "quyen loi",
    "pin",
    "lfp",
    "linh kien",
    "bao duong",
    "chan doan",
    "diagnostic",
    "telemetry",
    "ma loi",
    "loi",
    "sua chua",
    "xuong",
    "dich vu",
    "trung tam",
    "dat lich",
    "lich hen",
    "booking",
    "slot",
}

BOOKING_INTENT_KEYWORDS = {
    "dat lich",
    "lich hen",
    "bao duong",
    "sua chua",
    "thay pin",
    "xuong",
    "trung tam",
    "slot",
    "doi lich",
    "doi gio",
    "doi ngay",
    "doi xuong",
    "dat lai",
}

RESCHEDULE_INTENT_KEYWORDS = {
    "doi lich",
    "doi gio",
    "doi ngay",
    "doi xuong",
    "dat lai",
}

GENERIC_VEHICLE_REFERENCES = {
    "xe",
    "chiec xe",
    "xe nay",
    "xe kia",
    "xe do",
    "xe cua toi",
    "xe cua em",
    "con xe",
    "mau xe",
    "dong xe",
}

GREETING_KEYWORDS = {"xin chao", "chao", "hello", "hi", "alo", "cam on", "thank"}

CONFIRMATION_KEYWORDS = {
    "ok",
    "oke",
    "oke em",
    "dong y",
    "xac nhan",
    "tien hanh",
    "dat lich cho toi",
    "dat lich cho t",
    "dat cho toi",
    "dat cho t",
    "giu lich cho toi",
    "giu lich cho t",
    "duoc",
}

OUT_OF_SCOPE_HINTS = {
    "gia",
    "gia ban",
    "bao nhieu tien",
    "tra gop",
    "khuyen mai",
    "uu dai",
    "thiet ke",
    "tinh nang",
    "thong so",
    "toc do",
    "cong suat",
    "so sanh",
    "danh gia",
    "review",
    "thoi tiet",
    "bong da",
    "chung khoan",
    "am nhac",
    "phim",
    "nau an",
}

DATETIME_HINTS = {
    "hom nay",
    "ngay mai",
    "mai",
    "sang",
    "chieu",
    "toi nay",
    "buoi toi",
    "muon nhat",
    "som nhat",
    "cuoi cung",
    "dau tien",
    "tuan nay",
    "tuan sau",
    "cuoi tuan",
    "thu 2",
    "thu 3",
    "thu 4",
    "thu 5",
    "thu 6",
    "thu 7",
    "chu nhat",
}

WEEKDAY_KEYWORDS = {
    "thu 2": 0,
    "thu 3": 1,
    "thu 4": 2,
    "thu 5": 3,
    "thu 6": 4,
    "thu 7": 5,
    "chu nhat": 6,
}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_log: list[dict[str, Any]]
    booking: dict | None


_BOUND_MODEL = None
_GRAPH = None


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_topic(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in TOPIC_KEYWORDS)


def _contains_booking_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in BOOKING_INTENT_KEYWORDS)


def _contains_reschedule_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in RESCHEDULE_INTENT_KEYWORDS)


def _is_greeting_or_social(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in GREETING_KEYWORDS)


def _is_confirmation_message(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in CONFIRMATION_KEYWORDS)


def _contains_out_of_scope_hint(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in OUT_OF_SCOPE_HINTS)


def _contains_datetime_preference(text: str) -> bool:
    normalized = _normalize_text(text)

    if any(hint in normalized for hint in DATETIME_HINTS):
        return True

    if re.search(r"\b\d{1,2}[:hg]\d{0,2}\b", normalized):
        return True

    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized):
        return True

    return False


def _contains_service_location(text: str, service_centers: list[dict]) -> bool:
    normalized = _normalize_text(text)
    generic_location_hints = {
        "ha noi",
        "hanoi",
        "tp hcm",
        "tphcm",
        "ho chi minh",
        "da nang",
        "hai phong",
        "can tho",
        "dong nai",
        "gan toi",
        "gan nha",
        "khu vuc",
    }
    if any(hint in normalized for hint in generic_location_hints):
        return True

    for center in service_centers:
        values = {center["city"], center["district"]}
        if any(_normalize_text(value) in normalized for value in values):
            return True

    return False


def _get_center_aliases(center: dict) -> set[str]:
    center_id = _normalize_text(center["id"])
    center_name = _normalize_text(center["name"])
    aliases = {center_id, center_name}
    if center_name.startswith("vinfast "):
        aliases.add(center_name.replace("vinfast ", "vf ", 1))
    if center_name.startswith("vf "):
        aliases.add(center_name.replace("vf ", "vinfast ", 1))
    return {alias for alias in aliases if alias}


def _contains_specific_service_center(text: str, service_centers: list[dict]) -> bool:
    normalized = _normalize_text(text)
    for center in service_centers:
        if any(alias in normalized for alias in _get_center_aliases(center)):
            return True
    return False


def _find_vehicle_reference(text: str, vehicles: list[dict], selected_vehicle_id: str | None = None) -> dict | None:
    normalized = _normalize_text(text)

    for vehicle in vehicles:
        candidates = {vehicle["id"], vehicle["model"], vehicle["vin"]}
        if any(_normalize_text(candidate) in normalized for candidate in candidates):
            return vehicle

    if selected_vehicle_id and any(ref in normalized for ref in GENERIC_VEHICLE_REFERENCES):
        return next((vehicle for vehicle in vehicles if vehicle["id"] == selected_vehicle_id), None)

    return None


def _get_recent_topic_context(messages: list[dict], lookback: int = 4) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_topic(content) for content in text_messages)


def _get_recent_booking_context(messages: list[dict], lookback: int = 6) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_booking_intent(content) for content in text_messages)


def _get_recent_reschedule_context(messages: list[dict], lookback: int = 6) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_reschedule_intent(content) for content in text_messages)


def _history_contains_service_location(messages: list[dict], service_centers: list[dict], lookback: int = 6) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_service_location(content, service_centers) for content in text_messages)


def _history_contains_specific_service_center(
    messages: list[dict],
    service_centers: list[dict],
    lookback: int = 6,
) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_specific_service_center(content, service_centers) for content in text_messages)


def _history_contains_datetime_preference(messages: list[dict], lookback: int = 6) -> bool:
    text_messages = []
    for message in reversed(messages[:-1]):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break
    return any(_contains_datetime_preference(content) for content in text_messages)


def _resolve_relative_dates(text: str, base_date: date | None = None) -> list[tuple[str, str]]:
    normalized = _normalize_text(text)
    reference_date = base_date or datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
    resolutions: list[tuple[str, str]] = []

    if "hom nay" in normalized:
        resolutions.append(("hôm nay", reference_date.isoformat()))

    if "ngay mai" in normalized or re.search(r"\bmai\b", normalized):
        resolutions.append(("ngày mai", (reference_date + timedelta(days=1)).isoformat()))

    start_of_week = reference_date - timedelta(days=reference_date.weekday())

    for keyword, weekday in WEEKDAY_KEYWORDS.items():
        if f"{keyword} tuan nay" in normalized:
            target_date = start_of_week + timedelta(days=weekday)
            resolutions.append((f"{keyword} tuần này", target_date.isoformat()))
        if f"{keyword} tuan sau" in normalized:
            target_date = start_of_week + timedelta(days=7 + weekday)
            resolutions.append((f"{keyword} tuần sau", target_date.isoformat()))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in resolutions:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def _build_relative_date_context(messages: list[dict]) -> str:
    if not messages:
        return ""

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return ""

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return ""

    resolved_dates = _resolve_relative_dates(latest_content)
    if not resolved_dates:
        return ""

    lines = [
        f"- '{phrase}' = {absolute_date}"
        for phrase, absolute_date in resolved_dates
    ]
    return (
        "\n\n## Relative date resolution for the latest user message\n"
        + "\n".join(lines)
        + "\nUse these exact dates when calling tools or answering. Do not invent a different date."
    )


def _build_runtime_context() -> str:
    current_dt = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    return (
        "\n\n## Runtime context"
        f"\nCurrent datetime: {current_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f"\nCurrent date: {current_dt.strftime('%Y-%m-%d')}"
        "\nWhen the customer mentions today, tomorrow, warranty remaining days, or appointment dates,"
        " use this runtime date as the source of truth."
    )


def _extract_date_from_text(text: str) -> str | None:
    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso_match:
        return iso_match.group(1)

    slash_match = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
    if slash_match:
        day, month, year = slash_match.groups()
        return f"{year}-{month}-{day}"

    return None


def _extract_time_from_text(text: str) -> str | None:
    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if time_match:
        hour, minute = time_match.group(1).split(":")
        return f"{int(hour):02d}:{minute}"

    shorthand_match = re.search(r"\b(\d{1,2})\s*[hg](\d{2})?\b", _normalize_text(text))
    if not shorthand_match:
        return None

    hour = int(shorthand_match.group(1))
    minute = shorthand_match.group(2) or "00"
    return f"{hour:02d}:{minute}"


def _extract_booking_id_from_text(text: str) -> str | None:
    booking_match = re.search(r"\b(BK_[A-Z0-9]+)\b", text, re.IGNORECASE)
    if not booking_match:
        return None
    return booking_match.group(1).upper()


def _extract_recent_booking_id(messages: list[dict], lookback: int = 6) -> str | None:
    text_messages = []
    for message in reversed(messages[:-1]):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break

    for content in text_messages:
        booking_id = _extract_booking_id_from_text(content)
        if booking_id:
            return booking_id
    return None


def _infer_target_booking_for_reschedule(
    messages: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None,
):
    from app import data as _data

    vehicle_id = selected_vehicle_id

    if not vehicle_id:
        for message in reversed(messages):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            vehicle = _find_vehicle_reference(content, vehicles, selected_vehicle_id)
            if vehicle:
                vehicle_id = vehicle["id"]
                break

    if vehicle_id:
        active_bookings = [
            booking
            for booking in _data.get_user_bookings(user_id="U_VIN_001", vehicle_id=vehicle_id)
            if booking.get("status") in {"PENDING", "CONFIRMED"}
        ]
        if len(active_bookings) == 1:
            return active_bookings[0]

    active_bookings = [
        booking
        for booking in _data.get_user_bookings(user_id="U_VIN_001", vehicle_id=None)
        if booking.get("status") in {"PENDING", "CONFIRMED"}
    ]
    if len(active_bookings) == 1:
        return active_bookings[0]

    return None


def _resolve_center_from_text(text: str, service_centers: list[dict]) -> dict | None:
    normalized = _normalize_text(text)
    for center in service_centers:
        if any(alias in normalized for alias in _get_center_aliases(center)):
            return center
    return None


def _infer_service_type(messages: list[dict]) -> str:
    service_patterns = [
        ("bao duong dinh ky", "bảo dưỡng định kỳ"),
        ("kiem tra pin", "kiểm tra pin"),
        ("bao hanh pin", "bảo hành pin"),
        ("thay pin", "thay pin"),
        ("sua chua", "sửa chữa"),
        ("bao hanh", "bảo hành"),
        ("kiem tra", "kiểm tra"),
        ("bao duong", "bảo dưỡng"),
    ]

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = _normalize_text(content)
        for pattern, label in service_patterns:
            if pattern in normalized:
                return label

    return "bảo dưỡng"


def _extract_note_from_confirmation(text: str) -> str:
    normalized = _normalize_text(text)
    if "ghi chu" not in normalized:
        return ""

    note_match = re.search(r"ghi chú[:\-\s]*(.+)$", text, re.IGNORECASE)
    if note_match:
        return note_match.group(1).strip()
    return ""


def _extract_recent_booking_proposal(
    messages: list[dict],
    service_centers: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None,
) -> dict | None:
    from app import data as _data

    assistant_message = None
    for message in reversed(messages[:-1]):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            assistant_message = message["content"]
            break

    if not assistant_message:
        return None

    normalized_assistant = _normalize_text(assistant_message)
    if (
        "dat lich" not in normalized_assistant
        and "lich hen" not in normalized_assistant
        and not _contains_reschedule_intent(assistant_message)
    ):
        return None

    center = _resolve_center_from_text(assistant_message, service_centers)
    booking_date = _extract_date_from_text(assistant_message)
    time_slot = _extract_time_from_text(assistant_message)
    booking_id = _extract_booking_id_from_text(assistant_message) or _extract_recent_booking_id(messages)
    existing_booking = _data.get_booking(booking_id) if booking_id else None
    reschedule_requested = _contains_reschedule_intent(assistant_message) or _get_recent_reschedule_context(messages)
    if not existing_booking and reschedule_requested:
        existing_booking = _infer_target_booking_for_reschedule(messages, vehicles, selected_vehicle_id)
        if existing_booking:
            booking_id = existing_booking["booking_id"]
    vehicle = _find_vehicle_reference(assistant_message, vehicles, selected_vehicle_id)
    if not vehicle and selected_vehicle_id:
        vehicle = next((item for item in vehicles if item["id"] == selected_vehicle_id), None)
    if not vehicle and existing_booking:
        vehicle = next(
            (item for item in vehicles if item["id"] == existing_booking.get("vehicle_id")),
            None,
        )

    if not center or not booking_date or not time_slot:
        return None

    is_reschedule = bool(existing_booking) and reschedule_requested
    if not is_reschedule and not vehicle:
        return None

    proposal = {
        "vehicle_id": (vehicle or {}).get("id") or (existing_booking or {}).get("vehicle_id"),
        "center_id": center["id"],
        "center_name": center["name"],
        "booking_date": booking_date,
        "time_slot": time_slot,
        "service_type": _infer_service_type(messages),
    }
    if is_reschedule:
        proposal["booking_id"] = existing_booking["booking_id"]
        proposal["action"] = "reschedule"
    else:
        proposal["action"] = "create"

    return proposal


def _extract_recent_slot_selection_context(
    messages: list[dict],
    service_centers: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None,
) -> dict | None:
    from app import data as _data

    recent_messages: list[dict] = []
    assistant_message = None
    for message in reversed(messages[:-1]):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        recent_messages.append(message)
        if message.get("role") == "assistant" and assistant_message is None:
            normalized = _normalize_text(content)
            if "khung gio" in normalized or "slot" in normalized:
                assistant_message = content
        if len(recent_messages) >= 8 and assistant_message:
            break

    if not assistant_message:
        return None

    normalized_assistant = _normalize_text(assistant_message)
    if (
        "chon khung gio" not in normalized_assistant
        and "khung gio nao" not in normalized_assistant
        and "khung gio con trong" not in normalized_assistant
        and "khung gio kha dung" not in normalized_assistant
    ):
        return None

    search_texts = [assistant_message] + [msg["content"] for msg in recent_messages if msg["content"] != assistant_message]
    booking_date = None
    center = None
    vehicle = None

    for content in search_texts:
        if not booking_date:
            booking_date = _extract_date_from_text(content)
        if not center:
            center = _resolve_center_from_text(content, service_centers)
        if not vehicle:
            vehicle = _find_vehicle_reference(content, vehicles, selected_vehicle_id)
        if booking_date and center and vehicle:
            break

    booking_id = _extract_recent_booking_id(messages)
    existing_booking = _data.get_booking(booking_id) if booking_id else None
    reschedule_requested = _contains_reschedule_intent(assistant_message) or _get_recent_reschedule_context(messages)
    if not existing_booking and reschedule_requested:
        existing_booking = _infer_target_booking_for_reschedule(messages, vehicles, selected_vehicle_id)
        if existing_booking:
            booking_id = existing_booking["booking_id"]
    if not vehicle and selected_vehicle_id:
        vehicle = next((item for item in vehicles if item["id"] == selected_vehicle_id), None)
    if not vehicle and existing_booking:
        vehicle = next(
            (item for item in vehicles if item["id"] == existing_booking.get("vehicle_id")),
            None,
        )

    if not booking_date or not center:
        return None

    is_reschedule = bool(existing_booking) and reschedule_requested
    if not is_reschedule and not vehicle:
        return None

    context = {
        "vehicle_id": (vehicle or {}).get("id") or (existing_booking or {}).get("vehicle_id"),
        "center_id": center["id"],
        "center_name": center["name"],
        "booking_date": booking_date,
        "service_type": _infer_service_type(messages),
    }
    if is_reschedule:
        context["booking_id"] = existing_booking["booking_id"]
        context["action"] = "reschedule"
    else:
        context["action"] = "create"

    return context


def _handle_slot_selection_choice(
    messages: list[dict],
    service_centers: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None,
) -> dict | None:
    if len(messages) < 2:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return None

    selected_time = _extract_time_from_text(latest_content)
    if not selected_time:
        return None

    context = _extract_recent_slot_selection_context(
        messages,
        service_centers,
        vehicles,
        selected_vehicle_id,
    )
    if not context:
        return None

    slots_result = tools.get_available_time_slots(context["center_id"], context["booking_date"])
    available_slots = slots_result.get("available_dates", {}).get(context["booking_date"], [])
    matched_slot = next((slot for slot in available_slots if slot.get("time") == selected_time), None)
    tool_calls_log = [
        {
            "tool": "get_available_time_slots",
            "arguments": {
                "center_id": context["center_id"],
                "date_str": context["booking_date"],
            },
        }
    ]

    if not matched_slot:
        remaining_times = [slot.get("time") for slot in available_slots if slot.get("time")]
        remaining_lines = "\n".join(f"- {slot_time}" for slot_time in remaining_times[:12])
        reply = (
            f"Khung giờ {selected_time} vào ngày {context['booking_date']} tại {context['center_name']} hiện không còn khả dụng."
        )
        if remaining_lines:
            reply += (
                f"\n\nAnh/chị có muốn chọn khung giờ khác trong ngày {context['booking_date']} không? "
                f"Dưới đây là các khung giờ còn trống:\n{remaining_lines}"
            )
        return {
            "reply": reply,
            "tool_calls_log": tool_calls_log,
            "booking": None,
        }

    if context.get("action") == "reschedule":
        booking_result = tools.reschedule_appointment(
            booking_id=context["booking_id"],
            center_id=context["center_id"],
            slot_id=matched_slot["slot_id"],
            service_type=context["service_type"],
            note="",
        )
        tool_calls_log.append(
            {
                "tool": "reschedule_appointment",
                "arguments": {
                    "booking_id": context["booking_id"],
                    "center_id": context["center_id"],
                    "slot_id": matched_slot["slot_id"],
                    "service_type": context["service_type"],
                    "note": "",
                },
            }
        )
    else:
        booking_result = tools.create_appointment(
            vehicle_id=context["vehicle_id"],
            center_id=context["center_id"],
            slot_id=matched_slot["slot_id"],
            service_type=context["service_type"],
            ai_diagnosis_log="",
            note="",
        )
        tool_calls_log.append(
            {
                "tool": "create_appointment",
                "arguments": {
                    "vehicle_id": context["vehicle_id"],
                    "center_id": context["center_id"],
                    "slot_id": matched_slot["slot_id"],
                    "service_type": context["service_type"],
                    "ai_diagnosis_log": "",
                    "note": "",
                },
            }
        )

    if not booking_result.get("success"):
        return {
            "reply": booking_result.get("error", "Khong the xu ly khung gio anh/chị vua chon."),
            "tool_calls_log": tool_calls_log,
            "booking": None,
        }

    booking = booking_result["booking"]
    if context.get("action") == "reschedule":
        return {
            "reply": (
                f"Em da cap nhat lich hen {context['service_type']} cua anh/chị sang {booking['center_name']} "
                f"vao luc {booking['time_slot']} ngay {booking['booking_date']}."
            ),
            "tool_calls_log": tool_calls_log,
            "booking": booking if booking.get("status") == "PENDING" else None,
        }

    return {
        "reply": (
            f"Em da giu cho lich hen {context['service_type']} cho anh/chị tai {booking['center_name']} "
            f"vao luc {booking['time_slot']} ngay {booking['booking_date']}. Vui long bam xac nhan trong 5 phut de hoan tat."
        ),
        "tool_calls_log": tool_calls_log,
        "booking": booking,
    }


def _handle_booking_confirmation(
    messages: list[dict],
    service_centers: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None,
) -> dict | None:
    if len(messages) < 2:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not _is_confirmation_message(latest_content):
        return None

    proposal = _extract_recent_booking_proposal(messages, service_centers, vehicles, selected_vehicle_id)
    if not proposal:
        return None

    slots_result = tools.get_available_time_slots(proposal["center_id"], proposal["booking_date"])
    available_slots = slots_result.get("available_dates", {}).get(proposal["booking_date"], [])
    matched_slot = next((slot for slot in available_slots if slot.get("time") == proposal["time_slot"]), None)

    tool_calls_log = [
        {
            "tool": "get_available_time_slots",
            "arguments": {
                "center_id": proposal["center_id"],
                "date_str": proposal["booking_date"],
            },
        }
    ]

    if not matched_slot:
        return {
            "reply": (
                f"Khung giờ {proposal['time_slot']} ngày {proposal['booking_date']} tại {proposal['center_name']} hiện không còn trống. "
                "Anh/chị vui lòng chọn một khung giờ khác trong danh sách khả dụng nhé."
            ),
            "tool_calls_log": tool_calls_log,
            "booking": None,
        }

    note = _extract_note_from_confirmation(latest_content)
    if proposal.get("action") == "reschedule":
        booking_result = tools.reschedule_appointment(
            booking_id=proposal["booking_id"],
            center_id=proposal["center_id"],
            slot_id=matched_slot["slot_id"],
            service_type=proposal["service_type"],
            note=note,
        )
        tool_calls_log.append(
            {
                "tool": "reschedule_appointment",
                "arguments": {
                    "booking_id": proposal["booking_id"],
                    "center_id": proposal["center_id"],
                    "slot_id": matched_slot["slot_id"],
                    "service_type": proposal["service_type"],
                    "note": note,
                },
            }
        )
    else:
        booking_result = tools.create_appointment(
            vehicle_id=proposal["vehicle_id"],
            center_id=proposal["center_id"],
            slot_id=matched_slot["slot_id"],
            service_type=proposal["service_type"],
            ai_diagnosis_log="",
            note=note,
        )
        tool_calls_log.append(
            {
                "tool": "create_appointment",
                "arguments": {
                    "vehicle_id": proposal["vehicle_id"],
                    "center_id": proposal["center_id"],
                    "slot_id": matched_slot["slot_id"],
                    "service_type": proposal["service_type"],
                    "ai_diagnosis_log": "",
                    "note": note,
                },
            }
        )

    if not booking_result.get("success"):
        return {
            "reply": booking_result.get("error", "Không thể giữ chỗ cho lịch hẹn này."),
            "tool_calls_log": tool_calls_log,
            "booking": None,
        }

    booking = booking_result["booking"]
    if proposal.get("action") == "reschedule":
        return {
            "reply": (
                f"Em da cap nhat lich hen {proposal['service_type']} cua anh/chị sang {booking['center_name']} "
                f"vao luc {booking['time_slot']} ngay {booking['booking_date']}."
            ),
            "tool_calls_log": tool_calls_log,
            "booking": booking if booking.get("status") == "PENDING" else None,
        }
    return {
        "reply": (
            f"Em đã giữ chỗ lịch hẹn {proposal['service_type']} cho anh/chị tại {booking['center_name']} "
            f"vào lúc {booking['time_slot']} ngày {booking['booking_date']}. Vui lòng bấm xác nhận trong 5 phút để hoàn tất."
        ),
        "tool_calls_log": tool_calls_log,
        "booking": booking,
    }


def _build_topic_clarification(vehicle: dict | None) -> str:
    vehicle_label = vehicle["model"] if vehicle else "xe nay"
    return (
        f"Anh/chị đang muốn em hỗ trợ gì với {vehicle_label}? "
        "Em có thể hỗ trợ tra cứu bảo hành, giải thích chính sách, chẩn đoán sơ bộ hoặc hỗ trợ đặt lịch kiểm tra."
    )


def _should_clarify_topic(
    messages: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None = None,
) -> dict | None:
    if not messages:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return None

    if _contains_topic(latest_content) or _get_recent_topic_context(messages):
        return None

    vehicle = _find_vehicle_reference(latest_content, vehicles, selected_vehicle_id)
    if not vehicle:
        return None

    return {"reply": _build_topic_clarification(vehicle), "tool_calls_log": [], "booking": None}


def _build_booking_clarification(needs_location: bool = False, needs_datetime: bool = False) -> str:
    if needs_location:
        return (
            "Anh/chị đang ở khu vực hoặc thành phố nào ạ? Em sẽ dựa vào vị trí hiện tại để gợi ý các xưởng VinFast "
            "gần nhất trước, rồi mình mới chọn xưởng và khung giờ phù hợp."
        )
    if needs_datetime:
        return (
            "Anh/chị muốn đặt vào ngày hoặc khung giờ nào ạ? Ví dụ: sáng mai, chiều thứ 6, hoặc 10/04 lúc 09:00."
        )
    return (
        "Anh/chị cho em biết thêm vị trí hiện tại hoặc xưởng muốn đến, rồi em sẽ hỗ trợ tìm lịch phù hợp."
    )


def _should_clarify_booking_details(
    messages: list[dict],
    vehicles: list[dict],
    service_centers: list[dict],
    selected_vehicle_id: str | None = None,
) -> dict | None:
    if not messages:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return None

    if not (_contains_booking_intent(latest_content) or _get_recent_booking_context(messages)):
        return None

    active_vehicle = selected_vehicle_id or (
        _find_vehicle_reference(latest_content, vehicles, selected_vehicle_id) or {}
    ).get("id")
    if not active_vehicle:
        return None

    latest_has_location = _contains_service_location(latest_content, service_centers)
    latest_has_specific_center = _contains_specific_service_center(latest_content, service_centers)
    latest_has_datetime = _contains_datetime_preference(latest_content)

    has_location = latest_has_location or _history_contains_service_location(messages, service_centers)
    has_specific_center = latest_has_specific_center or _history_contains_specific_service_center(
        messages,
        service_centers,
    )
    has_location = has_location or has_specific_center
    has_datetime = latest_has_datetime or _history_contains_datetime_preference(messages)

    if latest_has_location and not latest_has_specific_center:
        has_specific_center = False

    if not has_location:
        return {
            "reply": _build_booking_clarification(needs_location=True),
            "tool_calls_log": [],
            "booking": None,
        }

    if latest_has_location and not latest_has_specific_center and not latest_has_datetime:
        return None

    if latest_has_specific_center and not latest_has_datetime:
        has_datetime = False

    if has_specific_center and not has_datetime:
        return {
            "reply": _build_booking_clarification(needs_datetime=True),
            "tool_calls_log": [],
            "booking": None,
        }

    return None


def _should_reject_out_of_scope(
    messages: list[dict],
    vehicles: list[dict],
    selected_vehicle_id: str | None = None,
) -> dict | None:
    if not messages:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return None

    if _contains_topic(latest_content) or _get_recent_topic_context(messages):
        return None

    if _find_vehicle_reference(latest_content, vehicles, selected_vehicle_id):
        if _contains_out_of_scope_hint(latest_content):
            return {
                "reply": (
                    "Em chỉ hỗ trợ các vấn đề về bảo hành, chẩn đoán, lịch hẹn dịch vụ và dữ liệu xe VinFast đã có "
                    "trong hệ thống. Nếu anh/chị cần, em có thể hỗ trợ tra cứu bảo hành hoặc đặt lịch kiểm tra."
                ),
                "tool_calls_log": [],
                "booking": None,
            }
        return None

    if _is_greeting_or_social(latest_content):
        return None

    return {
        "reply": (
            "Em chỉ hỗ trợ các nội dung liên quan đến bảo hành xe VinFast từ dữ liệu đã nạp trong hệ thống, gồm "
            "tra cứu bảo hành, giải thích chính sách, chẩn đoán sơ bộ, tìm xưởng dịch vụ và đặt lịch."
        ),
        "tool_calls_log": [],
        "booking": None,
    }


def _build_system_message(selected_vehicle_id: str | None = None) -> str:
    from app import data as _data

    system_msg = SYSTEM_PROMPT + _build_runtime_context()

    all_vehicles = _data.get_all_vehicles()
    all_service_centers = _data.get_all_service_centers()
    if all_vehicles:
        vehicle_lines = []
        for vehicle in all_vehicles:
            vehicle_lines.append(
                f"  - **{vehicle['id']}**: {vehicle['model']} | VIN: {vehicle['vin']} | Mau: {vehicle['color']} | "
                f"Mua: {vehicle['purchase_date']} | ODO: {vehicle['telemetry']['odo_km']:,} km | "
                f"SOH pin: {vehicle['telemetry']['battery_soh_percent']}%"
            )

        system_msg += "\n\n## Danh sach xe cua khach hang\n" + "\n".join(vehicle_lines)
        system_msg += (
            "\n\n**Luu y:** Khi khach hoi ve xe, hay dung dung vehicle_id o tren. Neu khach chua chon xe cu the "
            "va co nhieu xe, hay hoi khach muon tra cuu xe nao."
        )

    if all_service_centers:
        center_lines = []
        for center in all_service_centers:
            center_lines.append(
                f"  - **{center['id']}**: {center['name']} | {center['district']}, {center['city']} | "
                f"Gio lam viec: {center['working_hours']}"
            )
        system_msg += "\n\n## Danh sach xuong dich vu\n" + "\n".join(center_lines)

    system_msg += (
        "\n\n## Clarification rule\n"
        "If the latest customer message only identifies a vehicle but does not state the support topic, ask what "
        "they want help with first and do not call tools yet."
    )
    system_msg += (
        "\n\n## Scope rule\n"
        "Only answer questions about warranty, diagnostics, service centers, appointments, and the vehicle data "
        "loaded into this system. Refuse unrelated requests such as price, promotions, entertainment, weather, or "
        "general knowledge."
    )
    system_msg += (
        "\n\n## Booking rule\n"
        "When the customer wants maintenance or booking support, ask for the customer's current location first if "
        "it is missing so you can suggest nearby VinFast service centers. If the customer provides only a city, "
        "district, or area, call find_nearest_service_center immediately and list the candidate centers instead of "
        "asking for time yet. Only ask for date or time after the customer chooses a specific center, even if there "
        "is only one center in the result. Never choose a service center, city, date, or time on the customer's behalf. If the customer changes the location or "
        "service center, ask them to confirm the date/time again instead of reusing an older slot choice. Always "
        "keep the exact requested service_type, such as bao duong dinh ky, kiem tra pin, sua loi phanh, or bao hanh pin."
    )
    system_msg += (
        "\n\n## Execution rule\n"
        "Do not say you will wait, check later, or process in the background. If enough information is available, "
        "call the needed tool immediately in the same turn and then answer with the actual result. To create an "
        "appointment, always call get_available_time_slots in the same turn first so you can use a real slot_id "
        "from the current availability before calling create_appointment. To reschedule, first identify the target "
        "booking from lookup_my_bookings when needed, then call get_available_time_slots for the target center/date, "
        "and only then call reschedule_appointment with a real slot_id."
    )

    if selected_vehicle_id:
        system_msg += (
            "\n\n## Xe dang duoc chon tren giao dien\n"
            f"Vehicle ID: {selected_vehicle_id} - Khi khach hoi ma khong chi ro xe, hay dung xe nay."
        )

    return system_msg


def _coerce_langchain_messages(messages: list[dict], system_message: str) -> list[BaseMessage]:
    converted: list[BaseMessage] = [SystemMessage(content=system_message)]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _get_bound_model():
    global _BOUND_MODEL
    if _BOUND_MODEL is None:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL, temperature=0)
        _BOUND_MODEL = llm.bind_tools(
            tools.TOOL_DEFINITIONS,
            tool_choice="auto",
            strict=True,
            parallel_tool_calls=False,
        )
    return _BOUND_MODEL


def _agent_node(state: AgentState) -> AgentState:
    model = _get_bound_model()
    response = model.invoke(state["messages"])

    tool_logs = list(state.get("tool_calls_log", []))
    for tool_call in getattr(response, "tool_calls", []) or []:
        tool_logs.append(
            {
                "tool": tool_call.get("name"),
                "arguments": tool_call.get("args", {}),
                "id": tool_call.get("id"),
            }
        )

    return {"messages": [response], "tool_calls_log": tool_logs, "booking": state.get("booking")}


def _tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {"messages": [], "tool_calls_log": state.get("tool_calls_log", []), "booking": state.get("booking")}

    booking_info = state.get("booking")
    tool_messages: list[ToolMessage] = []

    for tool_call in getattr(last_message, "tool_calls", []) or []:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        result_str = tools.execute_tool(tool_name, tool_args)

        if tool_name in {"create_appointment", "reschedule_appointment"}:
            result_obj = json.loads(result_str)
            if result_obj.get("success") and result_obj.get("booking", {}).get("status") == "PENDING":
                booking_info = result_obj.get("booking")

        tool_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": tool_messages, "tool_calls_log": state.get("tool_calls_log", []), "booking": booking_info}


def _route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def _get_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", _agent_node)
        workflow.add_node("tools", _tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", _route_after_agent)
        workflow.add_edge("tools", "agent")
        _GRAPH = workflow.compile()
    return _GRAPH


def _extract_final_reply(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = _extract_text_content(message.content)
            if content.strip():
                return content
    return ""


def chat(messages: list[dict], selected_vehicle_id: str | None = None) -> dict:
    from app import data as _data

    all_vehicles = _data.get_all_vehicles()
    all_service_centers = _data.get_all_service_centers()

    rejection_response = _should_reject_out_of_scope(messages, all_vehicles, selected_vehicle_id)
    if rejection_response:
        return rejection_response

    booking_confirmation_response = _handle_booking_confirmation(
        messages,
        all_service_centers,
        all_vehicles,
        selected_vehicle_id,
    )
    if booking_confirmation_response:
        return booking_confirmation_response

    slot_selection_response = _handle_slot_selection_choice(
        messages,
        all_service_centers,
        all_vehicles,
        selected_vehicle_id,
    )
    if slot_selection_response:
        return slot_selection_response

    booking_clarification_response = _should_clarify_booking_details(
        messages,
        all_vehicles,
        all_service_centers,
        selected_vehicle_id,
    )
    if booking_clarification_response:
        return booking_clarification_response

    clarification_response = _should_clarify_topic(messages, all_vehicles, selected_vehicle_id)
    if clarification_response:
        return clarification_response

    initial_state: AgentState = {
        "messages": _coerce_langchain_messages(
            messages,
            _build_system_message(selected_vehicle_id) + _build_relative_date_context(messages),
        ),
        "tool_calls_log": [],
        "booking": None,
    }

    try:
        graph = _get_agent_graph()
        result = graph.invoke(initial_state, config={"recursion_limit": 8})

        reply = _extract_final_reply(result["messages"])
        if not reply:
            reply = "Xin lỗi anh/chị, em chưa tổng hợp được câu trả lời phù hợp. Anh/chị vui lòng thử lại."

        return {
            "reply": reply,
            "tool_calls_log": result.get("tool_calls_log", []),
            "booking": result.get("booking"),
        }
    except GraphRecursionError:
        return {
            "reply": (
                "Xin lỗi anh/chị, em đang bị lặp quá nhiều bước xử lý nên chưa thể hoàn tất yêu cầu này. "
                "Anh/chị vui lòng nêu lại ngắn gọn hơn hoặc cung cấp thêm thông tin còn thiếu."
            ),
            "tool_calls_log": [],
            "booking": None,
        }
    except Exception as exc:
        print(f"[AGENT ERROR] {exc}")
        return {
            "reply": (
                "Xin lỗi anh/chị, hệ thống đang gặp sự cố kết nối tạm thời. "
                "Anh/chị vui lòng thử lại sau giây lát hoặc liên hệ hotline **1900 23 23 89** để được hỗ trợ."
            ),
            "tool_calls_log": [],
            "booking": None,
        }
