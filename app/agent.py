"""
OpenAI Agent module for VinFast Warranty AI.

Implements function calling with tool definitions, system prompt with guardrails,
and the agent loop that handles multi-turn tool calls.
"""

import json
import os
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI
from dotenv import load_dotenv
from app import tools

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─── System Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là **VinBot**, chuyên viên tư vấn bảo hành xe máy điện VinFast. Bạn hỗ trợ khách hàng qua kênh chat trực tuyến.

## Vai trò và phong cách
- Xưng "em", gọi khách hàng là "anh/chị".
- Ngôn ngữ tiếng Việt, lịch sự, chuyên nghiệp nhưng thân thiện.
- Trả lời ngắn gọn, rõ ràng, có cấu trúc (dùng bullet points, bảng khi phù hợp).
- Luôn dựa trên dữ liệu thực từ công cụ (tool), KHÔNG bịa ra thông tin.

## Khả năng
1. **Tra cứu bảo hành**: Kiểm tra tình trạng bảo hành xe và pin theo vehicle_id.
2. **Giải thích chính sách**: Giải thích quyền lợi bảo hành pin LFP, linh kiện, lịch bảo dưỡng.
3. **Chẩn đoán telemetry**: Phân tích ODO, SOH pin, chu kỳ sạc, nhiệt độ, áp suất lốp, mã lỗi.
4. **Tìm xưởng dịch vụ**: Gợi ý xưởng gần nhất theo thành phố.
5. **Đặt lịch hẹn**: Tạo lịch kiểm tra/bảo dưỡng (giữ slot 5 phút, cần xác nhận).
6. **Tra cứu lịch hẹn**: Xem lại các lịch hẹn đã đặt và trạng thái hiện tại.

## Quy trình đặt lịch
- Khi đặt lịch, trước tiên gọi `get_available_time_slots` để lấy slot trống.
- Sau đó gọi `create_appointment` để giữ slot (PENDING, TTL 5 phút).
- Thông báo cho khách hàng: "Em sẽ giữ chỗ này cho anh/chị trong 5 phút."
- Khách hàng cần XÁC NHẬN trước khi hết thời gian.
- Nếu khách chưa sẵn sàng, gợi ý chọn slot khác hoặc liên hệ hotline.

## ⛔ GUARDRAILS — TUYỆT ĐỐI TUÂN THỦ
1. **KHÔNG cam kết tài chính**: Không hứa hoàn tiền, đền bù, tặng quà, đổi xe mới.
2. **KHÔNG xác nhận miễn phí cho lỗi vật lý**: Các lỗi cần kiểm tra trực tiếp → gợi ý đến xưởng.
3. **KHÔNG chốt giờ phục vụ như cam kết cứng**: Chỉ ghi nhận lịch hẹn, thời gian thực tế có thể thay đổi.
4. **KHÔNG chốt lịch khi chưa có xác nhận từ backend**: Chỉ xác nhận booking khi tool trả về status CONFIRMED.
5. **Fallback**: Với các trường hợp phức tạp, gợi ý chuyển sang ticket kỹ thuật hoặc liên hệ CSKH hotline 1900 23 23 89.

## Thông tin user hiện tại
- User ID: U_VIN_001
- Tên: Nguyễn Văn An
- Đã đăng nhập, có thể truy cập tất cả xe của mình.

Khi khách CHƯA chọn xe cụ thể, hãy hỏi rõ xe nào trước khi tra cứu."""

# ─── Tool Definitions for OpenAI ─────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_warranty_status",
            "description": "Tra cứu tình trạng bảo hành (xe và pin) theo vehicle_id. Trả về thời hạn, ngày mua, số ngày còn lại.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_id": {
                        "type": "string",
                        "description": "Mã xe cần tra cứu, ví dụ: V001, V002, ..."
                    }
                },
                "required": ["vehicle_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_warranty_policy",
            "description": "Giải thích quyền lợi bảo hành theo danh mục: pin LFP, linh kiện/xe, lịch bảo dưỡng, hoặc tổng quát.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["pin", "linh_kien", "bao_duong", "tong_quat"],
                        "description": "Danh mục cần giải thích: 'pin' (pin LFP), 'linh_kien' (xe & linh kiện), 'bao_duong' (lịch bảo dưỡng), 'tong_quat' (tất cả)"
                    }
                },
                "required": ["category"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_telemetry",
            "description": "Chẩn đoán sơ bộ xe từ dữ liệu telemetry: ODO, SOH pin, số lần sạc, nhiệt độ, áp suất lốp, mã lỗi. Trả về danh sách vấn đề và khuyến nghị.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_id": {
                        "type": "string",
                        "description": "Mã xe cần chẩn đoán"
                    }
                },
                "required": ["vehicle_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearest_service_center",
            "description": "Tìm xưởng dịch vụ VinFast gần nhất theo tên thành phố. Trả về danh sách xưởng với địa chỉ, số điện thoại, dịch vụ.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Tên thành phố, ví dụ: Hà Nội, TP.HCM, Đà Nẵng"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_time_slots",
            "description": "Lấy danh sách khung giờ còn trống cho một xưởng dịch vụ, có thể lọc theo ngày cụ thể.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "center_id": {
                        "type": "string",
                        "description": "Mã xưởng dịch vụ, ví dụ: SC001"
                    },
                    "date_str": {
                        "type": ["string", "null"],
                        "description": "Ngày cần xem (YYYY-MM-DD). Null = xem tất cả ngày."
                    }
                },
                "required": ["center_id", "date_str"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": "Tạo lịch hẹn kiểm tra/bảo dưỡng xe. Backend sẽ giữ slot trong 5 phút (PENDING). User cần xác nhận trước khi hết thời gian.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_id": {
                        "type": "string",
                        "description": "Mã xe cần đặt lịch"
                    },
                    "center_id": {
                        "type": "string",
                        "description": "Mã xưởng dịch vụ"
                    },
                    "slot_id": {
                        "type": "string",
                        "description": "Mã khung giờ cụ thể, ví dụ: SLOT_SC001_2026-04-10_0830"
                    },
                    "service_type": {
                        "type": "string",
                        "description": "Loại dịch vụ: kiểm tra, bảo dưỡng, sửa chữa, bảo hành, thay pin, khác"
                    },
                    "ai_diagnosis_log": {
                        "type": "string",
                        "description": "Tóm tắt kết quả chẩn đoán AI (nếu có) để KTV tham khảo"
                    },
                    "note": {
                        "type": "string",
                        "description": "Ghi chú thêm từ khách hàng"
                    }
                },
                "required": ["vehicle_id", "center_id", "slot_id", "service_type", "ai_diagnosis_log", "note"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_my_bookings",
            "description": "Tra cứu danh sách lịch hẹn đã đặt của khách hàng. Có thể lọc theo xe cụ thể. Dùng khi khách hỏi xem lại lịch đã đặt, trạng thái booking, v.v.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_id": {
                        "type": ["string", "null"],
                        "description": "Mã xe cần lọc (V001, V002...). Null = xem tất cả lịch hẹn."
                    }
                },
                "required": ["vehicle_id"],
                "additionalProperties": False
            }
        }
    },
]


# ─── Tool dispatcher ─────────────────────────────────────────────
TOOL_MAP = {
    "lookup_warranty_status": tools.lookup_warranty_status,
    "explain_warranty_policy": tools.explain_warranty_policy,
    "diagnose_telemetry": tools.diagnose_telemetry,
    "find_nearest_service_center": tools.find_nearest_service_center,
    "get_available_time_slots": tools.get_available_time_slots,
    "create_appointment": tools.create_appointment,
    "lookup_my_bookings": tools.lookup_my_bookings,
}

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
    "kiem tra",
    "sua chua",
    "bao hanh",
    "thay pin",
    "xuong",
    "trung tam",
    "slot",
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

GREETING_KEYWORDS = {
    "xin chao",
    "chao",
    "hello",
    "hi",
    "alo",
    "cam on",
    "thank",
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
    "toi",
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


def _normalize_text(text: str) -> str:
    """Lowercase, strip accents, and collapse punctuation for keyword matching."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_topic(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in TOPIC_KEYWORDS)


def _contains_booking_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in BOOKING_INTENT_KEYWORDS)


def _is_greeting_or_social(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in GREETING_KEYWORDS)


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
        values = {
            center["id"],
            center["name"],
            center["city"],
            center["district"],
            center["address"],
        }
        if any(_normalize_text(value) in normalized for value in values):
            return True

    return False


def _find_vehicle_reference(text: str, vehicles: list[dict], selected_vehicle_id: str = None) -> dict | None:
    normalized = _normalize_text(text)

    for vehicle in vehicles:
        candidates = {
            vehicle["id"],
            vehicle["model"],
            vehicle["vin"],
        }
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
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_messages.append(content)
        if len(text_messages) >= lookback:
            break

    return any(_contains_booking_intent(content) for content in text_messages)


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


def _build_runtime_context() -> str:
    current_dt = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    return (
        "\n\n## Runtime context"
        f"\nCurrent datetime: {current_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f"\nCurrent date: {current_dt.strftime('%Y-%m-%d')}"
        "\nWhen the customer refers to today, tomorrow, warranty remaining days, or appointment dates,"
        " use this runtime date as the source of truth."
    )


def _build_topic_clarification(vehicle: dict | None) -> str:
    vehicle_label = vehicle["model"] if vehicle else "xe nay"
    return (
        f"Anh/chị đang muốn em hỗ trợ gì với {vehicle_label}? "
        "Em có thể hỗ trợ tra cứu bảo hành, giải thích chính sách, chẩn đoán sơ bộ "
        "hoặc hỗ trợ đặt lịch kiểm tra."
    )


def _should_clarify_topic(messages: list[dict], vehicles: list[dict], selected_vehicle_id: str = None) -> dict | None:
    if not messages:
        return None

    latest_message = messages[-1]
    if latest_message.get("role") != "user":
        return None

    latest_content = latest_message.get("content")
    if not isinstance(latest_content, str) or not latest_content.strip():
        return None

    if _contains_topic(latest_content):
        return None

    if _get_recent_topic_context(messages):
        return None

    vehicle = _find_vehicle_reference(latest_content, vehicles, selected_vehicle_id)
    if not vehicle:
        return None

    return {
        "reply": _build_topic_clarification(vehicle),
        "tool_calls_log": [],
        "booking": None,
    }


def _build_booking_clarification(needs_location: bool, needs_datetime: bool) -> str:
    if needs_location and needs_datetime:
        return (
            "Để đặt lịch bảo dưỡng/kiểm tra, anh/chị cho em biết giúp khu vực hoặc thành phố muốn làm dịch vụ "
            "và thời gian mong muốn, ví dụ: Hà Nội sáng mai hoặc TP.HCM ngày 12/04 buổi chiều."
        )

    if needs_location:
        return (
            "Anh/chị muốn làm dịch vụ ở khu vực hoặc thành phố nào ạ? "
            "Khi có vị trí, em mới tìm đúng xưởng và khung giờ khả dụng."
        )

    return (
        "Anh/chị muốn đặt vào ngày hoặc khung giờ nào ạ? "
        "Ví dụ: sáng mai, chiều thứ 6, hoặc 10/04 lúc 09:00."
    )


def _should_clarify_booking_details(
    messages: list[dict],
    vehicles: list[dict],
    service_centers: list[dict],
    selected_vehicle_id: str = None,
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

    has_location = (
        _contains_service_location(latest_content, service_centers)
        or _history_contains_service_location(messages, service_centers)
    )
    has_datetime = (
        _contains_datetime_preference(latest_content)
        or _history_contains_datetime_preference(messages)
    )

    if has_location and has_datetime:
        return None

    return {
        "reply": _build_booking_clarification(
            needs_location=not has_location,
            needs_datetime=not has_datetime,
        ),
        "tool_calls_log": [],
        "booking": None,
    }


def _should_reject_out_of_scope(messages: list[dict], vehicles: list[dict], selected_vehicle_id: str = None) -> dict | None:
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
                    "Em chỉ hỗ trợ các vấn đề về bảo hành, chẩn đoán, lịch hẹn dịch vụ và dữ liệu xe VinFast "
                    "đã có trong hệ thống. Nếu anh/chị cần, em có thể hỗ trợ tra cứu bảo hành hoặc đặt lịch kiểm tra."
                ),
                "tool_calls_log": [],
                "booking": None,
            }
        return None

    if _is_greeting_or_social(latest_content):
        return None

    return {
        "reply": (
            "Em chỉ hỗ trợ các nội dung liên quan đến bảo hành xe VinFast từ dữ liệu đã nạp trong hệ thống, "
            "gồm tra cứu bảo hành, giải thích chính sách, chẩn đoán sơ bộ, tìm xưởng dịch vụ và đặt lịch."
        ),
        "tool_calls_log": [],
        "booking": None,
    }


def _execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool function and return JSON string result."""
    func = TOOL_MAP.get(name)
    if not func:
        return json.dumps({"error": f"Tool '{name}' không tồn tại."}, ensure_ascii=False)
    try:
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"Lỗi khi thực thi tool '{name}': {str(e)}"}, ensure_ascii=False)


# ─── Agent chat function ─────────────────────────────────────────
def chat(messages: list[dict], selected_vehicle_id: str = None) -> dict:
    """
    Process a chat conversation with the VinFast warranty agent.

    Args:
        messages: List of conversation messages (role: user/assistant/tool)
        selected_vehicle_id: Currently selected vehicle ID from the UI

    Returns:
        dict with 'reply' (str), 'tool_calls_log' (list), and optionally 'booking' info
    """
    from app import data as _data

    # Build system message with vehicle context including full vehicle list
    system_msg = SYSTEM_PROMPT
    system_msg += _build_runtime_context()

    # Inject danh sách xe của user vào system prompt để AI biết vehicle_id hợp lệ
    all_vehicles = _data.get_all_vehicles()
    all_service_centers = _data.get_all_service_centers()
    rejection_response = _should_reject_out_of_scope(messages, all_vehicles, selected_vehicle_id)
    if rejection_response:
        return rejection_response

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
    if all_vehicles:
        vehicle_lines = []
        for v in all_vehicles:
            vehicle_lines.append(
                f"  - **{v['id']}**: {v['model']} | VIN: {v['vin']} | Màu: {v['color']} | Mua: {v['purchase_date']} | ODO: {v['telemetry']['odo_km']:,} km | SOH pin: {v['telemetry']['battery_soh_percent']}%"
            )
        system_msg += "\n\n## Danh sách xe của khách hàng\n" + "\n".join(vehicle_lines)
        system_msg += "\n\n**Lưu ý:** Khi khách hỏi về xe, hãy dùng đúng vehicle_id ở trên (V001, V002, ...). Nếu khách chưa chọn xe cụ thể và có nhiều xe, hãy hỏi khách muốn tra cứu xe nào."

    system_msg += "\n\n## Clarification rule\nIf the customer's latest message only identifies a vehicle/model/VIN but does not state the support topic, ask what they want help with first and do not call tools yet."
    system_msg += "\n\n## Scope rule\nOnly answer questions about warranty, diagnostics, service centers, appointments, and the vehicle data loaded into this system. Refuse unrelated requests such as price, promotions, entertainment, weather, or general knowledge."
    system_msg += "\n\n## Booking rule\nWhen the customer wants maintenance or booking support, do not choose a service center, city, date, or time on their behalf. If location or preferred date/time is missing, ask for the missing details first and do not call slot or booking tools yet."

    if selected_vehicle_id:
        system_msg += f"\n\n## Xe đang được chọn trên giao diện\nVehicle ID: {selected_vehicle_id} — Khi khách hỏi mà không chỉ rõ xe, hãy dùng xe này."

    full_messages = [{"role": "system", "content": system_msg}] + messages

    tool_calls_log = []
    booking_info = None
    max_iterations = 5

    try:
        for iteration in range(max_iterations):
            response = client.chat.completions.create(
                model=MODEL,
                messages=full_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )

            choice = response.choices[0]

            # If the model wants to call tools
            if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
                # Add assistant message with tool calls to history
                full_messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    tool_calls_log.append({
                        "tool": fn_name,
                        "arguments": fn_args,
                        "iteration": iteration + 1,
                    })

                    result_str = _execute_tool(fn_name, fn_args)

                    # Track booking info for frontend
                    if fn_name == "create_appointment":
                        result_obj = json.loads(result_str)
                        if result_obj.get("success"):
                            booking_info = result_obj.get("booking")

                    # Add tool result to message history
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
            else:
                # Model finished with a text response
                reply = choice.message.content or ""
                return {
                    "reply": reply,
                    "tool_calls_log": tool_calls_log,
                    "booking": booking_info,
                }

        # If we hit max iterations, get a final text response
        full_messages.append({
            "role": "user",
            "content": "(Hệ thống: Bạn đã sử dụng quá nhiều tool calls. Hãy tổng hợp thông tin đã thu thập và trả lời khách hàng ngay.)"
        })
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
        )
        return {
            "reply": response.choices[0].message.content or "Xin lỗi, em không thể xử lý yêu cầu này. Anh/chị vui lòng liên hệ hotline 1900 23 23 89.",
            "tool_calls_log": tool_calls_log,
            "booking": booking_info,
        }

    except Exception as e:
        error_msg = str(e)
        # Log the real error server-side
        print(f"[AGENT ERROR] {error_msg}")
        # Return user-friendly message instead of raw error
        return {
            "reply": "Xin lỗi anh/chị, hệ thống đang gặp sự cố kết nối tạm thời. Anh/chị vui lòng thử lại sau giây lát hoặc liên hệ hotline **1900 23 23 89** để được hỗ trợ.",
            "tool_calls_log": tool_calls_log,
            "booking": None,
        }


from app.agent_langgraph import *  # noqa: F401,F403,E402
