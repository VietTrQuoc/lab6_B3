"""
OpenAI Agent module for VinFast Warranty AI.

Implements function calling with tool definitions, system prompt with guardrails,
and the agent loop that handles multi-turn tool calls.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from app import tools

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

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
]


# ─── Tool dispatcher ─────────────────────────────────────────────
TOOL_MAP = {
    "lookup_warranty_status": tools.lookup_warranty_status,
    "explain_warranty_policy": tools.explain_warranty_policy,
    "diagnose_telemetry": tools.diagnose_telemetry,
    "find_nearest_service_center": tools.find_nearest_service_center,
    "get_available_time_slots": tools.get_available_time_slots,
    "create_appointment": tools.create_appointment,
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
    # Build system message with vehicle context
    system_msg = SYSTEM_PROMPT
    if selected_vehicle_id:
        system_msg += f"\n\n## Xe đang được chọn trên giao diện\nVehicle ID: {selected_vehicle_id}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    tool_calls_log = []
    booking_info = None
    max_iterations = 5

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
