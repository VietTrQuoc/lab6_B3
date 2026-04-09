"""
Tool functions for the VinFast Warranty AI Agent.

Each function corresponds to an OpenAI tool (function calling).
They interact with the in-memory data layer and return structured dicts.
"""

from datetime import datetime, date
from app import data


def lookup_warranty_status(vehicle_id: str) -> dict:
    """
    Tra cứu tình trạng bảo hành theo xe.
    Returns warranty status, remaining time, and basic vehicle info.
    """
    vehicle = data.get_vehicle(vehicle_id)
    if not vehicle:
        return {"error": f"Không tìm thấy xe với mã {vehicle_id}."}

    today = date.today()
    warranty_end_v = date.fromisoformat(vehicle["warranty_end_vehicle"])
    warranty_end_b = date.fromisoformat(vehicle["warranty_end_battery"])

    vehicle_active = today <= warranty_end_v
    battery_active = today <= warranty_end_b

    remaining_vehicle = (warranty_end_v - today).days if vehicle_active else 0
    remaining_battery = (warranty_end_b - today).days if battery_active else 0

    return {
        "vehicle_id": vehicle["id"],
        "model": vehicle["model"],
        "vin": vehicle["vin"],
        "owner_name": vehicle["owner_name"],
        "purchase_date": vehicle["purchase_date"],
        "color": vehicle["color"],
        "warranty": {
            "vehicle": {
                "start": vehicle["warranty_start"],
                "end": vehicle["warranty_end_vehicle"],
                "is_active": vehicle_active,
                "remaining_days": remaining_vehicle,
            },
            "battery": {
                "start": vehicle["warranty_start"],
                "end": vehicle["warranty_end_battery"],
                "type": vehicle["battery_type"],
                "capacity_kwh": vehicle["battery_capacity_kwh"],
                "is_active": battery_active,
                "remaining_days": remaining_battery,
            },
        },
        "last_service_date": vehicle["telemetry"].get("last_service_date"),
    }


def explain_warranty_policy(category: str) -> dict:
    """
    Giải thích quyền lợi bảo hành pin và linh kiện theo policy hiện hành.
    Category: "pin", "linh_kien", "tong_quat", "bao_duong"
    """
    policy = data.get_warranty_policy()

    if category == "pin":
        return {
            "category": "Pin LFP",
            "policy_version": policy["policy_version"],
            "details": policy["warranty_terms"]["battery"],
            "applicable_models": policy["applicable_models"],
        }
    elif category == "linh_kien":
        return {
            "category": "Xe và Linh kiện",
            "policy_version": policy["policy_version"],
            "details": policy["warranty_terms"]["vehicle"],
            "applicable_models": policy["applicable_models"],
        }
    elif category == "bao_duong":
        return {
            "category": "Lịch bảo dưỡng định kỳ",
            "policy_version": policy["policy_version"],
            "schedule": policy["maintenance_schedule"],
        }
    else:  # tong_quat
        return {
            "category": "Tổng quát",
            "policy_version": policy["policy_version"],
            "vehicle_warranty": policy["warranty_terms"]["vehicle"],
            "battery_warranty": policy["warranty_terms"]["battery"],
            "maintenance_schedule": policy["maintenance_schedule"],
            "applicable_models": policy["applicable_models"],
            "contact": policy["contact"],
        }


def diagnose_telemetry(vehicle_id: str) -> dict:
    """
    Chẩn đoán sơ bộ từ telemetry data của xe.
    Analyzes ODO, SOH, charge cycles, temp, error codes, tire pressure.
    Returns structured diagnosis with severity levels.
    """
    vehicle = data.get_vehicle(vehicle_id)
    if not vehicle:
        return {"error": f"Không tìm thấy xe với mã {vehicle_id}."}

    policy = data.get_warranty_policy()
    error_code_db = policy.get("error_codes", {})
    telemetry = vehicle["telemetry"]
    issues = []
    recommendations = []

    # ─── SOH Analysis ─────────────────────────────────────────
    soh = telemetry["battery_soh_percent"]
    if soh < 70:
        issues.append({
            "type": "CRITICAL",
            "component": "Pin",
            "detail": f"SOH pin ở mức {soh}% (dưới ngưỡng 70%). Pin hoạt động kém hiệu quả.",
        })
        recommendations.append("Cần đến xưởng dịch vụ kiểm tra pin ngay. Nếu trong thời hạn bảo hành, pin có thể được thay thế theo chính sách.")
    elif soh < 80:
        issues.append({
            "type": "WARNING",
            "component": "Pin",
            "detail": f"SOH pin ở mức {soh}%. Pin đang suy giảm, cần theo dõi.",
        })
        recommendations.append("Nên đặt lịch kiểm tra pin tại xưởng dịch vụ trong vòng 1 tháng.")

    # ─── Charge Cycles ────────────────────────────────────────
    cycles = telemetry["charge_cycles"]
    if cycles > 1500:
        issues.append({
            "type": "WARNING",
            "component": "Pin",
            "detail": f"Số chu kỳ sạc cao ({cycles} lần). Pin LFP thường duy trì 70% SOH sau 2.000 chu kỳ.",
        })
    elif cycles > 1000:
        issues.append({
            "type": "INFO",
            "component": "Pin",
            "detail": f"Đã trải qua {cycles} chu kỳ sạc. Nên theo dõi SOH thường xuyên.",
        })

    # ─── Temperature ──────────────────────────────────────────
    temp = telemetry["operating_temp_avg_c"]
    if temp > 42:
        issues.append({
            "type": "WARNING",
            "component": "Nhiệt độ",
            "detail": f"Nhiệt độ vận hành trung bình {temp}°C khá cao. Ngưỡng cảnh báo là 45°C.",
        })
        recommendations.append("Tránh sạc và vận hành trong điều kiện nắng nóng kéo dài. Kiểm tra hệ thống tản nhiệt.")

    # ─── Tire Pressure ────────────────────────────────────────
    front = telemetry.get("tire_pressure_front_bar", 2.2)
    rear = telemetry.get("tire_pressure_rear_bar", 2.2)
    if front < 1.9:
        issues.append({
            "type": "WARNING",
            "component": "Lốp trước",
            "detail": f"Áp suất lốp trước thấp: {front} bar (khuyến nghị 2.0 - 2.5 bar).",
        })
        recommendations.append(f"Xác nhận lốp non hơi {front} bar. Cần bơm lốp trước lên áp suất tiêu chuẩn và kiểm tra có bị thủng không.")
    if rear < 1.9:
        issues.append({
            "type": "WARNING",
            "component": "Lốp sau",
            "detail": f"Áp suất lốp sau thấp: {rear} bar (khuyến nghị 2.0 - 2.5 bar).",
        })
        recommendations.append(f"Lốp sau non hơi {rear} bar. Cần kiểm tra và bơm lốp.")

    # ─── ODO-based maintenance check ─────────────────────────
    odo = telemetry["odo_km"]
    last_service = telemetry.get("last_service_date")
    maintenance_schedule = policy.get("maintenance_schedule", {}).get("intervals", [])
    for interval in reversed(maintenance_schedule):
        if odo >= interval["at_km"]:
            if last_service:
                days_since = (date.today() - date.fromisoformat(last_service)).days
                if days_since > interval.get("or_months", 12) * 30:
                    issues.append({
                        "type": "INFO",
                        "component": "Bảo dưỡng",
                        "detail": f"ODO {odo:,} km. Đã {days_since} ngày kể từ lần bảo dưỡng gần nhất. Xe có thể đến hạn bảo dưỡng '{interval['description']}'.",
                    })
                    recommendations.append(f"Nên đặt lịch bảo dưỡng '{interval['description']}' sớm.")
            break

    # ─── Error Codes ──────────────────────────────────────────
    error_codes = telemetry.get("last_error_codes", [])
    error_details = []
    for code in error_codes:
        info = error_code_db.get(code, {})
        error_details.append({
            "code": code,
            "severity": info.get("severity", "unknown"),
            "description": info.get("description", "Mã lỗi không xác định"),
            "recommendation": info.get("recommendation", "Liên hệ xưởng dịch vụ để kiểm tra."),
        })
        if info.get("recommendation"):
            recommendations.append(f"[{code}] {info['recommendation']}")

    # ─── Status summary ──────────────────────────────────────
    has_critical = any(i["type"] == "CRITICAL" for i in issues)
    has_warning = any(i["type"] == "WARNING" for i in issues)

    if has_critical:
        overall = "CRITICAL"
    elif has_warning:
        overall = "WARNING"
    elif issues:
        overall = "INFO"
    else:
        overall = "GOOD"

    return {
        "vehicle_id": vehicle["id"],
        "model": vehicle["model"],
        "vin": vehicle["vin"],
        "overall_status": overall,
        "telemetry_snapshot": {
            "odo_km": odo,
            "battery_soh_percent": soh,
            "charge_cycles": cycles,
            "operating_temp_avg_c": temp,
            "tire_pressure_front_bar": front,
            "tire_pressure_rear_bar": rear,
            "error_codes": error_codes,
        },
        "issues": issues,
        "error_code_details": error_details,
        "recommendations": recommendations,
    }


def find_nearest_service_center(city: str) -> dict:
    """
    Tìm xưởng dịch vụ gần nhất theo thành phố.
    Returns list of centers in the given city with available slot counts.
    """
    centers = data.get_service_centers_by_city(city)
    if not centers:
        all_cities = sorted(set(sc["city"] for sc in data.get_all_service_centers()))
        return {
            "error": f"Không tìm thấy xưởng dịch vụ tại {city}.",
            "available_cities": all_cities,
            "hotline": "1900 23 23 89",
        }

    results = []
    for c in centers:
        # Count available slots for next 7 days
        slots = data.get_available_slots(c["id"])
        results.append({
            "id": c["id"],
            "name": c["name"],
            "address": c["address"],
            "city": c["city"],
            "district": c["district"],
            "phone": c["phone"],
            "working_hours": c["working_hours"],
            "type": c["type"],
            "services": c["services"],
            "available_slots_count": len(slots),
        })

    return {
        "city": city,
        "centers": results,
        "total": len(results),
    }


def get_available_time_slots(center_id: str, date_str: str = None) -> dict:
    """
    Lấy danh sách time slot khả dụng cho xưởng dịch vụ.
    """
    center = data.get_service_center(center_id)
    if not center:
        return {"error": f"Không tìm thấy xưởng dịch vụ mã {center_id}."}

    slots = data.get_available_slots(center_id, date_str)

    # Group by date for readability, include slot_id so AI can book directly
    by_date = {}
    for s in slots:
        d = s["date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append({
            "slot_id": s["slot_id"],
            "time": s["time"],
        })

    return {
        "center_id": center_id,
        "center_name": center["name"],
        "filter_date": date_str,
        "available_dates": by_date,
        "total_slots": len(slots),
    }


def create_appointment(vehicle_id: str, center_id: str, slot_id: str,
                       service_type: str, ai_diagnosis_log: str = "",
                       note: str = "") -> dict:
    """
    Đặt lịch kiểm tra hoặc bảo dưỡng.
    Atomic: AVAILABLE -> PENDING with 5-minute TTL.
    """
    vehicle = data.get_vehicle(vehicle_id)
    if not vehicle:
        return {"error": f"Không tìm thấy xe với mã {vehicle_id}."}

    center = data.get_service_center(center_id)
    if not center:
        return {"error": f"Không tìm thấy xưởng dịch vụ mã {center_id}."}

    booking = data.hold_slot(
        slot_id=slot_id,
        vehicle_id=vehicle_id,
        service_type=service_type,
        ai_diagnosis_log=ai_diagnosis_log,
        note=note,
    )

    if not booking:
        return {
            "error": "Khung giờ này đã được đặt hoặc không tồn tại. Vui lòng chọn khung giờ khác.",
            "suggestion": "Sử dụng công cụ get_available_time_slots để xem các slot còn trống.",
        }

    return {
        "success": True,
        "message": "Em đã giữ chỗ cho anh/chị. Lịch hẹn đang ở trạng thái chờ xác nhận.",
        "ttl_message": "Em sẽ giữ chỗ này cho anh/chị trong 5 phút. Vui lòng xác nhận để hoàn tất đặt lịch.",
        "booking": booking,
    }


def lookup_my_bookings(vehicle_id: str = None) -> dict:
    """
    Tra cứu tất cả lịch hẹn đã đặt của khách hàng.
    Có thể lọc theo vehicle_id. Trả về danh sách booking với trạng thái hiện tại.
    """
    bookings = data.get_user_bookings(user_id="U_VIN_001", vehicle_id=vehicle_id)

    if not bookings:
        return {
            "message": "Anh/chị chưa có lịch hẹn nào.",
            "bookings": [],
            "total": 0,
        }

    return {
        "bookings": bookings,
        "total": len(bookings),
        "filter_vehicle_id": vehicle_id,
    }
