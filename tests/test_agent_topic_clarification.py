import csv
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import agent_langgraph as agent, data


class AgentTopicClarificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._original_bookings_csv_path = data.BOOKINGS_CSV_PATH
        data.BOOKINGS_CSV_PATH = Path(cls._tmpdir.name) / "bookings.csv"

    @classmethod
    def tearDownClass(cls):
        data.BOOKINGS_CSV_PATH = cls._original_bookings_csv_path
        cls._tmpdir.cleanup()

    def setUp(self):
        if data.BOOKINGS_CSV_PATH.exists():
            data.BOOKINGS_CSV_PATH.unlink()
        data.init_data()

    def test_vehicle_only_message_requires_topic_clarification(self):
        result = agent._should_clarify_topic(
            messages=[{"role": "user", "content": "Evo200"}],
            vehicles=data.get_all_vehicles(),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["tool_calls_log"], [])
        self.assertIn("Evo200", result["reply"])

    def test_recent_topic_context_skips_extra_clarification(self):
        result = agent._should_clarify_topic(
            messages=[
                {"role": "user", "content": "Toi muon kiem tra bao hanh cho xe cua toi."},
                {"role": "assistant", "content": "Anh/chi muon tra cuu xe nao?"},
                {"role": "user", "content": "Evo200"},
            ],
            vehicles=data.get_all_vehicles(),
        )

        self.assertIsNone(result)

    def test_chat_does_not_call_openai_for_ambiguous_selected_vehicle_message(self):
        with patch.object(
            agent,
            "_get_agent_graph",
            side_effect=AssertionError("Graph should not be called for ambiguous vehicle-only prompts."),
        ):
            result = agent.chat(
                messages=[{"role": "user", "content": "xe nay"}],
                selected_vehicle_id="V001",
            )

        normalized_reply = agent._normalize_text(result["reply"])
        self.assertIn("ho tro gi", normalized_reply)
        self.assertEqual(result["tool_calls_log"], [])
        self.assertIsNone(result["booking"])

    def test_out_of_scope_vehicle_question_is_rejected(self):
        result = agent._should_reject_out_of_scope(
            messages=[{"role": "user", "content": "Gia ban cua Evo200 bao nhieu?"}],
            vehicles=data.get_all_vehicles(),
        )

        self.assertIsNotNone(result)
        self.assertIn("bao hanh", agent._normalize_text(result["reply"]))

    def test_runtime_context_uses_current_datetime(self):
        runtime_context = agent._build_runtime_context()
        current_date = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")

        self.assertIn(current_date, runtime_context)

    def test_relative_date_resolution_for_this_saturday(self):
        resolved = agent._resolve_relative_dates(
            "cho toi khung gio muon nhat vao thu 7 tuan nay",
            base_date=datetime(2026, 4, 9).date(),
        )

        self.assertIn(("thu 7 tuần này", "2026-04-11"), resolved)

    def test_booking_flow_requires_location_first(self):
        result = agent._should_clarify_booking_details(
            messages=[{"role": "user", "content": "Bao duong"}],
            vehicles=data.get_all_vehicles(),
            service_centers=data.get_all_service_centers(),
            selected_vehicle_id="V001",
        )

        self.assertIsNotNone(result)
        normalized_reply = agent._normalize_text(result["reply"])
        self.assertIn("thanh pho", normalized_reply)
        self.assertIn("gan nhat", normalized_reply)

    def test_booking_flow_does_not_auto_pick_location_on_short_follow_up(self):
        with patch.object(
            agent,
            "_get_agent_graph",
            side_effect=AssertionError("Graph should not be called before booking details are clarified."),
        ):
            result = agent.chat(
                messages=[
                    {"role": "assistant", "content": "Anh/chị muốn em hỗ trợ gì với Evo200?"},
                    {"role": "user", "content": "bảo dưỡng"},
                    {"role": "assistant", "content": "Anh/chị muốn đặt ở khu vực nào và khi nào?"},
                    {"role": "user", "content": "đâu"},
                ],
                selected_vehicle_id="V001",
            )

        normalized_reply = agent._normalize_text(result["reply"])
        self.assertIn("thanh pho", normalized_reply)
        self.assertEqual(result["tool_calls_log"], [])

    def test_booking_flow_location_only_can_continue_to_center_lookup(self):
        result = agent._should_clarify_booking_details(
            messages=[
                {"role": "user", "content": "Dat lich bao duong cho xe nay"},
                {"role": "assistant", "content": "Anh/chị muốn làm ở đâu?"},
                {"role": "user", "content": "Ha Noi"},
            ],
            vehicles=data.get_all_vehicles(),
            service_centers=data.get_all_service_centers(),
            selected_vehicle_id="V001",
        )

        self.assertIsNone(result)

    def test_booking_flow_requires_time_when_specific_center_selected(self):
        result = agent._should_clarify_booking_details(
            messages=[
                {"role": "user", "content": "Dat lich bao duong cho xe nay"},
                {"role": "assistant", "content": "Anh/chị đang ở khu vực nào?"},
                {"role": "user", "content": "Ha Noi"},
                {"role": "assistant", "content": "Em da tim duoc cac xuong gan nhat."},
                {"role": "user", "content": "VinFast Long Bien"},
            ],
            vehicles=data.get_all_vehicles(),
            service_centers=data.get_all_service_centers(),
            selected_vehicle_id="V001",
        )

        self.assertIsNotNone(result)
        normalized_reply = agent._normalize_text(result["reply"])
        self.assertIn("ngay", normalized_reply)

    def test_booking_flow_requires_time_again_when_user_changes_center(self):
        result = agent._should_clarify_booking_details(
            messages=[
                {"role": "user", "content": "Dat lich o Ha Noi sang mai"},
                {"role": "assistant", "content": "Da co lich tai Ha Noi."},
                {"role": "user", "content": "Toi muon dat lich hen VF Hai Phong"},
            ],
            vehicles=data.get_all_vehicles(),
            service_centers=data.get_all_service_centers(),
            selected_vehicle_id="V001",
        )

        self.assertIsNotNone(result)
        normalized_reply = agent._normalize_text(result["reply"])
        self.assertIn("khung gio", normalized_reply)

    def test_hai_phong_has_latest_slot_based_on_working_hours(self):
        slots = data.get_available_slots("SC008")
        self.assertTrue(slots)
        self.assertEqual(slots[-1]["time"], "17:00")

    def test_thu_duc_has_latest_slot_based_on_working_hours(self):
        slots = data.get_available_slots("SC005")
        self.assertTrue(slots)
        self.assertEqual(slots[-1]["time"], "17:30")

    def test_booking_is_saved_to_csv_with_exact_service_type(self):
        slot = data.get_available_slots("SC003")[0]
        booking = data.hold_slot(
            slot_id=slot["slot_id"],
            vehicle_id="V001",
            service_type="kiem tra pin",
            note="khach muon kiem tra pin",
        )

        self.assertIsNotNone(booking)
        self.assertTrue(data.BOOKINGS_CSV_PATH.exists())

        with open(data.BOOKINGS_CSV_PATH, "r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["service_type"], "kiem tra pin")

    def test_reschedule_booking_updates_csv_and_survives_reload(self):
        original_slot = data.get_available_slots("SC003")[0]
        booking = data.hold_slot(
            slot_id=original_slot["slot_id"],
            vehicle_id="V001",
            service_type="bao duong dinh ky",
        )
        confirmed_booking = data.confirm_booking(booking["booking_id"])
        replacement_slot = data.get_available_slots("SC008")[0]

        updated_booking = data.reschedule_booking(
            booking_id=booking["booking_id"],
            center_id="SC008",
            slot_id=replacement_slot["slot_id"],
            service_type="kiem tra pin",
            note="doi lich sang Hai Phong",
        )

        self.assertIsNotNone(confirmed_booking)
        self.assertIsNotNone(updated_booking)
        self.assertEqual(updated_booking["center_id"], "SC008")
        self.assertEqual(updated_booking["service_type"], "kiem tra pin")
        self.assertEqual(updated_booking["status"], "CONFIRMED")

        data.init_data()
        reloaded_booking = data.get_booking(booking["booking_id"])
        self.assertIsNotNone(reloaded_booking)
        self.assertEqual(reloaded_booking["center_id"], "SC008")
        self.assertEqual(reloaded_booking["service_type"], "kiem tra pin")


if __name__ == "__main__":
    unittest.main()
