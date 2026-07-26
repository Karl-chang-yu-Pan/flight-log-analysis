import json

import pytest


REPORT_KEYS = {
    "airframe_summary",
    "question_intent_summary",
    "ranked_hypotheses",
    "excluded_mechanisms",
    "confirmed",
    "unconfirmed",
    "final_summary",
}


@pytest.fixture
def assert_analysis_engine_contract():
    """Assert the stable artifacts shared by legacy and replacement engines."""

    def assert_contract(report, output_dir, dev_log_root, run_id):
        report_path = output_dir / "report.json"
        run_dir = dev_log_root / run_id

        assert output_dir.is_dir()
        assert report_path.is_file()
        assert run_dir.is_dir()
        assert (run_dir / "run_events.jsonl").is_file()
        assert (run_dir / "metadata.json").is_file()

        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert set(report_payload) == REPORT_KEYS
        expected_payload = (
            report.model_dump()
            if hasattr(report, "model_dump")
            else report
        )
        assert report_payload == expected_payload

        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["report_path"] == str(report_path)

        events = [
            json.loads(line)
            for line in (run_dir / "run_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert events[-1]["event"] == "run.finished"

    return assert_contract
