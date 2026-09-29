from __future__ import annotations

import json

from oracle_builder.training.control import FileSegmentControl, write_segment_result


def test_file_control_is_monotonic_and_segment_result_is_atomic(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"sequence": 2, "command": "yield", "reason": "fairness"}))
    control = FileSegmentControl(control_path)
    assert control.poll().command == "yield"
    # Replays and malformed commands cannot supersede the accepted directive.
    control_path.write_text(json.dumps({"sequence": 2, "command": "stop"}))
    assert control.poll().command == "yield"
    control_path.write_text(json.dumps({"sequence": 3, "command": "invalid"}))
    assert control.poll().command == "yield"

    result = tmp_path / "segment-result.json"
    write_segment_result(result, {"phase": "train", "cursor": {"completed_epoch": 1}})
    assert json.loads(result.read_text()) == {
        "cursor": {"completed_epoch": 1}, "phase": "train"
    }
