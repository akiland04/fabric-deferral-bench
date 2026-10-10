"""Frozen thresholds file (T31c). Runs without the dataset or a detector."""
import json
import math

import pytest

from fdb.curve import risk_coverage_curve
from fdb.score_table import ScoredFrame
from fdb.selection import sweep
from fdb.thresholds import build_record, id_fingerprint, read_thresholds, write_thresholds

META = {"weights_sha256": "ab" * 32, "subset": "smoke",
        "settings": {"conf": 0.001, "iou": 0.7, "imgsz": 512, "max_det": 300, "device": "mps"}}

# normal, normal, DEFECTIVE, normal: risk 0, 0, 1/3, 1/4
FRAMES = [ScoredFrame("000003", "cal", 0.1, False), ScoredFrame("000001", "cal", 0.2, False),
          ScoredFrame("000004", "cal", 0.3, True), ScoredFrame("000002", "cal", 0.4, False)]
IDS = [f.image_id for f in FRAMES]


def record(budgets=(0.01, 0.3)):
    return build_record(sweep(risk_coverage_curve(FRAMES), budgets), META, "cal", IDS)


def test_t_round_trips_exactly(tmp_path):
    rec = record((0.01, 0.3))                         # 0.3 accepts everything: t is just above 0.4
    assert rec["thresholds"][1]["t"] == math.nextafter(0.4, math.inf)
    write_thresholds(tmp_path / "t.json", rec)
    back = read_thresholds(tmp_path / "t.json")
    assert [x["t"] for x in back["thresholds"]] == [x["t"] for x in rec["thresholds"]]


def test_identical_rewrite_is_a_no_op(tmp_path):
    path = tmp_path / "t.json"
    assert write_thresholds(path, record()) is True
    before = path.read_bytes()
    assert write_thresholds(path, record()) is False
    assert path.read_bytes() == before


def test_different_thresholds_are_refused(tmp_path):
    path = tmp_path / "t.json"
    write_thresholds(path, record((0.01, 0.3)))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_thresholds(path, record((0.01, 0.2)))
    assert path.read_bytes() == before


def test_same_selection_gives_the_same_bytes(tmp_path):
    write_thresholds(tmp_path / "a.json", record())
    write_thresholds(tmp_path / "b.json", record())
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert b"\r\n" not in (tmp_path / "a.json").read_bytes()


def test_records_the_model_and_no_timestamp_or_path():
    rec = record()
    assert rec["model"] == {"weights_sha256": META["weights_sha256"], "subset": "smoke"}
    assert rec["inference"] == META["settings"]
    assert set(rec) == {"schema_version", "model", "score", "rule", "inference", "selection", "thresholds"}


def test_infeasible_budget_is_stored_with_undefined_risk(tmp_path):
    # with 1 defective in 4, risk 0 is reachable, so make a set where it is not
    frames = [ScoredFrame("a", "cal", 0.1, True), ScoredFrame("b", "cal", 0.2, False)]
    rec = build_record(sweep(risk_coverage_curve(frames), [0.01]), META, "cal", ["a", "b"])
    write_thresholds(tmp_path / "t.json", rec)
    entry = read_thresholds(tmp_path / "t.json")["thresholds"][0]
    assert entry["feasible"] is False and entry["t"] == 0.0
    assert entry["selective_risk"] is None
    assert '"selective_risk": null' in (tmp_path / "t.json").read_text()


def test_fingerprint_ignores_order_but_not_content():
    assert id_fingerprint(["b", "a", "c"]) == id_fingerprint(["c", "b", "a"])
    assert id_fingerprint(["a", "b"]) != id_fingerprint(["a", "c"])
    with pytest.raises(ValueError):
        id_fingerprint(["a", "a"])


def test_rejects_ids_that_do_not_match_the_curve():
    with pytest.raises(ValueError):
        build_record(sweep(risk_coverage_curve(FRAMES), [0.01]), META, "cal", IDS[:3])


def test_rejects_an_unknown_schema_version(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError):
        read_thresholds(path)