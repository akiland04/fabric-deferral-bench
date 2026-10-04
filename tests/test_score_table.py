"""Per-frame score table (T21b). Runs without the dataset or a detector."""
import pytest

from fdb.score_table import ScoredFrame, join_truth, read_score_table, write_score_table

SCORES = {"000002": 0.013214, "000001": 0.82, "000003": 0.0}
TRUTH = {"000001": True, "000002": False, "000003": True}


def test_join_keeps_every_frame_sorted_by_id():
    rows = join_truth(SCORES, TRUTH, "cal")
    assert rows == [
        ScoredFrame("000001", "cal", 0.82, True),
        ScoredFrame("000002", "cal", 0.013214, False),
        ScoredFrame("000003", "cal", 0.0, True),
    ]


def test_join_rejects_a_labelled_image_without_a_score():
    with pytest.raises(ValueError, match="1 labelled images have no score"):
        join_truth({"000001": 0.82}, {"000001": True, "000002": False}, "cal")


def test_join_rejects_a_scored_image_without_a_label():
    with pytest.raises(ValueError, match="1 scored images have no label"):
        join_truth({"000001": 0.82, "000009": 0.1}, {"000001": True}, "cal")


def test_write_then_read_gives_the_same_table(tmp_path):
    rows = join_truth(SCORES, TRUTH, "cal")
    write_score_table(tmp_path / "s.csv", rows)
    assert read_score_table(tmp_path / "s.csv") == rows


def test_file_is_identical_on_rewrite_and_uses_lf(tmp_path):
    rows = join_truth(SCORES, TRUTH, "cal")
    write_score_table(tmp_path / "a.csv", rows)
    write_score_table(tmp_path / "b.csv", list(reversed(rows)))   # input order must not matter
    a = (tmp_path / "a.csv").read_bytes()
    assert a == (tmp_path / "b.csv").read_bytes()
    assert b"\r\n" not in a
    assert a.splitlines()[1] == b"000001,cal,0.820000,1"


@pytest.mark.parametrize("bad_file, message", [
    ("id,split,score,defective\n", "header"),
    ("image_id,split,score,defective\n000001,cal,1.5,1\n", "outside"),
    ("image_id,split,score,defective\n000001,cal,0.5,yes\n", "0 or 1"),
])
def test_read_rejects_a_malformed_table(tmp_path, bad_file, message):
    (tmp_path / "s.csv").write_text(bad_file)
    with pytest.raises(ValueError, match=message):
        read_score_table(tmp_path / "s.csv")