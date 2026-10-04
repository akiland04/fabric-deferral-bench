"""The predictions contract (T20c). Runs without the dataset or a detector."""
import pytest

from fdb.predictions import Detection, FramePrediction, check_complete, read_predictions, write_predictions

FRAMES = [
    FramePrediction("000101", "cal", 512, 512, (
        Detection(10.0, 20.0, 50.5, 60.25, 0.91234, 0),
        Detection(100.0, 100.0, 120.0, 140.0, 0.0015, 0),
    )),
    FramePrediction("000102", "cal", 512, 512, ()),  # no boxes: must survive the round trip
]


def test_round_trip_keeps_frames_without_boxes(tmp_path):
    path = tmp_path / "p.jsonl"
    assert write_predictions(path, FRAMES) == 2
    assert read_predictions(path) == FRAMES


def test_empty_frame_is_written_as_an_empty_list(tmp_path):
    path = tmp_path / "p.jsonl"
    write_predictions(path, FRAMES)
    assert '"image_id":"000102"' in path.read_text().splitlines()[1]
    assert '"boxes":[]' in path.read_text().splitlines()[1]


def test_record_without_boxes_key_is_rejected(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"image_id":"000103","split":"cal","width":512,"height":512}\n')
    with pytest.raises(ValueError, match="no 'boxes' key"):
        read_predictions(path)


def test_check_complete_accepts_an_exact_match():
    check_complete(FRAMES, ["000101", "000102"])


@pytest.mark.parametrize("expected, frames", [
    (["000101", "000102", "000103"], FRAMES),          # an image was never predicted
    (["000101"], FRAMES),                              # a prediction for an image not in the manifest
    (["000101", "000102"], FRAMES + [FRAMES[1]]),      # an image predicted twice
])
def test_check_complete_rejects_mismatches(expected, frames):
    with pytest.raises(ValueError, match="do not match the manifest"):
        check_complete(frames, expected)