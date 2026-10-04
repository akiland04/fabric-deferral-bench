"""Frame score s(x) (T21a). Runs without the dataset or a detector."""
import itertools
import math

import pytest

from fdb.predictions import Detection, FramePrediction
from fdb.scoring import max_conf_score, score_frames


def frame(*confs: float, image_id: str = "000001") -> FramePrediction:
    """A test frame with one box per confidence (box positions do not matter to the score)."""
    return FramePrediction(image_id, "cal", 512, 512, tuple(Detection(0, 0, 10, 10, c) for c in confs))


def test_no_boxes_scores_exactly_zero():
    score = max_conf_score(frame())
    assert score == 0.0 and isinstance(score, float)


def test_one_box_scores_its_confidence():
    assert max_conf_score(frame(0.37)) == 0.37


@pytest.mark.parametrize("confs", list(itertools.permutations([0.82, 0.10, 0.004])))
def test_several_boxes_score_the_highest_in_any_order(confs):
    assert max_conf_score(frame(*confs)) == 0.82


def test_stronger_evidence_never_lowers_the_score():
    # direction of the score: adding a more confident box must raise it; no boxes is the lowest
    assert max_conf_score(frame()) < max_conf_score(frame(0.2)) < max_conf_score(frame(0.2, 0.9))


@pytest.mark.parametrize("bad", [1.2, -0.1, math.nan, math.inf])
def test_confidence_outside_zero_one_is_rejected(bad):
    with pytest.raises(ValueError, match="outside"):
        max_conf_score(frame(0.5, bad))


def test_score_frames_gives_one_score_per_image():
    scores = score_frames([frame(0.3, image_id="a"), frame(image_id="b")])
    assert scores == {"a": 0.3, "b": 0.0}


def test_score_frames_accepts_another_scorer():
    def count_boxes(f: FramePrediction) -> float:  # any function FramePrediction -> float plugs in
        return float(len(f.boxes))

    assert score_frames([frame(0.3, 0.4, image_id="a")], scorer=count_boxes) == {"a": 2.0}


def test_score_frames_rejects_a_duplicate_image():
    with pytest.raises(ValueError, match="appears twice"):
        score_frames([frame(0.3, image_id="a"), frame(0.5, image_id="a")])