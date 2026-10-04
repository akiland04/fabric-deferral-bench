"""Decoding YOLO label lines back to pixel boxes (T19d). Runs without the dataset."""
import pytest

from fdb.yolo import Box, from_yolo_line, to_yolo_line


def corners(b: Box) -> tuple[float, float, float, float]:
    return (b.xmin, b.ymin, b.xmax, b.ymax)


def test_decode_known_line():
    cls, box = from_yolo_line("0 0.984375 0.426758 0.031250 0.201172", 512, 512)
    assert cls == 0
    assert corners(box) == pytest.approx((496, 167, 512, 270), abs=0.01)


@pytest.mark.parametrize("box", [
    Box(496, 167, 512, 270),
    Box(215, 457, 311, 512),
    Box(0, 0, 512, 512),
    Box(10, 20, 13, 21),
])
def test_encode_then_decode_returns_the_same_box(box):
    _, decoded = from_yolo_line(to_yolo_line(box, 512, 512), 512, 512)
    assert corners(decoded) == pytest.approx(corners(box), abs=0.01)