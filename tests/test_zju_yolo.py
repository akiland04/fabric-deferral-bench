"""ZJU XML -> YOLO conversion (T19a). Runs without the dataset."""
from fdb.yolo import Box, to_yolo_line
from fdb.zju import read_boxes

XML_002678 = """<?xml version="1.0" encoding="UTF-8"?>
<annotation>
\t<filename>002678.jpg</filename>
\t<defective>1</defective>
\t<bbox><xmin>496</xmin><ymin>167</ymin><xmax>512</xmax><ymax>270</ymax></bbox>
\t<bbox><xmin>215</xmin><ymin>457</ymin><xmax>311</xmax><ymax>512</ymax></bbox>
</annotation>
"""
XML_NORMAL = """<?xml version="1.0" encoding="UTF-8"?>
<annotation><filename>000001.jpg</filename><defective>0</defective></annotation>
"""


def test_read_boxes_defective(tmp_path):
    p = tmp_path / "002678.xml"
    p.write_text(XML_002678)
    assert read_boxes(p) == (True, [Box(496, 167, 512, 270), Box(215, 457, 311, 512)])


def test_read_boxes_normal(tmp_path):
    p = tmp_path / "000001.xml"
    p.write_text(XML_NORMAL)
    assert read_boxes(p) == (False, [])


def test_known_boxes_convert_exactly():
    assert to_yolo_line(Box(496, 167, 512, 270), 512, 512) == "0 0.984375 0.426758 0.031250 0.201172"
    assert to_yolo_line(Box(215, 457, 311, 512), 512, 512) == "0 0.513672 0.946289 0.187500 0.107422"


def test_box_past_the_edge_is_clamped():
    assert to_yolo_line(Box(-5, 500, 520, 530), 512, 512) == "0 0.500000 0.988281 1.000000 0.023438"


def test_zero_area_box_is_dropped():
    assert to_yolo_line(Box(10, 10, 10, 20), 512, 512) is None