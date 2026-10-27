"""RF-DETR uses COCO 91-id indexing; class_name preferred when present."""

from streetsense.rfdetr_detector import coco91_name, resolve_label


def test_coco91_core_street_classes():
    assert coco91_name(1) == "person"
    assert coco91_name(3) == "car"
    assert coco91_name(10) == "traffic light"
    assert coco91_name(2) == "bicycle"
    assert coco91_name(4) == "motorcycle"
    assert coco91_name(6) == "bus"
    assert coco91_name(8) == "truck"
    assert coco91_name(91) is None  # out of range


def test_resolve_prefers_class_name():
    assert resolve_label(72, "person") == "person"  # would be tv by id alone
    assert resolve_label(72, None) == "tv"
    assert resolve_label(1, "PERSON") == "person"
