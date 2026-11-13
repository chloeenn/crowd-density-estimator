import numpy as np

from crowd.detect import Detection, draw_detections, label_color


def test_label_color_is_stable_per_class_and_differs_between_classes():
    assert label_color("car") == label_color("car")
    assert label_color("car") != label_color("person")


def test_draw_detections_marks_the_box_and_tag_without_touching_the_input():
    img = np.full((100, 200, 3), 128, np.uint8)
    out = draw_detections(img, [Detection("car", 0.9, (50, 40, 150, 90))])
    assert (img == 128).all()
    assert tuple(out[60, 50]) == label_color("car")  # left edge of the box
    assert not (out[30:40, 50:80] == 128).all()  # name tag above the box
    assert (out[60, 100] == 128).all()  # interior untouched


def test_tag_stays_on_image_for_a_box_at_the_top_edge():
    out = draw_detections(
        np.zeros((50, 50, 3), np.uint8), [Detection("person", 1.0, (5, 0, 40, 40))]
    )
    assert out.shape == (50, 50, 3)
