import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from edge_vision.geometry import BBox, Line, Point, Polygon, Side

COORD = st.floats(min_value=-1000, max_value=1000, allow_nan=False)
POINT = st.builds(Point, COORD, COORD)


@st.composite
def boxes(draw: st.DrawFn) -> BBox:
    x1, y1 = draw(COORD), draw(COORD)
    width = draw(st.floats(min_value=1, max_value=500))
    height = draw(st.floats(min_value=1, max_value=500))
    return BBox(x1, y1, x1 + width, y1 + height)


class TestBBox:
    def test_anchor_is_bottom_centre(self) -> None:
        assert BBox(10, 20, 30, 60).anchor == Point(20, 60)

    def test_area(self) -> None:
        assert BBox(0, 0, 4, 5).area == 20

    def test_iou_of_partial_overlap(self) -> None:
        assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_iou_of_disjoint_boxes_is_zero(self) -> None:
        assert BBox(0, 0, 1, 1).iou(BBox(2, 2, 3, 3)) == 0

    def test_iou_with_degenerate_boxes_is_zero(self) -> None:
        assert BBox(0, 0, 0, 0).iou(BBox(0, 0, 0, 0)) == 0

    def test_rejects_inverted_corners(self) -> None:
        with pytest.raises(ValueError, match="corner"):
            BBox(10, 0, 5, 5)

    def test_clip_limits_box_to_frame(self) -> None:
        assert BBox(-5, -5, 50, 50).clip(40, 30) == BBox(0, 0, 40, 30)

    @given(boxes(), boxes())
    def test_iou_is_symmetric_and_bounded(self, a: BBox, b: BBox) -> None:
        assert a.iou(b) == pytest.approx(b.iou(a))
        assert 0 <= a.iou(b) <= 1

    @given(boxes())
    def test_iou_with_itself_is_one(self, box: BBox) -> None:
        assert box.iou(box) == pytest.approx(1)


class TestPolygon:
    SQUARE = Polygon.of([(0, 0), (10, 0), (10, 10), (0, 10)])
    L_SHAPE = Polygon.of([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])

    @pytest.mark.parametrize(
        ("point", "inside"),
        [((5, 5), True), ((11, 5), False), ((-1, 5), False), ((5, 15), False)],
    )
    def test_contains_for_convex_polygon(self, point: tuple[float, float], inside: bool) -> None:
        assert self.SQUARE.contains(Point(*point)) is inside

    @pytest.mark.parametrize(("point", "inside"), [((2, 8), True), ((8, 2), True), ((8, 8), False)])
    def test_contains_for_concave_polygon(self, point: tuple[float, float], inside: bool) -> None:
        assert self.L_SHAPE.contains(Point(*point)) is inside

    def test_requires_three_vertices(self) -> None:
        with pytest.raises(ValueError, match="three"):
            Polygon.of([(0, 0), (1, 1)])

    @given(boxes(), st.floats(0.01, 0.99), st.floats(0.01, 0.99))
    def test_points_strictly_inside_a_rectangle_are_contained(
        self, box: BBox, fx: float, fy: float
    ) -> None:
        polygon = Polygon.of(
            [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]
        )
        point = Point(box.x1 + fx * (box.x2 - box.x1), box.y1 + fy * (box.y2 - box.y1))

        assert polygon.contains(point)


class TestLine:
    DOWNWARD = Line(Point(0, 0), Point(0, 10))

    def test_side_follows_image_coordinates(self) -> None:
        assert self.DOWNWARD.side(Point(-5, 5)) is Side.RIGHT
        assert self.DOWNWARD.side(Point(5, 5)) is Side.LEFT
        assert self.DOWNWARD.side(Point(0, 5)) is Side.ON

    def test_detects_crossing_and_its_direction(self) -> None:
        assert self.DOWNWARD.crossing(Point(5, 5), Point(-5, 5)) is Side.RIGHT
        assert self.DOWNWARD.crossing(Point(-5, 5), Point(5, 5)) is Side.LEFT

    def test_movement_beyond_segment_end_is_not_a_crossing(self) -> None:
        assert self.DOWNWARD.crossing(Point(5, 20), Point(-5, 20)) is None

    def test_movement_on_one_side_is_not_a_crossing(self) -> None:
        assert self.DOWNWARD.crossing(Point(5, 5), Point(1, 5)) is None

    def test_touching_the_line_is_not_a_crossing(self) -> None:
        assert self.DOWNWARD.crossing(Point(5, 5), Point(0, 5)) is None

    def test_rejects_zero_length_line(self) -> None:
        with pytest.raises(ValueError, match="length"):
            Line(Point(1, 1), Point(1, 1))

    @given(POINT, POINT, POINT, POINT)
    def test_reversed_movement_reports_opposite_side(
        self, a: Point, b: Point, start: Point, end: Point
    ) -> None:
        assume(a != b)
        line = Line(a, b)
        forward = line.crossing(start, end)
        backward = line.crossing(end, start)

        assert (forward is None) == (backward is None)
        if forward is not None:
            assert backward is not None
            assert forward is not backward
