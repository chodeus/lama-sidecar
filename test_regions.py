"""Unit tests for the torch-free geometry in regions.py.

Runs with numpy + Pillow only:
    python test_regions.py
"""

import unittest

import numpy as np
from PIL import Image, ImageFilter

from regions import (
    connected_components,
    count_runs,
    dilate,
    erode,
    group_regions,
    hole_scale,
    labeled_components,
    mask_bbox,
    snap_mask,
)


def _naive_components(mask):
    """Ground-truth 8-connected bboxes via BFS flood fill."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            y0 = y1 = sy
            x0 = x1 = sx
            while stack:
                y, x = stack.pop()
                y0, y1 = min(y0, y), max(y1, y)
                x0, x1 = min(x0, x), max(x1, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and mask[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            boxes.append((y0, x0, y1 + 1, x1 + 1))
    return sorted(boxes)


class CountRuns(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count_runs(np.zeros((5, 5), dtype=bool)), 0)

    def test_rows(self):
        m = np.zeros((3, 10), dtype=bool)
        m[0, 1:4] = True  # one run
        m[1, [0, 2, 4, 6]] = True  # four runs
        m[2, :] = True  # one run
        self.assertEqual(count_runs(m), 6)

    def test_bounds_components(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            m = rng.random((40, 40)) > 0.7
            self.assertGreaterEqual(count_runs(m), len(connected_components(m)))


class Dilate(unittest.TestCase):
    def test_identity_at_zero(self):
        m = np.random.default_rng(1).random((20, 20)) > 0.5
        self.assertTrue((dilate(m, 0) == m).all())

    def test_matches_pil_maxfilter(self):
        rng = np.random.default_rng(3)
        m = rng.random((64, 48)) > 0.85
        for px in (1, 3, 5):
            img = Image.fromarray(m.astype("uint8") * 255)
            for _ in range(px):
                img = img.filter(ImageFilter.MaxFilter(3))
            expected = np.asarray(img) > 127
            self.assertTrue((dilate(m, px) == expected).all(), f"px={px}")


class MaskBbox(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(mask_bbox(np.zeros((4, 4), dtype=bool)))

    def test_extent(self):
        m = np.zeros((10, 12), dtype=bool)
        m[2:5, 3:9] = True
        self.assertEqual(mask_bbox(m), (2, 3, 5, 9))


class HoleScale(unittest.TestCase):
    def test_thin_band_stays_native(self):
        # A wide title wordmark: 1685x260 — thin enough for context to reach.
        self.assertEqual(hole_scale(1685, 260, 512, 384), 1.0)

    def test_small_hole_native(self):
        self.assertEqual(hole_scale(50, 40, 512, 384), 1.0)

    def test_bulky_hole_downscales_to_easier_criterion(self):
        # 800x800: long-edge rule gives 0.64, thickness rule 0.48 — take 0.64.
        self.assertAlmostEqual(hole_scale(800, 800, 512, 384), 0.64)

    def test_never_upscales(self):
        self.assertLessEqual(hole_scale(10000, 9000, 512, 384), 1.0)


class ConnectedComponents(unittest.TestCase):
    def test_single_blob(self):
        m = np.zeros((10, 10), dtype=bool)
        m[2:6, 3:8] = True
        self.assertEqual(connected_components(m), [(2, 3, 6, 8)])

    def test_diagonal_touch_merges(self):
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = m[1, 1] = True  # 8-connectivity joins diagonals
        self.assertEqual(len(connected_components(m)), 1)

    def test_separate_blobs(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1, 1] = True
        m[8, 8] = True
        self.assertEqual(len(connected_components(m)), 2)

    def test_matches_naive_on_random_masks(self):
        rng = np.random.default_rng(42)
        for density in (0.2, 0.5, 0.8):
            m = rng.random((40, 40)) < density
            self.assertEqual(sorted(connected_components(m)), _naive_components(m))


class Erode(unittest.TestCase):
    def test_identity_at_zero(self):
        m = np.random.default_rng(2).random((20, 20)) > 0.5
        self.assertTrue((erode(m, 0) == m).all())

    def test_shrinks_interior_blob(self):
        m = np.zeros((30, 30), dtype=bool)
        m[5:25, 5:25] = True
        expected = np.zeros_like(m)
        expected[8:22, 8:22] = True
        self.assertTrue((erode(m, 3) == expected).all())

    def test_border_side_not_eroded(self):
        # Beyond-edge counts as True, so a mask touching the border keeps it.
        m = np.zeros((20, 20), dtype=bool)
        m[0:10, 0:10] = True
        out = erode(m, 3)
        self.assertTrue(out[0:7, 0:7].all())
        self.assertFalse(out[7:10, :].any())
        self.assertFalse(out[:, 7:10].any())


class LabeledComponents(unittest.TestCase):
    def test_labels_cover_exactly_the_mask(self):
        rng = np.random.default_rng(11)
        m = rng.random((50, 50)) > 0.6
        labels, n = labeled_components(m)
        self.assertTrue(((labels > 0) == m).all())
        self.assertEqual(n, len(connected_components(m)))

    def test_one_label_per_component(self):
        rng = np.random.default_rng(13)
        for density in (0.3, 0.6):
            m = rng.random((40, 40)) < density
            labels, n = labeled_components(m)
            # Every component bbox from the label image must match the
            # bbox list from connected_components.
            got = []
            for i in range(1, n + 1):
                got.append(mask_bbox(labels == i))
            self.assertEqual(sorted(got), sorted(connected_components(m)))


class SnapMask(unittest.TestCase):
    @staticmethod
    def _flat_rgb(h, w, value=200):
        return np.full((h, w, 3), value, dtype=np.uint8)

    def test_swallows_clipped_stroke(self):
        # A black bar the hole covers ~73% of: the clipped tail must be
        # swallowed so no stub is left at the hole edge.
        rgb = self._flat_rgb(60, 80)
        rgb[20:30, 10:40] = 0
        hole = np.zeros((60, 80), dtype=bool)
        hole[15:35, 5:32] = True
        grown, added = snap_mask(rgb, hole)
        self.assertGreater(added, 0)
        self.assertTrue(grown[20:30, 10:40].all())
        self.assertTrue(grown[hole].all(), "snap must be purely additive")

    def test_keeps_grazed_structure(self):
        # The hole merely grazes a big rectangle (~1% covered): that is
        # brush intent — the structure must NOT be swallowed.
        rgb = self._flat_rgb(60, 90)
        rgb[10:50, 40:75] = 0
        hole = np.zeros((60, 90), dtype=bool)
        hole[28:32, 36:42] = True
        grown, added = snap_mask(rgb, hole)
        self.assertEqual(added, 0)
        self.assertTrue((grown == hole).all())

    def test_scene_scale_component_ignored(self):
        # Mostly-covered but far larger than the hole (area cap): keep it.
        rgb = self._flat_rgb(60, 200)
        rgb[10:50, 20:180] = 0
        hole = np.zeros((60, 200), dtype=bool)
        hole[8:52, 18:150] = True  # covers ~81% of a 6400px component
        grown, added = snap_mask(rgb, hole, max_area_frac=0.5)
        self.assertEqual(added, 0)

    def test_strong_level_rescues_soft_fused_stroke(self):
        # A black bar sitting on a soft shadow: at the soft threshold the bar
        # fuses with the (mostly-outside) shadow and fails the overlap test,
        # but the strong pass sees the bar alone and swallows its clipped tail.
        rgb = self._flat_rgb(60, 120)
        rgb[20:40, 10:110] = 150  # soft shadow, |d|=50: soft-level content only
        rgb[25:35, 20:50] = 0  # black bar crossing the hole edge
        hole = np.zeros((60, 120), dtype=bool)
        hole[22:38, 15:44] = True
        grown, added = snap_mask(rgb, hole)
        self.assertGreater(added, 0)
        self.assertTrue(grown[25:35, 20:50].all())
        # The shadow's own outside area must not be swallowed wholesale.
        self.assertFalse(grown[20:40, 60:110].any())

    def test_reach_caps_distant_tail(self):
        # A qualifying component whose tail runs far from the hole: only the
        # part within `reach` px of the hole may be swallowed.
        rgb = self._flat_rgb(40, 200)
        rgb[15:25, 10:150] = 0  # long bar, ~71% inside the hole
        hole = np.zeros((40, 200), dtype=bool)
        hole[10:30, 5:110] = True
        grown, added = snap_mask(rgb, hole, reach=6)
        self.assertGreater(added, 0)
        self.assertTrue(grown[15:25, 110:116].all())  # collar swallowed
        self.assertFalse(grown[15:25, 117:150].any())  # far tail kept

    def test_noop_on_busy_content(self):
        rng = np.random.default_rng(17)
        rgb = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        hole = np.zeros((60, 80), dtype=bool)
        hole[20:40, 20:60] = True
        grown, added = snap_mask(rgb, hole, max_runs=50)
        self.assertEqual(added, 0)
        self.assertTrue((grown == hole).all())

    def test_empty_hole_passthrough(self):
        rgb = self._flat_rgb(20, 20)
        hole = np.zeros((20, 20), dtype=bool)
        grown, added = snap_mask(rgb, hole)
        self.assertEqual(added, 0)


class GroupRegions(unittest.TestCase):
    def test_nearby_boxes_merge(self):
        boxes = [(10, 10, 20, 20), (22, 22, 30, 30)]
        out = group_regions(boxes, 0.5, 8, 100, 100)
        self.assertEqual(len(out), 1)

    def test_far_boxes_stay_separate(self):
        boxes = [(0, 0, 4, 4), (80, 80, 84, 84)]
        out = group_regions(boxes, 0.5, 8, 100, 100)
        self.assertEqual(len(out), 2)

    def test_clamped_to_bounds(self):
        out = group_regions([(0, 0, 5, 5)], 0.5, 20, 30, 30)
        self.assertEqual(out, [(0, 0, 25, 25)])

    def test_min_pad_applied(self):
        ((y0, x0, y1, x1),) = group_regions([(50, 50, 52, 52)], 0.0, 8, 100, 100)
        self.assertEqual((y0, x0, y1, x1), (42, 42, 60, 60))


if __name__ == "__main__":
    unittest.main()
