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
    group_regions,
    hole_scale,
    mask_bbox,
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
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            boxes.append((y0, x0, y1 + 1, x1 + 1))
    return sorted(boxes)


class CountRuns(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count_runs(np.zeros((5, 5), dtype=bool)), 0)

    def test_rows(self):
        m = np.zeros((3, 10), dtype=bool)
        m[0, 1:4] = True          # one run
        m[1, [0, 2, 4, 6]] = True  # four runs
        m[2, :] = True             # one run
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
        (y0, x0, y1, x1), = group_regions([(50, 50, 52, 52)], 0.0, 8, 100, 100)
        self.assertEqual((y0, x0, y1, x1), (42, 42, 60, 60))


if __name__ == "__main__":
    unittest.main()
