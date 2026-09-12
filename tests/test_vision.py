import unittest

import numpy as np

from flypv.env.vision import _annulus_coverage, _circle_overlap_area


class VisionCoverageTests(unittest.TestCase):
    def test_subpixel_ring_is_fractional(self):
        # The center ray is still inside the gate hole, but a finite receptor
        # footprint overlaps the ring. Point sampling used to return zero here.
        coverage = _annulus_coverage(
            np.array([0.045]),
            np.array([0.050]),
            np.array([0.060]),
            np.array([0.010]),
        )[0]
        self.assertGreater(coverage, 0.0)
        self.assertLess(coverage, 1.0)

    def test_footprint_fully_inside_ring_is_full(self):
        coverage = _annulus_coverage(
            np.array([0.055]),
            np.array([0.050]),
            np.array([0.060]),
            np.array([0.001]),
        )[0]
        self.assertAlmostEqual(float(coverage), 1.0, places=12)

    def test_footprint_inside_hole_is_zero(self):
        coverage = _annulus_coverage(
            np.array([0.0]),
            np.array([0.050]),
            np.array([0.060]),
            np.array([0.010]),
        )[0]
        self.assertEqual(float(coverage), 0.0)

    def test_circle_containment(self):
        area = _circle_overlap_area(
            np.array([0.020]), np.array([0.010]), np.array([0.0])
        )[0]
        self.assertAlmostEqual(float(area), np.pi * 0.010**2, places=14)


if __name__ == "__main__":
    unittest.main()
