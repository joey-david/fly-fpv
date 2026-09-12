import unittest

import numpy as np
import scipy.sparse as sp

from flypv.connectome.circuits import _recurrent_closure
from flypv.policy.connectome_policy import SENSOR_FEATURES, _soft_row_normalize


class CircuitExtractionTests(unittest.TestCase):
    def test_recurrent_closure_keeps_feedback_not_one_way_fanout(self):
        # W[post, pre]. Base={0,1}. Neuron 2 receives from 0 and sends to 1, so
        # it closes a local feedback/side loop. Neurons 3 and 4 are one-way only.
        row = np.array([2, 1, 3, 0])
        col = np.array([0, 2, 0, 4])
        W = sp.csr_matrix((np.ones(4), (row, col)), shape=(5, 5))
        got = _recurrent_closure(W, np.array([0, 1]), hops=1)
        self.assertEqual(set(got.tolist()), {0, 1, 2})


class NormalisationTests(unittest.TestCase):
    def test_strict_row_norm_matches_unit_total(self):
        W = sp.csr_matrix(np.diag([10.0, 40.0, 160.0]))
        got = _soft_row_normalize(W, power=1.0)
        totals = np.asarray(np.abs(got).sum(axis=1)).ravel()
        np.testing.assert_allclose(totals, np.ones(3), rtol=1e-6)

    def test_default_preserves_sublinear_convergence_strength(self):
        W = sp.csr_matrix(np.diag([10.0, 40.0, 160.0]))
        got = _soft_row_normalize(W, power=0.5)
        totals = np.asarray(np.abs(got).sum(axis=1)).ravel()
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])
        self.assertAlmostEqual(float(totals[1]), 1.0, places=6)


class SensoryRoutingTests(unittest.TestCase):
    def test_actor_sensors_do_not_receive_privileged_gate_features(self):
        # HoopRaceEnv proprio indices 26..39 encode explicit next-gate geometry.
        for name, indices in SENSOR_FEATURES.items():
            with self.subTest(sensor=name):
                self.assertLess(max(indices), 26)


if __name__ == "__main__":
    unittest.main()
