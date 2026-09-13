import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from flypv.connectome.circuits import FlightCircuit
from flypv.connectome.loader import Connectome
from flypv.policy.connectome_policy import ConnectomePolicy, PolicyConfig
from flypv.train.config import TrainConfig


class ConnectomeMaskPolicyTests(unittest.TestCase):
    def _circuit(self, weights=(8.0, -3.0), signs=(1.0, -1.0, 1.0)):
        # Directed chain 0 -> 1 -> 2. Magnitudes/signs should not constrain the
        # learned network; only these two allowed coordinates should survive.
        W = sp.csr_matrix(
            (np.asarray(weights, dtype=np.float32),
             (np.array([1, 2]), np.array([0, 1]))),
            shape=(3, 3),
        )
        meta = pd.DataFrame({
            "type": ["sensor", "inter", "DLMn a, b"],
            "superclass": ["vnc_sensory", "vnc_intrinsic", "vnc_motor"],
            "somaSide": ["R", "R", "R"],
        })
        cx = Connectome(
            body_ids=np.arange(3, dtype=np.int64), W=W, meta=meta,
            nt_sign=np.asarray(signs, dtype=np.float32),
        )
        return FlightCircuit(
            cx=cx,
            afferent={"retina": np.array([], dtype=np.int64),
                      "haltere": np.array([0], dtype=np.int64)},
            efferent={"power": np.array([2], dtype=np.int64)},
            intrinsic=np.array([1], dtype=np.int64),
            hex_coords={},
        )

    def _policy(self, circuit=None):
        torch.manual_seed(0)
        return ConnectomePolicy(
            circuit or self._circuit(), 40, 15,
            PolicyConfig(n_iters=3, learn_synapses=True), "cpu"
        )

    def test_previous_state_does_not_change_output(self):
        policy = self._policy()
        obs = {"proprio": torch.randn(2, 40)}
        zero = torch.zeros(3, 2)
        arbitrary = torch.randn(3, 2) * 100
        out0 = policy.forward(obs, zero)
        out1 = policy.forward(obs, arbitrary)
        for a, b in zip(out0, out1):
            torch.testing.assert_close(a, b)

    def test_mask_edges_are_trainable_and_receive_gradient(self):
        policy = self._policy()
        self.assertEqual(tuple(policy.edge_weight.shape), (3, 2))
        self.assertTrue(policy.edge_weight.requires_grad)
        obs = {"proprio": torch.randn(4, 40)}
        mean, _, _, _ = policy.forward(obs)
        mean.square().mean().backward()
        self.assertIsNotNone(policy.edge_weight.grad)
        self.assertTrue(torch.isfinite(policy.edge_weight.grad).all())
        self.assertGreater(float(policy.edge_weight.grad.abs().sum()), 0.0)

    def test_synapse_magnitude_and_transmitter_sign_do_not_set_weights(self):
        torch.manual_seed(123)
        p1 = ConnectomePolicy(
            self._circuit(weights=(1.0, -100.0), signs=(1.0, -1.0, 1.0)),
            40, 15, PolicyConfig(n_iters=2), "cpu"
        )
        torch.manual_seed(123)
        p2 = ConnectomePolicy(
            self._circuit(weights=(900.0, 2.0), signs=(-1.0, 0.0, -1.0)),
            40, 15, PolicyConfig(n_iters=2), "cpu"
        )
        torch.testing.assert_close(p1.edge_index, p2.edge_index)
        torch.testing.assert_close(p1.edge_weight, p2.edge_weight)

    def test_default_config_learns_edges_and_uses_six_layers(self):
        cfg = TrainConfig()
        self.assertTrue(cfg.learn_synapses)
        self.assertEqual(cfg.n_iters, 6)

    def test_legacy_recurrent_config_fields_are_ignored(self):
        cfg = TrainConfig.from_dict({
            "recurrent": True,
            "tbptt_steps": 32,
            "state_carry": 0.9,
        })
        self.assertFalse(hasattr(cfg, "recurrent"))
        self.assertFalse(hasattr(cfg, "tbptt_steps"))
        self.assertFalse(hasattr(cfg, "state_carry"))


if __name__ == "__main__":
    unittest.main()
