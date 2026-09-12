import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from flypv.connectome.circuits import FlightCircuit
from flypv.connectome.loader import Connectome
from flypv.policy.connectome_policy import ConnectomePolicy, PolicyConfig
from flypv.train.config import TrainConfig


class StatelessPolicyTests(unittest.TestCase):
    def _policy(self):
        # 0 -> 1 -> 2, with one sensory input and one power output.
        W = sp.csr_matrix(
            (np.array([8.0, 8.0], dtype=np.float32),
             (np.array([1, 2]), np.array([0, 1]))),
            shape=(3, 3),
        )
        meta = pd.DataFrame({
            "type": ["sensor", "inter", "DLMn a, b"],
            "superclass": ["vnc_sensory", "vnc_intrinsic", "vnc_motor"],
            "somaSide": ["R", "R", "R"],
        })
        cx = Connectome(
            body_ids=np.arange(3, dtype=np.int64),
            W=W,
            meta=meta,
            nt_sign=np.ones(3, dtype=np.float32),
        )
        circuit = FlightCircuit(
            cx=cx,
            afferent={"retina": np.array([], dtype=np.int64),
                      "haltere": np.array([0], dtype=np.int64)},
            efferent={"power": np.array([2], dtype=np.int64)},
            intrinsic=np.array([1], dtype=np.int64),
            hex_coords={},
        )
        return ConnectomePolicy(circuit, 40, 15, PolicyConfig(n_iters=3), "cpu")

    def test_previous_state_does_not_change_output(self):
        torch.manual_seed(0)
        policy = self._policy()
        obs = {"proprio": torch.randn(2, 40)}
        zero = torch.zeros(3, 2)
        arbitrary = torch.randn(3, 2) * 100
        out0 = policy.forward(obs, zero)
        out1 = policy.forward(obs, arbitrary)
        for a, b in zip(out0, out1):
            torch.testing.assert_close(a, b)

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
