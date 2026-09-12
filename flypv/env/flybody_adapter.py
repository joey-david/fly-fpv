"""Swapping in the DeepMind / Janelia musculoskeletal fly, if you want it.

`HoopRaceEnv` deliberately uses its own 6-DOF rigid body with two blade-element
wings. That buys ~2,000 environment steps a second on a laptop with no MuJoCo
dependency, and for a task that is entirely about airborne flight it models the
part that matters — unsteady flapping aerodynamics — at full fidelity.

What it does not model is everything below the thorax: leg contact, adhesion,
the ~100 body degrees of freedom, and the inertial coupling of the abdomen
beyond the centre-of-mass shift we apply. If you want takeoff, landing, or
walking-to-flight transitions, you want the real body model.

    pip install -e ".[body]"
    pip install git+https://github.com/TuragaLab/flybody

The pieces that carry over unchanged:

  * the policy — `ConnectomePolicy` only needs a proprioceptive vector and the
    two retinal channels, whatever produced them;
  * the motor decoder — flybody's flight task takes 12 controls covering wing
    torques, head and abdomen angles, and wing-pattern-generator frequency, which
    is the same structure as our 15, so `MotorDecoder.WIRING` needs its channel
    names remapped and nothing else;
  * the reward shape — progress, gate clearance, power, jerk.

What has to be rebuilt: the hoop course and the gate-crossing test become MuJoCo
geometry and contact queries, and the compound eye becomes flybody's own
eye cameras rather than our ray-caster — at which point you lose the exact
ommatidium-to-neuron mapping that `assignedOlHex1/2` gives us here, and would
need to resample the rendered image onto the hex lattice yourself.

This module is documentation rather than code on purpose: a half-built adapter
that silently disagrees with the real one is worse than none.
"""

FLYBODY_REPO = "https://github.com/TuragaLab/flybody"
FLYBODY_PAPER = "https://www.nature.com/articles/s41586-025-09029-4"


def available() -> bool:
    try:
        import flybody  # noqa: F401
        return True
    except ImportError:
        return False
