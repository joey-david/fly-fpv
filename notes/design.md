# Design notes

Decisions that took measurement rather than judgement, and the numbers behind them.

## Filtering 151.8 M edges down to 535 k

The raw `connectome-weights` table is segment-to-segment across every
reconstruction fragment. Two filters: bodies with `status == "Traced"` (165,122
of 211,577), and edges of at least 5 synapses — the conventional threshold below
which automatic synapse prediction's false-discovery rate dominates. Result:
6,235,682 connections carrying 89,731,552 synapses, 63% excitatory.

## Why the flight circuit is 28 k neurons and not 166 k

Forward BFS 2 hops from the flight sensors, backward BFS 2 hops from the flight
muscles, intersect. 139,266 neurons are within two synapses of *some* sensor —
mostly the enormous optic lobe — but only 13,633 are within two of a muscle, and
12,891 sit on a path between the two. Adding the seeds gives 28,361. This is the
tissue that can influence a wing inside the latency budget of a wingbeat, and it
is consistent with the policy's four-hop settling.

## Bugs worth recording

**Moment of inertia off by 10³.** Written as 1.16e-16 kg·m² from computing
`(2/5) m b²` with the mass in grams. Peak aerodynamic torque of 1.5e-8 N·m then
implies 1.3e8 rad/s² and the integrator produced NaN within one control step.
Correct value is 1.16e-13, which puts peak within-beat angular acceleration at
5.2e4 rad/s² — a body oscillation of a few degrees per stroke, which is what is
measured in real flies.

**The tanh correction applied in only one place.** `act()` returned the raw
Gaussian log-probability while the PPO update recomputed it with the tanh
Jacobian subtracted. Every importance ratio carried a large constant offset, and
KL sat at 6–8 instead of 0.02. Both paths now go through `squashed_log_prob`.

**Textbook kinematics do not hover.** 140° amplitude, 218 Hz, 45° angle of
attack, −15° stroke plane gives 0.95 body weights and a residual nose-up torque
of ~2100 rad/s²; the fly tumbles in 30 ms and every episode ended in a crash
before the agent could learn anything. `wing/trim.py` solves the three-parameter
system instead — stroke offset, amplitude scale, stroke-plane tilt — against
zero net force and zero pitch torque. It converges to +7.4°, 141°, 0.0°, giving
1.0000 body weights, no drift, and passive rate damping.

**A single-stage PD tumbles executing its own command.** Attitude error up to π
with ω_n = 70 rad/s asks for 15,000 rad/s², which crosses the tumble threshold
in two control steps. The cascaded form — angle error sets a desired body rate,
clamped at 35 rad/s (a real fly's saccade peak), and an inner loop drives the
rate — took the reflex pilot from 0 gates to 2.5 of 6.

**The retina eats every budget it is given.** 14,201 of the flight circuit's
28,361 neurons are retinotopic. Both the `core` circuit pruning and the
monitor's display sampling initially kept all afferents first and had nothing
left for the brain. Both now subsample the eye — by whole ommatidial column, so
the map stays coherent — and spend the rest on interneurons.

## Measured control authority

Per unit control, torque normalised by weight × wing length, force by weight:

| | effect |
|---|---|
| differential stroke amplitude | roll 0.173 |
| collective stroke offset | pitch −0.211 |
| differential angle of attack | yaw 0.329 |
| collective amplitude | lift 0.760 |
| stroke-plane tilt | thrust 0.512 |

The destabilising pitch torque from flying at 0.3 m/s is 0.0215 — about a fifth
of the collective-offset authority, so the plant is comfortably controllable.

## Throughput

| | env steps/s |
|---|---|
| 16 envs, naive | 412 |
| 64 envs, after vision cadence + gate windowing + course pool | 1,811 |

The three fixes: render the eye at 50 Hz rather than 200, ray-cast only the four
gates that can be in front of the fly instead of all eight, and draw resets from
a pre-generated pool of 256 courses rather than building one per reset.
