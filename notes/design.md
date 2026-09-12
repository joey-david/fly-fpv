# Design notes

Implementation decisions that materially affect the experiment.

## Connectome filtering

The raw MaleCNS weights table contains segment-level reconstruction fragments.
`loader.py` keeps traced bodies and edges with at least five predicted synapses,
then compacts the result into a signed CSR matrix. Neurotransmitter sign belongs
to the presynaptic neuron; unresolved/modulatory cells are handled separately by
the policy.

## Flight-circuit extraction

The default circuit takes the union of flight sensory/motor seeds and neurons on
short paths between them. In the current MaleCNS cache this is about 28k neurons
and 535k connections. The two-hop extraction is a practical sensorimotor subset,
not a claim that all relevant biological computation must finish in four graph
operations.

The policy now carries neural state across 5 ms control ticks. Four sparse graph
iterations happen per tick by default, while temporal credit assignment is
truncated separately (16 ticks / 80 ms by default). This removes the old,
unnecessary coupling between graph-extraction depth and temporal memory.

## Recorded bugs and corrections

**Moment of inertia was off by 10³.** The original value mixed grams and
kilograms. Correcting the body inertia moved angular acceleration back into a
plausible range and stopped the integrator exploding immediately.

**Tanh likelihood correction was inconsistent.** Sampling and PPO likelihood
recomputation now both use the same Jacobian-corrected log probability.

**Time-limit truncation was treated like death.** PPO now bootstraps from the
final observation at a pure episode timeout and only treats actual terminations
as zero-value endpoints.

**The nominal wing kinematics did not hover.** `wing/trim.py` solves for stroke
offset, amplitude scale and stroke-plane tilt so zero action is a balanced trim
rather than an immediate tumble.

**The reflex pilot initially asked for impossible angular accelerations.** It now
uses a cascaded attitude/rate controller with bounded desired body rates.

**The retina consumed small circuit/display budgets.** Both circuit pruning and
the monitor explicitly reserve capacity for non-visual sensorimotor tissue.

**The old degree-preserving baseline was not degree-preserving after sparse
coalescing.** Independent endpoint permutations could create duplicate edges and
could attach an inhibitory weight to a different presynaptic cell. The control
now uses directed double-edge swaps, preserving every neuron's in/out degree and
each source neuron's outgoing weight/sign multiset.

**The aerodynamic module claimed an added-mass term that was identically zero.**
The dead placeholder and unused wing-mass field were removed. The implemented
plant claims only the terms it actually computes: translational lift/drag,
rotational circulation, body drag, and the resulting torques/power.

## Control authority

Measured near the hover trim, per unit normalized control (torque normalized by
weight × wing length, force by weight):

| control | principal effect |
|---|---:|
| differential stroke amplitude | roll 0.173 |
| collective stroke offset | pitch −0.211 |
| differential angle of attack | yaw 0.329 |
| collective amplitude | lift 0.760 |
| stroke-plane tilt | thrust 0.512 |

These coefficients seed the reflex expert; PPO is not given the inverse control
map directly.

## Throughput choices

The simulator keeps several deliberately cheap operations: vectorized parallel
environments, a pre-generated course pool, compound-eye rendering below the
control rate, and ray-casting only nearby gates. The monitor is latest-value-wins
and runs on a separate server thread so browser rendering cannot backpressure
training.

## Reproducibility and restart semantics

A checkpoint contains policy and optimizer states, PPO counters, curriculum
difficulty, recent episode statistics, RNG state, metrics and the resolved
configuration. Numbered checkpoints are atomically written and tracked by
`latest.json` and `best.json`.

Resume intentionally starts new physical episodes instead of serializing the
entire vectorized simulator. The learning process continues; the exact physical
trajectory does not. This keeps checkpoints compact and makes the restart
semantics explicit.
