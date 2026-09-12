# flypv

Reinforcement learning on a real fruit fly connectome. The wiring diagram of
166,000 neurons from Janelia and Google's **MaleCNS v1.0** release is loaded as a
fixed policy network, given a physically simulated fly body with flapping wings,
and trained to fly a 3D hoop course as fast, as cheaply, and as smoothly as it can.

The premise borrowed from the Twitter/X posts that started this — people wiring
the connectome up to Trackmania, to parallel parking, to Deadlock — is that the
fly brain is an interesting *substrate* to train. This repo takes that seriously
and gives it the task it was actually evolved for.

```
flypv fetch                 # ~1.2 GB of connectome from Janelia's public bucket
flypv build                 # filter and cache it
flypv circuit               # show the flight circuit it extracts
flypv train                 # train, with the live monitor on :8777
```

---

## What is real here

This is the part worth being precise about, because "training the connectome" can
mean almost anything.

**The connectivity is measured, not learned.** `W` is 534,681 connections between
28,361 real neurons, each carrying its measured synapse count. It is a PyTorch
*buffer*, not a parameter — gradient descent never changes who talks to whom.

**Every synapse has a biologically determined sign.** Excitatory or inhibitory is
read off the presynaptic neuron's predicted neurotransmitter (acetylcholine and
histamine positive, GABA and glutamate negative), resolved for 97.5% of neurons.
The network comes out 63% excitatory, which is what the fly is.

**The inputs land on the neurons that really receive them.** MaleCNS annotates
each medulla columnar cell with `assignedOlHex1/2`, the ommatidium its column
belongs to. There are ~800 per eye, matching the real eye, so the renderer
ray-casts through that lattice and writes each ommatidium's output onto its own
neurons — and splits it into the ON and OFF channels that L1/L5/Mi1 and
L2/L3/Tm1/Tm2/Tm9 actually carry. Body rotation goes to the 205 haltere
afferents; airspeed to the 672 Johnston's organ neurons; wing strain to the wing
campaniform sensilla.

**The outputs are the muscles.** The action vector is not abstract. It is the 12
power and steering muscles a fly steers with — DLM and DVM for the wingbeat
oscillation, b1/b2/b3, i1/i2, iii1/iii3, hg1–hg4, ps1/ps2, tp1/tp2 for the hinge —
plus neck and abdomen. The readout from motor-neuron activity to wing controls is
initialised to the known anatomy and then refined.

**The aerodynamics are quasi-steady blade-element, not a thrust vector.** Lift and
drag coefficients are the Sane & Dickinson measurements, with rotational
circulation and body drag. At nominal kinematics the model produces **1.000 body
weights** of lift at **19 µW**, about 67 W per kg of flight muscle — both inside
the published range for *Drosophila*, with no tuning.

## What is a modelling choice

Stated plainly so nothing here is oversold:

- Neurons are **leaky rate units**, not spiking. Learnable per neuron: gain,
  threshold, membrane time constant, and the sign for the 2.5% with no confident
  transmitter call. 355,430 trainable parameters against 534,681 fixed synapses.
- **Four synaptic hops per control step.** Information has to physically cross
  several synapses to get from an eye to a wing; four is the budget within one
  wingbeat, and the circuit is extracted to be consistent with it.
- The **non-visual sensory encoder is learned**. Halteres genuinely encode
  angular velocity, but the connectome does not label which afferent is tuned to
  which axis, so a small linear map is trained instead of invented.
- The body is a **6-DOF rigid body with two blade-element wings**, not the full
  musculoskeletal model. `flypv/env/flybody_adapter.py` documents swapping in the
  DeepMind/Janelia MuJoCo fly if you want contact and leg dynamics.
- The wingbeat comes from a **pattern generator** the policy modulates. This is
  not a shortcut: *Drosophila* flight muscle is asynchronous and the thorax is a
  mechanical resonator, so no neuron commands individual strokes. The policy gets
  the same knobs the fly has.

## The controls

15 channels, all in [-1, 1], zero meaning hover. The hover trim — stroke offset
+7.4°, amplitude 141°, horizontal stroke plane — is **solved numerically**, not
assumed, so that "do nothing" balances weight and pitch exactly.

| | control | muscle |
|---|---|---|
| 0 | wingbeat frequency | DLM / DVM power muscles |
| 1–2 | stroke amplitude, per side | power + b1 |
| 3–4 | stroke offset, per side | pterale, ps1/ps2 |
| 5–6 | stroke deviation, per side | axillary |
| 7–10 | angle of attack, up and down stroke, per side | hg1–hg4, iii |
| 11 | stroke-plane tilt | axillary |
| 12–13 | head pitch and yaw | neck motor neurons |
| 14 | abdomen pitch | abdominal motor neurons |

Measured authority, from perturbing each channel and integrating over a wingbeat:
roll from differential amplitude, pitch from collective stroke offset, yaw from
differential angle of attack, thrust from stroke-plane tilt — the same assignments
real flies use.

## The task

Eight hoops, 3–5 cm across, 16–26 cm apart, in a 3 m arena — to a 2.5 mm animal,
a drone-racing track. Reward is progress and gate clearance, minus aerodynamic
power in units of hovering power (*efficient*), minus control jerk and body
angular rate (*elegant*). Course difficulty is a single number the curriculum
raises as the fly starts clearing gates.

Hovering is an unstable equilibrium here, as it is for a real fly, so training
starts by cloning a hand-written reflex pilot (`flypv/control/reflex.py`) and then
improves on it with PPO. That pilot clears about 2.5 of 6 gates and finishes 16%
of courses; it exists to prove the course is flyable and to give PPO somewhere
to start, not to be good.

## Does the connectome actually help?

The interesting claim is not that a network can learn to fly. It's that *this*
wiring helps. So the controls ship in the box:

```
flypv train --arch connectome   # the real thing
flypv train --arch shuffled     # rewired, degree sequence preserved exactly
flypv train --arch erdos        # random graph, node and edge count matched
flypv train --arch mlp          # conventional network, comparable parameters
```

`shuffled` is the one that matters: same neurons, same synapse count, same
degree distribution, same excitatory/inhibitory ratio — only the pattern of who
connects to whom destroyed. If the real connectome doesn't beat it, the
connectome is doing nothing, and that is worth knowing.

## The monitor

`http://127.0.0.1:8777` while training. The connectome is drawn at real soma
coordinates, so you are looking at an actual fly nervous system: optic lobes
either side of the central brain, descending axons through the neck connective,
motor pools at the bottom of the nerve cord — lighting up as the fly flies.
Alongside it, what the eye sees through its hex lattice, the ON/OFF split, the
course and the fly's line through it, the wingbeat traces, and per-muscle firing
rates.

## Scales

| `--scale` | neurons | connections | use |
|---|---|---|---|
| `core` | 6,000 | 96 k | fast iteration on a laptop |
| `flight` | 28,361 | 535 k | default; everything within a few synapses of a flight sensor *and* a flight muscle |
| `full` | 165,122 | 6.2 M | the whole traced CNS |

## Data

MaleCNS v1.0, CC-BY, from `storage.googleapis.com/flyem-male-cns`. Acquired,
reconstructed and annotated by the Janelia FlyEM Project Team and the Cambridge
Drosophila Connectomics Group with the Connectomics group at Google Research.
`flypv fetch --print-urls` prints the download lines if you would rather run them
yourself.

## References

- Male CNS connectome — <https://male-cns.janelia.org/download/>
- FlyWire FAFB connectome — <https://codex.flywire.ai>
- MANC nerve cord circuits — <https://elifesciences.org/articles/96084>
- flybody MuJoCo model — <https://github.com/TuragaLab/flybody>, Nature (2025)
- Sane & Dickinson, quasi-steady flapping aerodynamics, J. Exp. Biol. (1999, 2002)
