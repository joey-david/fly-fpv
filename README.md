# flypv

Train a connectome-constrained controller built from the **Drosophila MaleCNS v1.0** wiring diagram to fly a simulated 3D hoop course.

```bash
flypv fetch
flypv build
flypv circuit
flypv train --run rec-s0
```

The default training run opens a live monitor at `http://127.0.0.1:8777`.

## What is fixed vs learned

The default `flight` circuit contains 28,361 traced neurons and about 535k measured connections. Connectivity comes from MaleCNS and is not a trainable parameter. Synaptic sign is taken from the presynaptic transmitter prediction: acetylcholine is excitatory; GABA, glutamate and histamine are inhibitory in the fast model; unresolved/modulatory cases are handled separately.

The policy is a leaky rate network over that sparse graph. Per-neuron gain, bias and leak are learned. Visual input is placed on the connectome-derived retinotopic lattice; non-visual proprioception uses a learned sensory encoder. Motor-neuron activity is pooled by anatomical motor groups and decoded to 15 wing/head/abdomen controls.

Neural state is **persistent across control steps by default**. The simulator runs at 200 Hz, and PPO uses truncated backpropagation through time (TBPTT) over 16 control steps = 80 ms by default. `--stateless` is the old reset-every-tick ablation.

## Training

A normal run:

```bash
flypv train \
  --run connectome-rec-s0 \
  --steps 5000000 \
  --envs 32 \
  --rollout 128 \
  --lr 1e-4 \
  --tbptt 16 \
  --seed 0
```

Useful controls are exposed directly:

```bash
flypv train --help
```

That includes PPO epochs/minibatches, gamma, GAE lambda, clip range, entropy/value coefficients, gradient norm, KL target, recurrent state settings, warm-start length, curriculum controls, vision, and checkpoint cadence.

For larger experiments, use YAML instead of a long command:

```bash
flypv train --config experiment.yaml
flypv train --config experiment.yaml --lr 5e-5   # CLI overrides YAML
flypv train --config experiment.yaml --print-config
```

The YAML keys match `TrainConfig`; environment settings live under `env:`. Each run writes its resolved configuration to `runs/<run>/config.yaml`.

## Checkpoints and resume

Checkpoints are full training checkpoints, not just model weights. They contain the policy, Adam state, PPO step/update counters, curriculum difficulty, recent episode statistics, elapsed time, RNG state, metrics and the resolved configuration.

By default a checkpoint is written every 20 PPO updates, the newest 5 are retained, and `latest.json` / `best.json` point to the relevant numbered checkpoint:

```text
runs/connectome-rec-s0/
  config.yaml
  checkpoints/
    step_000081920.pt
    step_000163840.pt
    ...
    latest.json
    best.json
```

Training also checkpoints on `Ctrl-C` and on clean completion.

Resume the same experiment:

```bash
flypv train --run connectome-rec-s0 --resume
flypv train --run connectome-rec-s0 --resume best
```

Or resume an explicit file:

```bash
flypv train --resume runs/connectome-rec-s0/checkpoints/step_000163840.pt
```

Optimizer state is restored by default. To keep the weights/counters but deliberately restart Adam:

```bash
flypv train --run connectome-rec-s0 --resume --reset-optimizer --lr 5e-5
```

A resumed process starts fresh physical episodes; policy, optimizer, counters and curriculum continue. This avoids serializing a giant simulator object while retaining the training state that matters across runs.

Inspect runs without opening files manually:

```bash
flypv runs
flypv runs --run connectome-rec-s0
```

## Experimental controls

The scientific comparison is between the real graph and matched controls:

```bash
flypv train --arch connectome --run real-s0
flypv train --arch shuffled  --run shuffled-s0
flypv train --arch erdos     --run erdos-s0
flypv train --arch mlp       --run mlp-s0
```

The degree-preserving shuffle uses directed double-edge swaps: every neuron's in/out degree is preserved exactly, outgoing synaptic weights/signs stay attached to the same presynaptic neuron, and duplicate edges are rejected. The MLP baseline is sized close to the trainable-parameter budget of the flight-scale connectome policy.

Other useful ablations:

```bash
flypv train --stateless
flypv train --no-vision
flypv train --scale core
flypv train --scale full
flypv train --learn-synapses
```

`--learn-synapses` changes the scientific question because edge efficacies become trainable; leave it off for the main fixed-wiring experiment.

## Environment

The plant is a vectorized 6-DOF rigid body driven by a flapping-wing pattern generator and quasi-steady blade-element aerodynamics. The policy outputs 15 bounded controls covering wingbeat frequency, left/right stroke amplitude and offset, deviation, angle of attack, stroke-plane tilt, head pitch/yaw and abdomen pitch.

The default course has eight gates. Current `GateSpec` uses a 10 cm clear radius, 2 cm ring thickness and 46–66 cm spacing before difficulty scaling. Reward combines progress and gate clearance with penalties for aerodynamic power, control jerk and body angular rate.

Training begins with behavior cloning from a hand-written reflex pilot unless `--warmstart 0` is supplied, then switches to PPO.

Time-limit truncations bootstrap from the final observation rather than being treated as terminal deaths.

## Monitor

The monitor is deliberately separate from experiment management. It is a lightweight FastAPI/WebSocket backend plus a vanilla Canvas frontend showing:

- third-person simulated flight with instantaneous wing geometry;
- connectome activity at anatomical soma coordinates;
- compound-eye luminance / ON / OFF channels;
- motor-pool activity and wingbeat traces;
- learning curves.

The browser consumes latest-value telemetry and never blocks training.

## Data and references

MaleCNS v1.0 is downloaded from the Janelia FlyEM public release. `flypv fetch --print-urls` prints the exact source commands.

- MaleCNS: <https://male-cns.janelia.org/download/>
- FlyWire/Codex: <https://codex.flywire.ai>
- flybody: <https://github.com/TuragaLab/flybody>
- Sane & Dickinson, quasi-steady flapping aerodynamics (1999, 2002)
