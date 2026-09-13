# flypv

Train a sparse controller built from the **Drosophila MaleCNS v1.0** wiring graph to fly a simulated 3D hoop course.

```bash
flypv fetch
flypv build
flypv circuit
flypv train --config configs/overnight-m5.yaml
```

The live monitor opens at `http://127.0.0.1:8777` by default.

## Policy

The connectome is used as a neural-network architecture constraint, not as a physiological model of fly computation.

`scale: flight` extracts the compact sensor-to-motor subgraph used for training. The important constraint is the **adjacency mask**:

- a measured MaleCNS connection means that weight is allowed;
- an absent connection remains exactly zero;
- allowed edge weights and signs are learned freely from the flight task;
- visual input is kept retinotopic;
- actions are decoded only from anatomically identified efferent/motor neurons.

Each control decision independently:

1. maps compound-eye ON/OFF values onto their retinotopic neurons and the low-dimensional task/proprio vector onto the remaining sensory population;
2. runs `n_iters` sparse residual layers using the same MaleCNS adjacency mask but independently learned edge weights per layer;
3. learns a dense readout from the actual motor/efferent population to the 15 simulator controls.

There is no persistent neural state or TBPTT. Biological transmitter signs, synapse counts and hand-written motor-group semantics are not hard constraints. The default six-layer flight policy is a few-million-parameter sparse neural network.

## Training

```bash
flypv train \
  --run connectome-s0 \
  --steps 5000000 \
  --envs 16 \
  --rollout 128 \
  --lr 1e-4 \
  --iters 6 \
  --seed 0
```

For unattended training:

```bash
flypv train --config configs/overnight-m5.yaml
```

YAML keys match `TrainConfig`; CLI flags override YAML. Use `--print-config` to inspect the resolved configuration.

Training begins with behavior cloning from the geometric reflex pilot. The warm-start now evaluates both the expert and the deterministic clone in the actual environment, so a bad bootstrap is visible before PPO silently spends hours optimizing a controller that cannot fly.

## Checkpoints

Runs are stored under `runs/<run>/`. Checkpoints include policy weights, Adam state, PPO counters, curriculum state, recent episode statistics, RNG state and the resolved config. Training checkpoints periodically, on clean completion, and on `Ctrl-C`.

```bash
flypv runs
flypv runs --run connectome-s0
flypv train --run connectome-s0 --resume
flypv train --run connectome-s0 --resume best
```

To restart Adam while keeping the checkpointed policy/counters:

```bash
flypv train --run connectome-s0 --resume --reset-optimizer --lr 5e-5
```

## Controls

The main topology comparison is:

```bash
flypv train --arch connectome --run real-s0
flypv train --arch shuffled  --run shuffled-s0
flypv train --arch erdos     --run erdos-s0
flypv train --arch mlp       --run mlp-s0
```

`shuffled` preserves directed in/out degree while rewiring the mask; `erdos` keeps node and edge counts but randomizes adjacency. The dense MLP baseline is widened to roughly match the parameter budget of the trainable sparse policy.

Useful options:

```bash
flypv train --help
flypv train --iters 8
flypv train --no-vision
flypv train --scale core
flypv train --scale full
flypv train --no-learn-synapses   # freezes random edge values; debugging ablation
```

## Environment

The plant is a vectorized 6-DOF rigid body driven by a flapping-wing pattern generator and quasi-steady blade-element aerodynamics. The policy outputs 15 bounded controls covering wingbeat frequency, left/right stroke amplitude and offset, deviation, angle of attack, stroke-plane tilt, head pitch/yaw and abdomen pitch.

The course curriculum progressively shrinks gates and increases turn/climb difficulty only after the current level is reliably solved and speed has plateaued.

## Monitor

The monitor shows third-person flight, connectome activations, compound-eye luminance, wing/motor traces and learning curves. It is read-only telemetry and does not block training.

## Data

MaleCNS v1.0 is downloaded from the Janelia FlyEM public release.

- MaleCNS: <https://male-cns.janelia.org/download/>
- FlyWire/Codex: <https://codex.flywire.ai>
