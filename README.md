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

The connectome is used as an architectural prior, not as a claim about how a fly computes in real time.

`scale: flight` extracts a compact two-hop sensor/output corridor from MaleCNS. Connectivity is fixed by default. Each control decision independently:

1. encodes the current visual + task/proprio observation onto sensory populations;
2. runs `n_iters` sparse message-passing rounds through the measured graph;
3. decodes activity at motor populations into the 15 flight controls.

There is no persistent neural state or TBPTT. PPO is ordinary feed-forward PPO, which is substantially simpler and cheaper to train.

## Training

```bash
flypv train \
  --run connectome-s0 \
  --steps 5000000 \
  --envs 16 \
  --rollout 128 \
  --lr 1e-4 \
  --iters 4 \
  --seed 0
```

For unattended training:

```bash
flypv train --config configs/overnight-m5.yaml
```

YAML keys match `TrainConfig`; CLI flags override YAML. Use `--print-config` to inspect the resolved configuration.

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

Architecture baselines:

```bash
flypv train --arch connectome --run real-s0
flypv train --arch shuffled  --run shuffled-s0
flypv train --arch erdos     --run erdos-s0
flypv train --arch mlp       --run mlp-s0
```

Useful options:

```bash
flypv train --help
flypv train --iters 6
flypv train --no-vision
flypv train --scale core
flypv train --scale full
flypv train --learn-synapses
```

`--learn-synapses` makes per-edge efficacy trainable; the default keeps the measured graph fixed.

## Environment

The plant is a vectorized 6-DOF rigid body driven by a flapping-wing pattern generator and quasi-steady blade-element aerodynamics. The policy outputs 15 bounded controls covering wingbeat frequency, left/right stroke amplitude and offset, deviation, angle of attack, stroke-plane tilt, head pitch/yaw and abdomen pitch.

Training starts with behavior cloning from a hand-written reflex pilot unless `--warmstart 0` is used, then switches to PPO. The course curriculum can progressively shrink gates and increase turn/climb difficulty as performance saturates.

## Monitor

The monitor shows third-person flight, connectome activations, compound-eye luminance, wing/motor traces and learning curves. It is read-only telemetry and does not block training.

## Data

MaleCNS v1.0 is downloaded from the Janelia FlyEM public release.

- MaleCNS: <https://male-cns.janelia.org/download/>
- FlyWire/Codex: <https://codex.flywire.ai>
