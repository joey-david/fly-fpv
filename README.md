# flypv

Train a controller built on the **Drosophila MaleCNS v1.0** connectome to fly a simulated 3D hoop course.

```bash
flypv fetch
flypv build
flypv circuit
flypv train --run rec-s0
```

The default run uses the flight-scale connectome, persistent neural state, PPO with truncated BPTT, and opens the live monitor at `http://127.0.0.1:8777`.

## Model

The default `flight` circuit contains 28,361 neurons and ~535k measured connections. Connectivity is fixed; per-neuron dynamics and the sensory/motor interfaces are learned. Vision is mapped onto the connectome-derived retinotopic lattice, and motor-neuron populations drive 15 wing/head/abdomen controls.

Neural state persists across control steps. The simulator runs at 200 Hz and TBPTT defaults to 16 steps (80 ms). Use `--stateless` for the old reset-every-step ablation.

The environment is a vectorized 6-DOF fly model with a flapping-wing pattern generator and quasi-steady blade-element aerodynamics. Training starts from behavior cloning of a hand-written reflex pilot, then switches to PPO.

## Training

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

See all controls with:

```bash
flypv train --help
```

For reproducible experiments, use YAML:

```bash
flypv train --config experiment.yaml
flypv train --config experiment.yaml --lr 5e-5
flypv train --config experiment.yaml --print-config
```

CLI arguments override YAML. The resolved config is written to `runs/<run>/config.yaml`.

## Checkpoints and resume

Full checkpoints contain policy and optimizer state, PPO counters, curriculum state, recent metrics, RNG state, and config. They are written periodically, on `Ctrl-C`, and at clean completion.

```bash
flypv runs
flypv runs --run connectome-rec-s0

flypv train --run connectome-rec-s0 --resume
flypv train --run connectome-rec-s0 --resume best
```

To resume weights while restarting Adam:

```bash
flypv train --run connectome-rec-s0 --resume --reset-optimizer --lr 5e-5
```

A resumed run starts fresh simulator episodes; policy, optimizer, counters, curriculum, and RNG state continue.

## Experimental controls

```bash
flypv train --arch connectome --run real-s0
flypv train --arch shuffled  --run shuffled-s0
flypv train --arch erdos     --run erdos-s0
flypv train --arch mlp       --run mlp-s0
```

Other useful ablations:

```bash
flypv train --stateless
flypv train --no-vision
flypv train --scale core
flypv train --scale full
flypv train --learn-synapses
```

`--learn-synapses` makes edge efficacies trainable, so leave it off for the fixed-wiring experiment.

## Monitor

The lightweight browser monitor shows third-person flight, live connectome activity, compound-eye channels, motor/wingbeat activity, and learning curves. It uses FastAPI/WebSockets and does not block training.

## Data

MaleCNS v1.0 is loaded from the Janelia FlyEM public release.

- MaleCNS: <https://male-cns.janelia.org/download/>
- FlyWire/Codex: <https://codex.flywire.ai>
- flybody: <https://github.com/TuragaLab/flybody>
