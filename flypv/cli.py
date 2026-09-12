"""flypv command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _add_train_args(t: argparse.ArgumentParser) -> None:
    g = t.add_argument_group("experiment")
    g.add_argument("--config", help="YAML config file; CLI flags override it")
    g.add_argument("--run", dest="run_name", default=None, help="run name under --out-dir")
    g.add_argument("--out-dir", default=None, help="experiment root (default: runs)")
    g.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="resume PATH, or latest/best checkpoint for --run (bare --resume means latest)",
    )
    g.add_argument("--reset-optimizer", action="store_true", default=None,
                   help="resume weights/counters but start Adam state fresh")
    g.add_argument("--print-config", action="store_true",
                   help="print the resolved YAML configuration and exit")

    g = t.add_argument_group("model")
    g.add_argument("--arch", choices=["connectome", "shuffled", "erdos", "mlp"], default=None)
    g.add_argument("--scale", choices=["full", "flight", "core"], default=None)
    g.add_argument("--max-neurons", type=int, default=None)
    g.add_argument("--iters", dest="n_iters", type=int, default=None,
                   help="connectome propagation iterations per 5 ms control tick")
    g.add_argument("--tbptt", dest="tbptt_steps", type=int, default=None,
                   help="temporal credit-assignment window in control steps")
    g.add_argument("--state-carry", type=float, default=None,
                   help="global state retention between control ticks, in [0,1]")
    g.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=None,
                   help="persistent connectome state (default on)")
    g.add_argument("--stateless", action="store_true", default=None,
                   help="alias for --no-recurrent; explicit ablation")
    g.add_argument("--learn-synapses", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--warmstart", dest="warmstart_steps", type=int, default=None,
                   help="expert transitions per environment; 0 disables behaviour cloning")

    g = t.add_argument_group("PPO")
    g.add_argument("--steps", dest="total_steps", type=int, default=None)
    g.add_argument("--envs", dest="n_envs", type=int, default=None)
    g.add_argument("--rollout", type=int, default=None)
    g.add_argument("--ppo-epochs", dest="epochs", type=int, default=None)
    g.add_argument("--minibatches", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--gamma", type=float, default=None)
    g.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=None)
    g.add_argument("--clip", type=float, default=None)
    g.add_argument("--vf-coef", dest="vf_coef", type=float, default=None)
    g.add_argument("--ent-coef", dest="ent_coef", type=float, default=None)
    g.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=None)
    g.add_argument("--target-kl", dest="target_kl", type=float, default=None)

    g = t.add_argument_group("task")
    g.add_argument("--gates", type=int, default=None)
    g.add_argument("--difficulty", type=float, default=None)
    g.add_argument("--episode-seconds", type=float, default=None)
    g.add_argument("--vision", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--curriculum-target", type=float, default=None)
    g.add_argument("--curriculum-rate", type=float, default=None)

    g = t.add_argument_group("runtime")
    g.add_argument("--save-every", type=int, default=None,
                   help="checkpoint every N PPO updates; 0 = only final/interrupt")
    g.add_argument("--keep-checkpoints", type=int, default=None,
                   help="number of recent numbered checkpoints to retain")
    g.add_argument("--best-metric", choices=["ep_gates", "ep_return"], default=None)
    g.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--port", type=int, default=8777)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default=None)


def _train_overrides(a) -> dict:
    top_names = [
        "run_name", "out_dir", "reset_optimizer", "arch", "scale", "max_neurons",
        "n_iters", "tbptt_steps", "state_carry", "recurrent", "learn_synapses",
        "warmstart_steps", "total_steps", "n_envs", "rollout", "epochs", "minibatches",
        "lr", "gamma", "gae_lambda", "clip", "vf_coef", "ent_coef", "max_grad_norm",
        "target_kl", "curriculum", "curriculum_target", "curriculum_rate", "save_every",
        "keep_checkpoints", "best_metric", "monitor", "seed", "device",
    ]
    out = {name: getattr(a, name) for name in top_names if getattr(a, name) is not None}
    if a.stateless:
        if a.recurrent is True:
            raise ValueError("--stateless conflicts with --recurrent")
        out["recurrent"] = False
    env = {
        "n_gates": a.gates,
        "difficulty": a.difficulty,
        "episode_seconds": a.episode_seconds,
        "vision": a.vision,
    }
    env = {k: v for k, v in env.items() if v is not None}
    if env:
        out["env"] = env
    return out


def _resolve_train_config(a):
    from .train.checkpoint import CheckpointManager
    from .train.config import TrainConfig, load_yaml, merge_config

    yaml_data = load_yaml(a.config) if a.config else {}
    cfg = merge_config(TrainConfig(), yaml_data)

    locator = {}
    if a.run_name is not None:
        locator["run_name"] = a.run_name
    if a.out_dir is not None:
        locator["out_dir"] = a.out_dir
    cfg = merge_config(cfg, locator)

    resume_path = None
    if a.resume:
        resume_path = CheckpointManager.resolve(Path(cfg.out_dir) / cfg.run_name, a.resume)
        payload = CheckpointManager.load(resume_path, map_location="cpu")
        saved = payload.get("cfg")
        if not saved:
            raise ValueError(f"checkpoint has no embedded config: {resume_path}")
        cfg = TrainConfig.from_dict(saved)
        cfg = merge_config(cfg, yaml_data)

    cfg = merge_config(cfg, _train_overrides(a))
    if resume_path is not None:
        cfg.resume = str(resume_path)
    cfg.validate()
    return cfg


def _show_runs(out_dir: str, run: str | None) -> int:
    root = Path(out_dir)
    if run:
        d = root / run
        if not d.exists():
            print(f"run not found: {d}", file=sys.stderr)
            return 1
        print(d)
        config = d / "config.yaml"
        if config.exists():
            print(f"  config: {config}")
        ck = d / "checkpoints"
        if ck.exists():
            for name in ("latest", "best"):
                p = ck / f"{name}.json"
                if p.exists():
                    meta = json.loads(p.read_text())
                    metric = "" if meta.get("metric_value") is None else f" ({meta['metric']}={meta['metric_value']:.3g})"
                    print(f"  {name:6s}: {meta['file']}{metric}")
            print(f"  checkpoints: {len(list(ck.glob('step_*.pt')))} retained")
        return 0

    if not root.exists():
        print(f"no runs directory: {root}")
        return 0
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        print(f"no runs in {root}")
        return 0
    for d in dirs:
        latest = d / "checkpoints" / "latest.json"
        suffix = ""
        if latest.exists():
            meta = json.loads(latest.read_text())
            suffix = f"  step {int(meta.get('step', 0)):,}"
        print(f"{d.name}{suffix}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="flypv",
        description="Reinforcement learning on the Drosophila MaleCNS connectome.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the connectome")
    f.add_argument("--tier", default="core", choices=["core", "extra", "all"])
    f.add_argument("--force", action="store_true")
    f.add_argument("--print-urls", action="store_true")

    b = sub.add_parser("build", help="build and cache the filtered connectome")
    b.add_argument("--min-synapses", type=int, default=5)

    c = sub.add_parser("circuit", help="describe the extracted flight circuit")
    c.add_argument("--scale", default="flight", choices=["full", "flight", "core"])
    c.add_argument("--hops", type=int, default=2)
    c.add_argument("--max-neurons", type=int, default=None)

    t = sub.add_parser("train", help="train the fly")
    _add_train_args(t)

    r = sub.add_parser("runs", help="list runs or inspect checkpoint status")
    r.add_argument("--out-dir", default="runs")
    r.add_argument("--run")

    m = sub.add_parser("monitor", help="serve the dashboard on its own")
    m.add_argument("--port", type=int, default=8777)

    a = ap.parse_args(argv)

    try:
        if a.cmd == "fetch":
            from .connectome.sources import LICENSE_NOTE, curl_commands, fetch
            if a.print_urls:
                print("\n".join(curl_commands(a.tier)))
                return 0
            fetch(tier=a.tier, force=a.force)
            print("\n" + LICENSE_NOTE)
            return 0

        if a.cmd == "build":
            from .connectome import load_connectome
            load_connectome(min_synapses=a.min_synapses, cache=True)
            return 0

        if a.cmd == "circuit":
            from .connectome import build_flight_circuit
            build_flight_circuit(scale=a.scale, hops=a.hops, max_neurons=a.max_neurons)
            return 0

        if a.cmd == "runs":
            return _show_runs(a.out_dir, a.run)

        if a.cmd == "monitor":
            from .monitor import serve
            print(f"[flypv] monitor -> http://127.0.0.1:{a.port}")
            serve(port=a.port)
            return 0

        if a.cmd == "train":
            cfg = _resolve_train_config(a)
            if a.print_config:
                print(yaml.safe_dump(cfg.to_dict(), sort_keys=False), end="")
                return 0

            from .monitor import serve_background
            from .train import train
            if cfg.monitor:
                serve_background(port=a.port)
            train(cfg)
            return 0
    except (ValueError, FileNotFoundError, TypeError) as exc:
        ap.error(str(exc))

    return 1


if __name__ == "__main__":
    sys.exit(main())
