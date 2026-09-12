"""flypv command line."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="flypv",
        description="Reinforcement learning on the Drosophila maleCNS connectome.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the connectome")
    f.add_argument("--tier", default="core", choices=["core", "extra", "all"])
    f.add_argument("--force", action="store_true")
    f.add_argument("--print-urls", action="store_true",
                   help="just print the curl commands and exit")

    b = sub.add_parser("build", help="build and cache the filtered connectome")
    b.add_argument("--min-synapses", type=int, default=5)

    c = sub.add_parser("circuit", help="describe the extracted flight circuit")
    c.add_argument("--scale", default="flight", choices=["full", "flight", "core"])
    c.add_argument("--hops", type=int, default=2)
    c.add_argument("--max-neurons", type=int, default=None)

    t = sub.add_parser("train", help="train the fly to fly")
    t.add_argument("--arch", default="connectome",
                   choices=["connectome", "shuffled", "erdos", "mlp"])
    t.add_argument("--scale", default="flight", choices=["full", "flight", "core"])
    t.add_argument("--max-neurons", type=int, default=None)
    t.add_argument("--steps", type=int, default=5_000_000)
    t.add_argument("--envs", type=int, default=32)
    t.add_argument("--rollout", type=int, default=128)
    t.add_argument("--gates", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--iters", type=int, default=4, help="synaptic hops per control step")
    t.add_argument("--learn-synapses", action="store_true")
    t.add_argument("--no-vision", action="store_true")
    t.add_argument("--no-monitor", action="store_true")
    t.add_argument("--port", type=int, default=8777)
    t.add_argument("--run", default="flypv")
    t.add_argument("--warmstart", type=int, default=400,
                   help="expert transitions per env for behaviour cloning (0 disables)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")

    m = sub.add_parser("monitor", help="serve the dashboard on its own")
    m.add_argument("--port", type=int, default=8777)

    a = ap.parse_args(argv)

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

    if a.cmd == "monitor":
        from .monitor import serve
        print(f"[flypv] monitor -> http://127.0.0.1:{a.port}")
        serve(port=a.port)
        return 0

    if a.cmd == "train":
        from .env import EnvConfig
        from .monitor import serve_background
        from .train import TrainConfig, train

        cfg = TrainConfig(
            arch=a.arch, scale=a.scale, max_neurons=a.max_neurons,
            total_steps=a.steps, n_envs=a.envs, rollout=a.rollout,
            lr=a.lr, n_iters=a.iters, learn_synapses=a.learn_synapses,
            monitor=not a.no_monitor, run_name=a.run, seed=a.seed, device=a.device,
            warmstart_steps=a.warmstart,
            env=EnvConfig(n_envs=a.envs, n_gates=a.gates, vision=not a.no_vision,
                          seed=a.seed),
        )
        if cfg.monitor:
            serve_background(port=a.port)
        train(cfg)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
