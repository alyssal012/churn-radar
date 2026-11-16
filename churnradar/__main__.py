"""
python -m churnradar demo                    # simulate a company and run everything
python -m churnradar run DATA_FOLDER config.json --budget 10000
"""
from __future__ import annotations

import argparse
import json
import pathlib

from . import simulate
from .features import Config
from .pipeline import run

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(prog="churnradar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--budget", type=float, default=10_000)
    r = sub.add_parser("run")
    r.add_argument("folder", type=pathlib.Path)
    r.add_argument("config", type=pathlib.Path)
    r.add_argument("--budget", type=float, required=True)
    r.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports"))
    args = ap.parse_args()

    if args.cmd == "demo":
        folder = ROOT / "examples" / "demo_data"
        if not (folder / "customers.csv").exists():
            print("Simulating a subscription company...")
            simulate.write(folder)
        cfg = Config(train_as_of=simulate.TRAIN_AS_OF, score_as_of=simulate.SCORE_AS_OF)
        stats = run(cfg, folder, args.budget, ROOT / "reports")
    else:
        stats = run(Config.load(args.config), args.folder, args.budget, args.out)
    print(stats["markdown"])


if __name__ == "__main__":
    main()
