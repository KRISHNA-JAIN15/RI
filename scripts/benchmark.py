"""Run the benchmark / ablation matrix in configs/benchmark.yaml with parallel workers.

    python scripts/benchmark.py                       # everything (resumable; finished jobs are skipped)
    python scripts/benchmark.py --scenes static --variants paper_ddpg,td3 --seeds 0 --episodes 500
    python scripts/benchmark.py --dry-run             # list jobs

Layout: runs/bench/<scene>/<variant>/seed<k>/{config.yaml, train_log.csv, train_summary.json, ddpg_final.pt, eval/}
Then:   python scripts/report_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SAVE_PATHS = "--save-paths" in (ROOT / "scripts" / "evaluate.py").read_text()  # added by the C5 plugin work


def load(path):
    return yaml.safe_load(open(path))


def set_args(sets: dict) -> list[str]:
    out = []
    for k, v in (sets or {}).items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        out += ["--set", f"{k}={v}"]
    return out


class Job:
    def __init__(self, name, cmd, out_dir: Path, done_file: Path, deps=(), log=None):
        self.name, self.cmd, self.out_dir, self.done_file, self.deps = name, cmd, out_dir, done_file, list(deps)
        self.log = log or (out_dir / "job.log")
        self.status = "done" if done_file.exists() else "pending"
        self.t = 0.0

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(self.log, "w") as f:
            f.write(" ".join(map(str, self.cmd)) + "\n\n")
            f.flush()
            rc = subprocess.call([str(c) for c in self.cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.t = time.time() - t0
        self.status = "done" if (rc == 0 and self.done_file.exists()) else "failed"
        return self


def build_jobs(bench: dict, scenes, variants, seeds, episodes, eval_episodes, out_root: Path) -> list[Job]:
    jobs: dict[str, Job] = {}
    for sc in scenes:
        sc_cfg = bench["scenes"][sc]["config"]
        for var in variants:
            v = bench["variants"][var]
            var_seeds = [sd for sd in seeds if sd in v["seeds"]] if v.get("seeds") else seeds
            for seed in var_seeds:
                d = out_root / sc / var / f"seed{seed}"
                train_name = f"train:{sc}/{var}/s{seed}"
                eval_name = f"eval:{sc}/{var}/s{seed}"
                if v.get("eval_only"):
                    src = out_root / sc / v["eval_only"] / f"seed{seed}"
                    # config = source config + this variant's overrides, written at run time by a tiny helper
                    cfg_path = d / "config.yaml"
                    cmd = [PY, "scripts/benchmark.py", "--_eval-only", str(src), str(d), json.dumps(v.get("set") or {}),
                           str(eval_episodes), str(bench["eval_seed"])]
                    jobs[eval_name] = Job(eval_name, cmd, d, d / "eval" / "table1.json", deps=[f"train:{sc}/{v['eval_only']}/s{seed}"])
                    continue
                script = v.get("script", "scripts/train.py")
                sets = dict(v.get("set") or {})
                sets.update((v.get("scene_set") or {}).get(sc) or {})  # per-scene overrides
                cmd = [PY, script, "--config", sc_cfg, "--out", str(d), "--seed", str(seed), "--episodes", str(episodes),
                       "--threads", "1", "--quiet", "--set", "train.viewer_episodes=0"] + set_args(sets)
                jobs[train_name] = Job(train_name, cmd, d, d / "train_summary.json")
                ecmd = [PY, "scripts/evaluate.py", "--checkpoint", str(d / "ddpg_final.pt"), "--episodes", str(eval_episodes),
                        "--seed", str(bench["eval_seed"]), "--n-figures", "0", "--out", str(d / "eval")]
                if SAVE_PATHS:
                    ecmd.append("--save-paths")
                if not v.get("rrt"):
                    ecmd.append("--no-rrt")
                jobs[eval_name] = Job(eval_name, ecmd, d, d / "eval" / "table1.json", deps=[train_name])
    # resolve deps to objects
    for j in jobs.values():
        j.deps = [jobs[n] for n in j.deps if n in jobs]
    return list(jobs.values())


def eval_only(src: str, dst: str, sets_json: str, eval_episodes: str, eval_seed: str) -> int:
    """Helper mode: evaluate an existing checkpoint under a modified method config."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = load(src / "config.yaml")
    for k, v in json.loads(sets_json).items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(str(v)) if isinstance(v, str) else v
    yaml.safe_dump(cfg, open(dst / "config.yaml", "w"), sort_keys=False)
    shutil.copy(src / "ddpg_final.pt", dst / "ddpg_final.pt")
    shutil.copy(src / "train_summary.json", dst / "train_summary.json")
    shutil.copy(src / "train_log.csv", dst / "train_log.csv")
    cmd = [PY, "scripts/evaluate.py", "--config", str(dst / "config.yaml"), "--checkpoint", str(dst / "ddpg_final.pt"),
           "--episodes", eval_episodes, "--seed", eval_seed, "--n-figures", "0", "--out", str(dst / "eval"), "--no-rrt"]
    if SAVE_PATHS:
        cmd.append("--save-paths")
    return subprocess.call(cmd, cwd=ROOT)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_eval-only":
        sys.exit(eval_only(*sys.argv[2:]))
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="configs/benchmark.yaml")
    p.add_argument("--scenes", default=None, help="comma list (default: all)")
    p.add_argument("--variants", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--train-only", action="store_true", help="skip evaluation jobs")
    args = p.parse_args()

    bench = load(ROOT / args.matrix)
    scenes = args.scenes.split(",") if args.scenes else list(bench["scenes"])
    variants = args.variants.split(",") if args.variants else list(bench["variants"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(bench["seeds"])
    episodes = args.episodes or int(bench["train_episodes"])
    eval_episodes = args.eval_episodes or int(bench["eval_episodes"])
    workers = args.workers or int(bench.get("workers", 4))
    out_root = ROOT / (args.out or bench["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"scenes": scenes, "variants": variants, "seeds": seeds, "episodes": episodes, "eval_episodes": eval_episodes},
                   open(out_root / "matrix_used.yaml", "w"))

    jobs = build_jobs(bench, scenes, variants, seeds, episodes, eval_episodes, out_root)
    if args.train_only:
        jobs = [j for j in jobs if j.name.startswith("train:")]
    pending = [j for j in jobs if j.status != "done"]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run, {workers} workers")
    if args.dry_run:
        for j in jobs:
            print(f"  [{j.status}] {j.name}: {' '.join(map(str, j.cmd))[:160]}")
        return

    t_start = time.time()
    running = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while True:
            # launch ready jobs
            for j in jobs:
                if j.status == "pending" and len(running) < workers and all(d.status == "done" for d in j.deps):
                    j.status = "running"
                    running[ex.submit(j.run)] = j
            if not running:
                break
            done_now = [f for f in list(running) if f.done()]
            if not done_now:
                time.sleep(2.0)
                continue
            for f in done_now:
                j = running.pop(f)
                n_done = sum(1 for x in jobs if x.status == "done")
                print(f"[{time.time() - t_start:7.0f}s] {j.status:6s} {j.name} ({j.t:.0f}s)  {n_done}/{len(jobs)} done", flush=True)
            # dependents of failed jobs can never run
            for j in jobs:
                if j.status == "pending" and any(d.status == "failed" for d in j.deps):
                    j.status = "failed"
            if all(j.status in ("done", "failed") for j in jobs):
                break
    summary = {j.name: {"status": j.status, "seconds": j.t} for j in jobs}
    json.dump(summary, open(out_root / "jobs.json", "w"), indent=2)
    failed = [n for n, s in summary.items() if s["status"] == "failed"]
    print(f"finished in {(time.time() - t_start) / 60:.1f} min; failed: {len(failed)}")
    for n in failed:
        print("  FAILED", n)


if __name__ == "__main__":
    main()
