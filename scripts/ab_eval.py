"""Run the frozen-vs-unfrozen A/B offline, at a sample size that can read it.

Run 9 (frozen) and run 10 (last 2 text layers unfrozen) are indistinguishable
on their in-run eval/mean -- 11/0/0/11/0/0/0/0 against 0/0/0/0/11/11/0/0 is
two draws from the same noise at 3 rollouts per task. This re-rolls both at
n=50 on the two tasks that discriminate, at matching epochs so the comparison
is at equal optimizer steps.

Checkpoints come from Beekeeper's per-run artifact API rather than the
workspace, which points at whichever run is current. Evaluation is farmed out
to scripts/evaluate.py subprocesses so the tested CLI path is the one that
runs, and so several checkpoints go at once -- a rollout leaves the GPU idle
most of its wall clock waiting on MuJoCo.

Driven by Beekeeper as train_file; restore train_file to scripts/train.py
after.
"""
import json
import os
import subprocess
import sys
import urllib.request

# Same pin as scripts/train.py: Beekeeper exports the nvidia-smi index, CUDA
# reads it in FASTEST_FIRST order, and the two disagree on lab. Set before the
# worker subprocesses inherit this environment, or they land on the 3060.
if os.environ.get("BEEKEEPER_RUN_DIR"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

API = os.environ.get("BEEKEEPER_API", "http://localhost:5000/api/v1")
PROJECT = "franka-kitchen-vla1"

EPOCHS = [10000, 12500, 15000, 17500, 20000]
ARMS = {9: "frozen", 10: "unfrozen2"}
TASKS = ["hinge cabinet", "top burner"]
ROLLOUTS = int(os.environ.get("AB_ROLLOUTS", "50"))
# MuJoCo's EGL contexts are not reliable many-at-once on one card: at 5 workers
# on lab, two died -- one raising from eglMakeCurrent during teardown, one on
# SIGSEGV before its first rollout. 3 is what survived. A desktop with a display
# uses GLFW instead and does not hit this.
WORKERS = int(os.environ.get("AB_WORKERS", "3"))

OUT = os.path.join(os.environ.get("BEEKEEPER_RUN_DIR", "."), "ab")


def fetch(run_id, epoch, dest):
    """Pull one checkpoint out of a specific run's persistent storage."""
    url = f"{API}/projects/{PROJECT}/runs/{run_id}/files/checkpoints/vla_network.e{epoch}"
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return os.path.getsize(dest)


def main():
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for run_id, arm in ARMS.items():
        for epoch in EPOCHS:
            path = os.path.join(OUT, f"{arm}.e{epoch}")
            size = fetch(run_id, epoch, path)
            print(f"fetched run {run_id} e{epoch}: {size / 1e6:.1f} MB", flush=True)
            jobs.append((f"{arm}.e{epoch}", path))

    # Bounded fan-out: each worker holds its own copy of the VLM.
    results, running = {}, []
    path_of = dict(jobs)
    queue = [(label, path, 0) for label, path in jobs]
    while queue or running:
        while queue and len(running) < WORKERS:
            label, path, attempt = queue.pop(0)
            out = os.path.join(OUT, f"{label}.json")
            cmd = [sys.executable, "-u", os.path.join(HERE, "evaluate.py"), path,
                   "--rollouts", str(ROLLOUTS), "--tasks", *TASKS, "--out", out]
            log = open(os.path.join(OUT, f"{label}.log"), "w")
            running.append((label, out, subprocess.Popen(cmd, stdout=log, stderr=log),
                            log, attempt))
            print(f"started {label}" + (" (retry)" if attempt else ""), flush=True)

        label, out, proc, log, attempt = running.pop(0)
        code = proc.wait()
        log.close()
        if code != 0:
            # EGL failures are intermittent, so one requeue is worth more than
            # a hole in the table. A real bug fails twice and is reported.
            if attempt == 0:
                print(f"retrying {label} (exit {code})", flush=True)
                queue.append((label, path_of[label], 1))
                continue
            print(f"FAILED {label} twice (exit {code}) -- see {label}.log", flush=True)
            continue
        with open(out) as f:
            results[label] = json.load(f)[0]
        m = results[label]["mean"]
        print(f"done {label}: {m['rate']:.1%} [{m['ci'][0]:.1%}, {m['ci'][1]:.1%}]",
              flush=True)

    print("\n" + "=" * 78)
    print(f"{'epoch':>8}" + "".join(f"{a:>22}" for a in ARMS.values()))
    for epoch in EPOCHS:
        row = f"{epoch:>8}"
        for arm in ARMS.values():
            r = results.get(f"{arm}.e{epoch}")
            row += (f"{r['mean']['rate']:>10.1%} "
                    f"[{r['mean']['ci'][0]:.0%},{r['mean']['ci'][1]:.0%}]".rjust(12)
                    if r else f"{'--':>22}")
        print(row)

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
