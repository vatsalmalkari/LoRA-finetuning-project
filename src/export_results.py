"""Copy finished results from Drive into the repo, with report-ready CSV tables, and push.

Used by the last notebook section. Adapters (adapter_model.safetensors) are never copied:
they are large and only needed for the subspace analysis, whose numbers are exported here.

    files = export(RESULTS, DATA_DIR, FIG_DIR, f"{REPO_DIR}/results/colab_L4", adapters=adapters, pairs=PAIRS)
    push(REPO_DIR, "results/colab_L4", "Add Colab results", token=secret("GITHUB_TOKEN"))
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

import analysis

README = """# Exported experiment results

Written by `src/export_results.py` (notebook section 9). Re-running the export overwrites this folder.

| Path | What it is |
|---|---|
| `runs/<run>.json` | one finished run: config, accuracy, params, memory, timing, latency (`lora_exp.run`) |
| `runs/<run>.predictions.json` | every test (or val) item: model response, strict/lenient correctness |
| `tables/summary_test.csv` | `lora_exp.summarize`: one row per configuration, mean and sd over seeds |
| `tables/summary_val.csv` | same for the learning-rate grid on the validation split |
| `tables/accuracy.csv` | accuracy by question type, MC split into rules/history, with 95% bootstrap CI |
| `tables/comparisons.csv` | paired comparisons: difference, 95% item-bootstrap CI, sign-flip permutation p |
| `tables/trend.json` | slope of accuracy on log10(trainable params), CI and p |
| `tables/stats_closeness.csv` | stats questions: share within 1/5/10%, median relative error |
| `tables/mc_baselines.csv` | chance / always-A / longest-answer baselines |
| `tables/subspace_<pair>_<side>.csv` | subspace similarity summaries (paper Section 7.2) |
| `figures/*.png` | figures saved by the notebook |

Accuracy is the *strict* metric unless a column says lenient (strict equals lenient for every run so far).
"""


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """summarize() has (metric, mean/sd) columns; flatten to metric_mean / metric_std for CSV."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [f"{a}_{b}" if b else a for a, b in df.columns]
    return df


def export(results_dir, data_dir, figures_dir, out_dir, adapters=None, pairs=None, log=print) -> list[Path]:
    """Copy run files and figures into out_dir and write the report tables. Returns written paths."""
    from lora_exp import summarize

    results_dir, out_dir = Path(results_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)                       # a clean copy: no stale runs left behind
    runs, tables, figs = out_dir / "runs", out_dir / "tables", out_dir / "figures"
    for d in (runs, tables, figs):
        d.mkdir(parents=True)
    written = []

    for p in sorted(results_dir.glob("*.json")):     # top level only: adapters/ is skipped
        written.append(Path(shutil.copy2(p, runs / p.name)))
    if figures_dir and Path(figures_dir).exists():
        for p in sorted(Path(figures_dir).glob("*.png")):
            written.append(Path(shutil.copy2(p, figs / p.name)))

    def save(df, name, index=True):
        path = tables / name
        df.to_csv(path, index=index)
        written.append(path)

    for split in ("test", "val"):
        s = summarize(results_dir, split)
        if len(s):
            save(_flat(s).round(5), f"summary_{split}.csv")

    A = analysis.Results(results_dir, data_dir)
    save(A.accuracy_table().round(5), "accuracy.csv")
    save(A.comparisons().round(5), "comparisons.csv", index=False)
    save(A.stats_closeness().round(5), "stats_closeness.csv")
    save(A.mc_baselines().round(5), "mc_baselines.csv")
    trend = A.trend()
    (tables / "trend.json").write_text(json.dumps(trend, indent=2))
    written.append(tables / "trend.json")

    for key, (a, b, *_labels) in (pairs or {}).items():
        for side in analysis.SIDES:
            try:
                sim = analysis.subspace_similarity(Path(adapters) / a, Path(adapters) / b, side=side)
            except (FileNotFoundError, ValueError) as e:
                log(f"skipped subspace {key} ({side}): {e}")
                continue
            save(sim["summary"].round(5), f"subspace_{key}_{side}.csv", index=False)

    (out_dir / "README.md").write_text(README, encoding="utf-8")
    written.append(out_dir / "README.md")
    log(f"exported {len(written)} files to {out_dir}")
    return written


def _git(repo, *args, secret: str | None = None, check=True):
    """Run git; hide `secret` in anything printed or raised."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if secret:
        out = out.replace(secret, "***")
    if check and r.returncode != 0:
        shown = [a.replace(secret, "***") if secret else a for a in args]
        raise RuntimeError(f"git {' '.join(shown)} failed:\n{out}")
    return out


def push(repo_dir, path, message, token=None, remote_url=None, branch="main",
         name="Colab export", email="colab-export@users.noreply.github.com", log=print) -> bool:
    """Commit `path` (relative to the repo) and push it. Returns False if there was nothing new.

    The token is used only in the push URL for this one command: it is not written to
    .git/config and is masked in any error message. Pulls with rebase first, so notebook
    saves made from the Colab UI in the meantime don't block the push.
    """
    repo = Path(repo_dir)
    _git(repo, "add", "-A", "--", str(path))
    if not _git(repo, "status", "--porcelain", "--", str(path)):
        log("nothing new to export")
        return False
    _git(repo, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", message,
         "--", str(path))
    url = remote_url or _git(repo, "remote", "get-url", "origin")
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://x-access-token:{token}@", 1)
    _git(repo, "pull", "-q", "--rebase", "--autostash", url, branch, secret=token)
    _git(repo, "push", "-q", url, f"HEAD:{branch}", secret=token)
    log(f"pushed: {_git(repo, 'log', '--oneline', '-1')}")
    return True
