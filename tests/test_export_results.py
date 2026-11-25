import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")          # export uses lora_exp.summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import export_results as ex  # noqa: E402
import synthetic_results as syn  # noqa: E402

DATA = ROOT / "data" / "processed"
PAIRS = {"r8_vs_r64": ("lora_r8_q-v_lr0.0002_s1", "lora_r64_q-v_lr0.0002_s1", "r=8", "r=64")}


def fake_adapter(path, r, rng):
    from safetensors.numpy import save_file
    t = {}
    for layer in range(4):
        for mod, d_out in [("q_proj", 64), ("v_proj", 16)]:
            pre = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
            t[f"{pre}.lora_A.weight"] = rng.standard_normal((r, 48)).astype(np.float32)
            t[f"{pre}.lora_B.weight"] = rng.standard_normal((d_out, r)).astype(np.float32)
    path.mkdir(parents=True)
    save_file(t, str(path / "adapter_model.safetensors"))


@pytest.fixture
def drive(tmp_path):
    """A Drive folder as the notebook leaves it: results, adapters, figures."""
    pytest.importorskip("safetensors")
    res = tmp_path / "drive" / "results"
    syn.make_results(res)
    rng = np.random.default_rng(0)
    fake_adapter(res / "adapters" / PAIRS["r8_vs_r64"][0], 8, rng)
    fake_adapter(res / "adapters" / PAIRS["r8_vs_r64"][1], 16, rng)
    figs = tmp_path / "drive" / "figures"
    figs.mkdir()
    (figs / "accuracy_vs_params.png").write_bytes(b"\x89PNG fake")
    return res, figs


def test_export_contents(drive, tmp_path):
    res, figs = drive
    out = tmp_path / "repo" / "results" / "colab_L4"
    ex.export(res, DATA, figs, out, adapters=res / "adapters", pairs=PAIRS, log=lambda *_: None)

    runs = {p.name for p in (out / "runs").iterdir()}
    assert {p.name for p in res.glob("*.json")} == runs                     # every run file
    assert not list(out.rglob("*.safetensors"))                           # never the adapters
    assert (out / "figures" / "accuracy_vs_params.png").exists()
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# Exported")

    acc = pd.read_csv(out / "tables" / "accuracy.csv", index_col=0)
    assert {"all", "mc_rules", "mc_history", "all_ci_low"} <= set(acc.columns)
    comp = pd.read_csv(out / "tables" / "comparisons.csv")
    assert ((comp.A == "full_lr2e-05") & (comp.B == "lora_r8_q-v_lr0.0002")).any()
    summary = pd.read_csv(out / "tables" / "summary_test.csv")
    assert "all_strict_mean" in summary.columns and "config" in summary.columns
    assert json.loads((out / "tables" / "trend.json").read_text())["n_configs"] == 11
    for side in ("input", "output"):
        s = pd.read_csv(out / "tables" / f"subspace_r8_vs_r64_{side}.csv")
        assert list(s.module) == ["q_proj", "v_proj"] and (s.side == side).all()


def test_export_replaces_stale_files_and_survives_missing_adapters(drive, tmp_path):
    res, figs = drive
    out = tmp_path / "out"
    (out / "runs").mkdir(parents=True)
    (out / "runs" / "deleted_run.json").write_text("{}")
    msgs = []
    ex.export(res, DATA, figs, out, adapters=tmp_path / "nowhere", pairs=PAIRS, log=msgs.append)
    assert not (out / "runs" / "deleted_run.json").exists()
    assert any("skipped subspace" in m for m in msgs)
    assert not list((out / "tables").glob("subspace_*"))


# ------------------------------------------------------------------ push
def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A bare 'GitHub' repo with one commit on main, and a clone of it (the Colab checkout)."""
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    bare = tmp_path / "origin.git"
    git("init", "-q", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git("clone", "-q", str(bare), str(seed))
    (seed / "README.md").write_text("hi\n")
    git("add", "README.md", cwd=seed)
    git("commit", "-q", "-m", "init", cwd=seed)
    git("push", "-q", "origin", "HEAD:main", cwd=seed)
    clone = tmp_path / "colab"
    git("clone", "-q", "-b", "main", str(bare), str(clone))
    return bare, seed, clone


def test_push_commits_only_the_export_and_rebases_over_new_remote_commits(remote):
    bare, seed, clone = remote
    out = clone / "results" / "colab_L4"
    out.mkdir(parents=True)
    (out / "a.json").write_text("{}")
    (clone / "scratch.txt").write_text("not part of the export\n")      # untracked noise in the checkout
    # Meanwhile the notebook is saved to GitHub from the Colab UI:
    (seed / "notebook.ipynb").write_text("{}")
    git("add", "notebook.ipynb", cwd=seed)
    git("commit", "-q", "-m", "Created using Colab", cwd=seed)
    git("push", "-q", "origin", "HEAD:main", cwd=seed)

    assert ex.push(clone, "results/colab_L4", "Add Colab results", log=lambda *_: None)
    files = git("--git-dir", str(bare), "ls-tree", "-r", "--name-only", "main")
    assert "results/colab_L4/a.json" in files and "notebook.ipynb" in files
    assert "scratch.txt" not in files
    log = git("--git-dir", str(bare), "log", "--format=%s", "main")
    assert log.splitlines()[:2] == ["Add Colab results", "Created using Colab"]
    assert ex.push(clone, "results/colab_L4", "again", log=lambda *_: None) is False   # nothing new


def test_token_never_shown(remote):
    _, _, clone = remote
    with pytest.raises(RuntimeError) as e:
        ex._git(clone, "push", "https://x-access-token:SECRET123@127.0.0.1:1/x.git", "HEAD:main",
                secret="SECRET123")
    assert "SECRET123" not in str(e.value) and "***" in str(e.value)
    assert "SECRET123" not in (clone / ".git" / "config").read_text()
