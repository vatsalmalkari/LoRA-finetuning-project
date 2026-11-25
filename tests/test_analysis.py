import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import analysis as an  # noqa: E402
import synthetic_results as syn  # noqa: E402

DATA = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    syn.make_results(out)
    return an.Results(out, DATA)


# ------------------------------------------------------------------ loading
def test_ignores_oom_pipeline_check_tagged_val_and_orphans(res):
    names = {r["name"] for r in res.runs}
    assert not any(n.endswith(("_oom", "_fp16", "__val")) or n in ("lora_check", "orphan") for n in names)
    assert len(res.runs) == 1 + 7 + 4 * 3   # base + 7 single-seed configs + 4 configs x 3 seeds


def test_item_matrix_shape_and_seed_pooling(res):
    assert res.mat.shape == (100, 2 + 7 + 4)
    assert {an.BASE_ZERO, an.BASE_FEW} <= set(res.mat.columns)
    lora8 = res.find("lora", 8)
    assert lora8 == "lora_r8_q-v_lr0.0002"
    assert res.meta.loc[lora8, "n_seeds"] == 3 and res.meta.loc["lora_r1_q-v_lr0.0002", "n_seeds"] == 1
    # Seed-averaged cells take values in {0, 1/3, 2/3, 1}.
    assert set(np.round(res.mat[lora8].unique() * 3, 6)) <= {0, 1, 2, 3}
    assert res.find("full") == "full_lr2e-05" and res.find("lora", 16) is None


def test_meta_families_and_labels(res):
    m = res.meta
    assert m.loc["lora_r64_q-v_lr0.0002", "label"] == "r=64"
    assert m.loc["lora_r8_q_lr0.0002", "family"] == "LoRA, other modules / QLoRA"
    assert m.loc["lora_r2_q-k-v-o_lr0.0002", "label"] == "q,k,v,o r=2"
    assert m.loc["top2_lr0.0001", "label"] == "Last 2 layers"
    assert m.loc["lora_r8_q-v_lr0.0002", "trainable_params"] == 745_472     # matches the Colab run
    assert list(m["trainable_params"]) == sorted(m["trainable_params"])


def test_accuracy_table(res):
    t = res.accuracy_table()
    row = t.loc["lora_r8_q-v_lr0.0002"]
    assert row["all"] == pytest.approx(res.mat["lora_r8_q-v_lr0.0002"].mean())
    # all = weighted mean of mc/short/stats (33/33/34 items); mc = weighted mean of rules/history (16/17).
    assert row["all"] == pytest.approx((33 * row["mc"] + 33 * row["short"] + 34 * row["stats"]) / 100)
    assert row["mc"] == pytest.approx((16 * row["mc_rules"] + 17 * row["mc_history"]) / 33)
    assert row["all_ci_low"] <= row["all"] <= row["all_ci_high"]
    assert (t["stats"] == 0).all()


# ------------------------------------------------------------------ statistics
def test_paired_compare_direction_and_identity():
    a = np.array([0, 1, 0, 1, 0, 0, 1, 0] * 10, float)
    same = an.paired_compare(a, a, B=2000)
    assert same["diff"] == 0 and same["p"] == 1.0
    better = an.paired_compare(a, np.ones_like(a), B=2000)
    assert better["diff"] == pytest.approx(1 - a.mean()) and better["p"] < 0.001
    assert 0 < better["ci_low"] <= better["diff"] <= better["ci_high"]


def test_sign_flip_p_is_calibrated_under_null():
    rng = np.random.default_rng(1)
    ps = [an.sign_flip_p(rng.choice([-1.0, 0, 0, 1.0], 100), B=500, seed=k) for k in range(200)]
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.10


def test_comparisons_default_pairs(res):
    c = res.comparisons(B=1000)
    pairs = set(zip(c.A, c.B))
    assert (an.BASE_FEW, "lora_r8_q-v_lr0.0002") in pairs
    assert ("full_lr2e-05", "lora_r8_q-v_lr0.0002") in pairs          # the paper's comparison
    assert ("lora_r8_q-v_lr0.0002", "qlora_r8_q-v_lr0.0002") in pairs
    row = c[(c.A == "full_lr2e-05") & (c.B == "lora_r8_q-v_lr0.0002")].iloc[0]
    assert row["diff"] == pytest.approx(row["acc_b"] - row["acc_a"])
    assert ((c.p > 0) & (c.p <= 1)).all()


def test_trend_detects_a_built_in_slope(tmp_path):
    # Accuracy falls steeply with log params: the slope and its CI must be negative.
    spec = [("lora", r, ("q_proj", "v_proj"), [1], acc) for r, acc in [(1, .30), (4, .22), (16, .14), (64, .06)]]
    syn.make_results(tmp_path, spec=spec, extras=False)
    t = an.Results(tmp_path, DATA).trend(B=2000)
    assert t["n_configs"] == 4 and t["slope_per_10x_params"] < 0
    assert t["ci_high"] < 0 and t["p"] < 0.05


def test_trend_flat_is_not_significant(tmp_path):
    spec = [("lora", r, ("q_proj", "v_proj"), [1], 0.2) for r in (1, 4, 16, 64)]
    syn.make_results(tmp_path, spec=spec, extras=False)
    t = an.Results(tmp_path, DATA).trend(B=2000)
    assert t["ci_low"] < 0 < t["ci_high"]


def test_ols_slope():
    assert an.ols_slope(np.array([0., 1, 2]), np.array([1., 3, 5])) == pytest.approx(2)


# ------------------------------------------------------------------ stats closeness
@pytest.mark.parametrize("resp,want", [
    ("0.287", 0.287), ("```\n.287\n```", 0.287), ("Answer: 3.46", 3.46),
    ("His average was about 0.3 overall", 0.3), ("I don't know", None),
])
def test_predicted_number(resp, want):
    got = an.predicted_number(resp)
    assert got == pytest.approx(want) if want is not None else got is None


def test_relative_error():
    assert an.relative_error(0.30, 0.25) == pytest.approx(0.2)
    assert an.relative_error(None, 0.25) == np.inf


def test_stats_closeness(res):
    s = res.stats_closeness()
    assert {"within_1pct", "within_5pct", "within_10pct", "median_rel_error", "no_number"} <= set(s.columns)
    assert (s["within_1pct"] <= s["within_5pct"]).all() and (s["within_5pct"] <= s["within_10pct"]).all()
    # Synthetic wrong answers are value * (1 + N(0, 0.08)): median error should be near 0.054.
    assert 0.02 < s.loc["lora_r8_q-v_lr0.0002", "median_rel_error"] < 0.10
    assert (s["no_number"] == 0).all()


def test_stats_closeness_counts_missing_numbers(tmp_path):
    syn.make_results(tmp_path, spec=[], extras=False)
    import json
    p = tmp_path / "base.predictions.json"
    preds = json.loads(p.read_text())
    for x in preds["zero_shot"]:
        if x["qtype"] == "stats":
            x["response"] = "I'm not sure."
    p.write_text(json.dumps(preds))
    s = an.Results(tmp_path, DATA).stats_closeness()
    assert s.loc[an.BASE_ZERO, "no_number"] == 1.0 and s.loc[an.BASE_ZERO, "within_10pct"] == 0
    assert s.loc[an.BASE_FEW, "no_number"] == 0


# ------------------------------------------------------------------ MC baselines
def test_mc_baselines(res):
    b = res.mc_baselines()
    assert list(b.index) == ["all MC", "rules MC", "history MC"]
    assert b.loc["all MC", "n"] == 33 and b.loc["rules MC", "n"] == 16
    items = [it for it in an.load_items(DATA).values() if it["qtype"] == "mc"]
    assert b.loc["all MC", "always_A"] == pytest.approx(np.mean([it["answer"] == "A" for it in items]))
    assert (b["best_single_letter"] >= b["always_A"]).all()
    assert ((b["longest_answer"] >= 0) & (b["longest_answer"] <= 1)).all()


# ------------------------------------------------------------------ subspace similarity
def test_phi_identity_and_bounds():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 300))
    P = an.phi(A, A)
    assert P.shape == (8, 8)
    assert np.allclose(np.diag(P), 1)
    assert P.min() >= -1e-12 and P.max() <= 1 + 1e-12


def test_phi_nested_subspace_and_random():
    rng = np.random.default_rng(0)
    big = rng.standard_normal((64, 1000))
    small = rng.standard_normal((8, 8)) @ big[:8]        # an 8-dim subspace of the 64-row span
    P = an.phi(small, big)
    assert P.shape == (8, 64)
    assert np.allclose(P[:, -1], 1)                      # all 64 directions contain the 8-dim subspace
    R = an.phi(rng.standard_normal((8, 1000)), rng.standard_normal((64, 1000)))
    assert R[0, 0] < 0.2                                 # expected ~ 64/1000


def test_subspace_similarity_from_adapter_files(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)

    def adapter(path, r):
        t = {}
        for layer in range(4):
            for mod, d in [("q_proj", 64), ("v_proj", 64)]:
                pre = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
                t[f"{pre}.lora_A.weight"] = rng.standard_normal((r, d)).astype(np.float32)
                t[f"{pre}.lora_B.weight"] = np.zeros((32, r), np.float32)
        path.mkdir()
        save_file(t, str(path / "adapter_model.safetensors"))

    adapter(tmp_path / "r8", 8)
    adapter(tmp_path / "r64", 32)
    A = an.load_lora_A(tmp_path / "r8")
    assert len(A) == 8 and A[(0, "q_proj")].shape == (8, 64)
    sim = an.subspace_similarity(tmp_path / "r8", tmp_path / "r64")
    assert sim["layer"] == 2 and sim["phi"]["q_proj"].shape == (8, 32)
    assert {"q_proj", "v_proj", "random_q_proj", "random_v_proj"} == set(sim["phi"])
    assert list(sim["summary"]["module"]) == ["q_proj", "v_proj"]


# ------------------------------------------------------------------ figures
def test_plot_accuracy_vs_params(res, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    fig = res.plot_accuracy_vs_params(tmp_path / "fig.png", B=200)
    assert (tmp_path / "fig.png").stat().st_size > 10_000
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert {"r=1", "r=64", "Full FT", "Last 2 layers", "QLoRA r=8"} <= texts
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(t.startswith("Base model, few-shot") for t in legend)


def test_plot_subspace(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((8, 100)), rng.standard_normal((64, 100))
    sim = {"layer": 13, "n_layers": 26,
           "phi": {"q_proj": an.phi(A, B), "random_q_proj": an.phi(rng.standard_normal((8, 100)), B)}}
    an.plot_subspace(sim, tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 5_000
