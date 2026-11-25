"""Analysis of saved experiment results. No GPU and no retraining needed.

Everything here works from the files `lora_exp.run` already wrote:
<results>/<run>.json, <results>/<run>.predictions.json and
<results>/adapters/<run>/adapter_model.safetensors.

    A = Results(RESULTS_DIR, DATA_DIR)
    A.accuracy_table()          # accuracy by question type, seeds averaged
    A.comparisons()             # paired bootstrap CIs + sign-flip permutation p-values
    A.trend()                   # does accuracy change with log10(trainable params)?
    A.stats_closeness()         # how close are the numeric answers?
    A.mc_baselines()            # chance / always-A / longest-answer baselines
    A.plot_accuracy_vs_params("accuracy_vs_params.png")

    sim = subspace_similarity(adapter_dir_r8, adapter_dir_r64)   # paper Section 7.2, Fig. 3
    plot_subspace(sim, "subspace.png")

Statistics. Every method is scored on the same 100 test items, so comparisons
are paired. Seeds are averaged per item first (seed-to-seed spread is small
compared with item-sampling noise), then:
  * confidence intervals: bootstrap over items (resample the 100 items);
  * p-values: sign-flip permutation test on the per-item differences.
The trend slope is bootstrapped over items too, which respects the fact that
all configurations share the same items.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import first_line, parse_numbers, strip_markdown

BASE_ZERO, BASE_FEW = "base_zero_shot", "base_few_shot"

# Reference palette (validated: all-pairs CVD dE >= 9.2, normal-vision dE >= 24).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ====================================================================== loading
def load_items(data_dir: str | Path, split: str = "test") -> dict[str, dict]:
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return {it["id"]: it for it in (json.loads(l) for l in f if l.strip())}


def load_runs(results_dir: str | Path, split: str = "test", include_tagged: bool = False) -> list[dict]:
    """Finished runs with predictions. Skips OOM, pipeline-check (n_train) and tagged runs."""
    runs = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name.endswith(".predictions.json"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r.get("config", {})
        if r.get("eval_split", "test") != split or r.get("oom") or c.get("n_train"):
            continue
        if c.get("tag") and not include_tagged:
            continue
        pp = p.with_name(p.stem + ".predictions.json")
        if not pp.exists():
            continue
        r["_preds"] = json.loads(pp.read_text(encoding="utf-8"))
        runs.append(r)
    return runs


def config_name(r: dict) -> str:
    return r["name"].replace(f"_s{r['config']['seed']}", "")


def describe(r: dict) -> dict:
    """Family and a short plot label for a trained run."""
    c = r["config"]
    m, rank, targets = c["method"], c["rank"], tuple(c["targets"])
    short = ",".join(t.replace("_proj", "") for t in targets)
    if m == "lora" and set(targets) == {"q_proj", "v_proj"}:
        return {"family": "LoRA on q,v (rank sweep)", "label": f"r={rank}"}
    if m == "lora":
        return {"family": "LoRA, other modules / QLoRA", "label": f"{short} r={rank}"}
    if m == "qlora":
        return {"family": "LoRA, other modules / QLoRA", "label": f"QLoRA r={rank}"}
    if m == "full":
        return {"family": "Full-weight fine-tuning", "label": "Full FT"}
    if m == "top2":
        return {"family": "Full-weight fine-tuning", "label": "Last 2 layers"}
    return {"family": m, "label": m}


# ====================================================================== statistics
def bootstrap_mean_ci(x: np.ndarray, B: int = 10000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boots = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def sign_flip_p(d: np.ndarray, B: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation p-value for mean(d) = 0 with paired differences d."""
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (signs * d).mean(axis=1)
    obs = abs(d.mean())
    return float((np.sum(np.abs(null) >= obs - 1e-12) + 1) / (B + 1))


def paired_compare(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 0) -> dict:
    """b - a on the same items: difference, bootstrap CI, permutation p."""
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi = bootstrap_mean_ci(d, B, seed)
    return {"acc_a": float(np.mean(a)), "acc_b": float(np.mean(b)), "diff": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "p": sign_flip_p(d, B, seed)}


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


# ====================================================================== numeric answers
def predicted_number(response: str) -> float | None:
    """The number the model gave: first number on the answer line, else anywhere."""
    nums = parse_numbers(first_line(response)) or parse_numbers(strip_markdown(response))
    return nums[0] if nums else None


def relative_error(pred: float | None, value: float) -> float:
    if pred is None:
        return math.inf
    return abs(pred - value) / abs(value) if value else abs(pred - value)


# ====================================================================== main object
class Results:
    def __init__(self, results_dir: str | Path, data_dir: str | Path, metric: str = "strict",
                 split: str = "test", include_tagged: bool = False):
        self.results_dir = Path(results_dir)
        self.metric = metric
        self.items = load_items(data_dir, split)
        self.runs = load_runs(results_dir, split, include_tagged)
        if not self.runs:
            raise ValueError(f"no finished runs with predictions in {results_dir}")
        self._build()

    # ------------------------------------------------------------ item matrix
    def _settings(self, r):
        if r["config"]["method"] == "base":
            return [(BASE_ZERO, r["_preds"]["zero_shot"]), (BASE_FEW, r["_preds"]["few_shot"])]
        return [(config_name(r), r["_preds"]["zero_shot"])]

    def _build(self):
        per_seed, meta = {}, {}
        for r in self.runs:
            for name, preds in self._settings(r):
                per_seed.setdefault(name, []).append({p["id"]: float(p[self.metric]) for p in preds})
                if name not in meta:
                    base = r["config"]["method"] == "base"
                    meta[name] = {"method": r["config"]["method"],
                                  "trainable_params": 0 if base else r["trainable_params"],
                                  "n_seeds": 0, **({"family": "Base model",
                                                    "label": "few-shot" if name == BASE_FEW else "zero-shot"}
                                                   if base else describe(r))}
                meta[name]["n_seeds"] += 1
        ids = sorted(self.items)
        self.mat = pd.DataFrame({n: pd.DataFrame(s).reindex(columns=ids).mean(axis=0)
                                 for n, s in per_seed.items()})
        self.meta = pd.DataFrame(meta).T
        self.meta["trainable_params"] = self.meta["trainable_params"].astype(float)
        order = self.meta.sort_values("trainable_params").index
        self.mat, self.meta = self.mat[order], self.meta.loc[order]
        cat = {}
        for i, it in self.items.items():
            cat[i] = "mc_rules" if it["source"] == "rules" else (
                "mc_history" if it["qtype"] == "mc" else it["qtype"])
        self.item_cat = pd.Series(cat).reindex(self.mat.index)
        self.item_type = pd.Series({i: it["qtype"] for i, it in self.items.items()}).reindex(self.mat.index)

    def find(self, method: str, rank: int | None = None, targets=("q_proj", "v_proj")) -> str | None:
        """Name of the configuration with this method (and rank/targets for LoRA)."""
        for r in self.runs:
            c = r["config"]
            if c["method"] != method:
                continue
            if method in ("lora", "qlora") and (c["rank"] != rank or tuple(c["targets"]) != tuple(targets)):
                continue
            return config_name(r)
        return None

    # ------------------------------------------------------------ tables
    def accuracy_table(self) -> pd.DataFrame:
        rows = {}
        for name in self.mat.columns:
            col = self.mat[name]
            row = {"n_seeds": self.meta.loc[name, "n_seeds"],
                   "trainable_params": self.meta.loc[name, "trainable_params"],
                   "all": col.mean()}
            for q in ["mc", "short", "stats"]:
                row[q] = col[self.item_type == q].mean()
            row["mc_rules"] = col[self.item_cat == "mc_rules"].mean()
            row["mc_history"] = col[self.item_cat == "mc_history"].mean()
            lo, hi = bootstrap_mean_ci(col.to_numpy(), 4000)
            row["all_ci_low"], row["all_ci_high"] = lo, hi
            rows[name] = row
        return pd.DataFrame(rows).T

    def comparisons(self, pairs: list[tuple[str, str]] | None = None, B: int = 10000) -> pd.DataFrame:
        """Paired comparisons. Default: every method vs few-shot base, plus the paper's key pairs."""
        if pairs is None:
            trained = [n for n in self.mat.columns if n not in (BASE_ZERO, BASE_FEW)]
            pairs = [(BASE_FEW, n) for n in trained]
            lora, full = self.find("lora", 8), self.find("full")
            qlora, top2 = self.find("qlora", 8), self.find("top2")
            pairs += [(a, b) for a, b in [(full, lora), (lora, qlora), (top2, lora), (BASE_ZERO, BASE_FEW)]
                      if a and b and a in self.mat and b in self.mat]
        rows = []
        for a, b in pairs:
            res = paired_compare(self.mat[a].to_numpy(), self.mat[b].to_numpy(), B)
            rows.append({"A": a, "B": b, **res})
        return pd.DataFrame(rows)

    def trend(self, B: int = 10000, seed: int = 0) -> dict:
        """OLS slope of accuracy on log10(trainable params) across trained configurations.

        Units: accuracy points per 10x more trainable parameters. CI and p from a
        bootstrap over test items (all configurations share the items).
        """
        names = [n for n in self.mat.columns if self.meta.loc[n, "trainable_params"] > 0]
        x = np.log10(self.meta.loc[names, "trainable_params"].to_numpy(float))
        Y = self.mat[names].to_numpy()                     # items x configs
        slope = ols_slope(x, Y.mean(axis=0))
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, Y.shape[0], size=(B, Y.shape[0]))
        boots = np.array([ols_slope(x, Y[i].mean(axis=0)) for i in idx])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        return {"slope_per_10x_params": slope, "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)), "p": float(min(max(p, 1 / B), 1.0)),
                "n_configs": len(names), "configs": names}

    def stats_closeness(self, tolerances=(0.01, 0.05, 0.10)) -> pd.DataFrame:
        """Numeric answers: share within each relative tolerance, and median relative error.

        Exact accuracy is 0 for every method, so this shows partial progress.
        Pooled over seeds. A response with no number counts as infinitely wrong.
        """
        stats_ids = {i for i, it in self.items.items() if it["qtype"] == "stats"}
        errs: dict[str, list[float]] = {}
        for r in self.runs:
            for name, preds in self._settings(r):
                for p in preds:
                    if p["id"] in stats_ids:
                        v = self.items[p["id"]]["value"]
                        errs.setdefault(name, []).append(relative_error(predicted_number(p["response"]), v))
        rows = {}
        for name in self.mat.columns:
            e = np.array(errs.get(name, []))
            if not len(e):
                continue
            row = {f"within_{int(t * 100)}pct": float((e <= t).mean()) for t in tolerances}
            row["median_rel_error"] = float(np.median(e))
            row["no_number"] = float(np.isinf(e).mean())
            rows[name] = row
        return pd.DataFrame(rows).T

    def mc_baselines(self) -> pd.DataFrame:
        """Accuracy of answer-blind strategies on the test MC items."""
        mc = [it for it in self.items.values() if it["qtype"] == "mc"]
        def longest(it):
            return max(it["choices"], key=lambda k: (len(it["choices"][k]), -ord(k)))
        rows = {}
        for group, sel in [("all MC", mc), ("rules MC", [i for i in mc if i["source"] == "rules"]),
                           ("history MC", [i for i in mc if i["source"] != "rules"])]:
            if not sel:
                continue
            rows[group] = {"n": len(sel), "chance": 0.25,
                           "always_A": np.mean([i["answer"] == "A" for i in sel]),
                           "best_single_letter": max(np.mean([i["answer"] == L for i in sel]) for L in "ABCD"),
                           "longest_answer": np.mean([longest(i) == i["answer"] for i in sel])}
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------ figure
    def plot_accuracy_vs_params(self, path: str | Path | None = None, B: int = 4000):
        """Accuracy vs trainable parameters (log x), 95% item-bootstrap CIs.

        Points with the same parameter count (e.g. LoRA r=8 and QLoRA r=8) are
        nudged sideways by a few percent so their error bars don't overlap.
        """
        import matplotlib.pyplot as plt
        from matplotlib import patheffects
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FixedLocator, NullLocator

        styles = {"LoRA on q,v (rank sweep)": (BLUE, "o"),
                  "LoRA, other modules / QLoRA": (ORANGE, "D"),
                  "Full-weight fine-tuning": (AQUA, "s")}
        fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        acc = self.mat.mean(axis=0) * 100
        ci = {n: [100 * v for v in bootstrap_mean_ci(self.mat[n].to_numpy(), B)] for n in self.mat.columns}
        trained = [n for n in self.mat.columns if self.meta.loc[n, "trainable_params"] > 0]

        # Plot positions: nudge configurations that share a parameter count
        # (blue family first, so the rank sweep stays at the true value).
        xpos, seen = {}, {}
        fam_rank = {f: k for k, f in enumerate(styles)}
        for n in sorted(trained, key=lambda n: fam_rank.get(self.meta.loc[n, "family"], 9)):
            p = float(self.meta.loc[n, "trainable_params"])
            k = seen.get(p, 0)
            seen[p] = k + 1
            xpos[n] = p * 10 ** (0.035 * k)

        handles = []
        for fam, (color, marker) in styles.items():
            names = [n for n in trained if self.meta.loc[n, "family"] == fam]
            if not names:
                continue
            x = np.array([xpos[n] for n in names])
            y = acc[names].to_numpy()
            err = np.array([[y[k] - ci[n][0], ci[n][1] - y[k]] for k, n in enumerate(names)]).T
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, elinewidth=1.2, alpha=0.4, zorder=2)
            if fam.startswith("LoRA on q,v"):
                order = np.argsort(x)
                ax.plot(x[order], y[order], color=color, lw=2, zorder=3)
            ax.scatter(x, y, s=58, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=4)
            handles.append(Line2D([], [], marker=marker, color=color, lw=2 if "rank" in fam else 0,
                                  markersize=7, markeredgecolor=SURFACE, label=fam))
        for name, ls, text in [(BASE_ZERO, (0, (4, 3)), "Base model, zero-shot"),
                               (BASE_FEW, (0, (1, 2)), "Base model, few-shot")]:
            if name in acc:
                ax.axhline(acc[name], color=MUTED, lw=1.2, ls=ls, zorder=1)
                handles.append(Line2D([], [], color=MUTED, lw=1.2, ls=ls, label=f"{text} ({acc[name]:.0f}%)"))

        xmin = max(min(xpos.values()) / 2.5, 1e4)
        xmax = max(xpos.values()) * 2.5
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        names_ = {1e4: "10K", 1e5: "100K", 1e6: "1M", 1e7: "10M", 1e8: "100M", 1e9: "1B", 1e10: "10B"}
        ticks = [t for t in names_ if xmin <= t <= xmax]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([names_[t] for t in ticks])
        top = max(ci[n][1] for n in self.mat.columns)
        ax.set_ylim(0, math.ceil((top + 3) / 5) * 5)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.set_xlabel("Trainable parameters (log scale)", color=INK_2, fontsize=9)
        ax.set_ylabel(f"Test accuracy ({self.metric})", color=INK_2, fontsize=9)
        ax.set_title(f"Test accuracy vs. trainable parameters ({len(self.mat)} items, 95% bootstrap CIs)",
                     color=INK, fontsize=10.5, loc="left", pad=10)
        leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5)
        for t in leg.get_texts():
            t.set_color(INK_2)
        fig.tight_layout()

        # Direct labels, placed greedily where they overlap no marker, legend or earlier label.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        blocked = []
        for n in trained:
            px, py = ax.transData.transform((xpos[n], acc[n]))
            blocked.append((px - 6, py - 6, px + 6, py + 6))
        blocked.append(tuple(leg.get_window_extent(rend).extents))
        ab = ax.get_window_extent(rend).extents
        halo = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]
        cands = [(7, 6, "left"), (7, -7, "left"), (-7, 6, "right"), (-7, -7, "right"),
                 (0, 11, "center"), (0, -13, "center"), (8, 16, "left"), (8, -17, "left"),
                 (-8, 16, "right"), (-8, -17, "right"), (0, 22, "center"), (0, -24, "center")]

        def overlaps(a, b):
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        def put(n, dx, dy, ha):
            return ax.annotate(self.meta.loc[n, "label"], (xpos[n], acc[n]), xytext=(dx, dy),
                               textcoords="offset points", ha=ha, va="center", fontsize=7.5,
                               color=INK_2, zorder=5, path_effects=halo)

        pad = 1.5 * fig.dpi / 72
        for n in sorted(trained, key=lambda n: -acc[n]):
            for dx, dy, ha in cands:
                t = put(n, dx, dy, ha)
                bb = t.get_window_extent(rend).extents
                inside = bb[0] >= ab[0] and bb[2] <= ab[2] and bb[1] >= ab[1] and bb[3] <= ab[3]
                if inside and not any(overlaps(bb, o) for o in blocked):
                    break
                t.remove()
            else:                       # nowhere free: fall back to the first spot
                t = put(n, *cands[0])
                bb = t.get_window_extent(rend).extents
            blocked.append((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        return fig


# ====================================================================== subspace similarity (paper 7.2)
_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(\w+_proj)\.lora_A(?:\.default)?\.weight$")


def load_lora_A(adapter_dir: str | Path) -> dict[tuple[int, str], np.ndarray]:
    """{(layer, module): A} with A of shape (r, d_in), from a saved PEFT adapter."""
    path = Path(adapter_dir) / "adapter_model.safetensors"
    try:
        from safetensors.torch import load_file
        tensors = {k: v.float().cpu().numpy() for k, v in load_file(str(path)).items()}
    except ImportError:
        from safetensors.numpy import load_file
        tensors = load_file(str(path))
    out = {}
    for k, v in tensors.items():
        m = _KEY.search(k)
        if m:
            out[(int(m.group(1)), m.group(2))] = np.asarray(v, dtype=np.float64)
    if not out:
        raise ValueError(f"no lora_A weights found in {path}")
    return out


def phi(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Paper eq. (4): phi(i, j) = ||U1_i^T U2_j||_F^2 / min(i, j), U = right-singular vectors of A.

    Returns an (r1, r2) matrix; entry [i-1, j-1] is phi(i, j), in [0, 1].
    """
    U1 = np.linalg.svd(A1, full_matrices=False)[2]     # (r1, d): rows are right-singular vectors
    U2 = np.linalg.svd(A2, full_matrices=False)[2]
    M2 = (U1 @ U2.T) ** 2                              # (r1, r2)
    C = M2.cumsum(axis=0).cumsum(axis=1)
    i = np.arange(1, M2.shape[0] + 1)[:, None]
    j = np.arange(1, M2.shape[1] + 1)[None, :]
    return C / np.minimum(i, j)


def subspace_similarity(adapter_a: str | Path, adapter_b: str | Path, layer: int | None = None,
                        modules=("q_proj", "v_proj"), seed: int = 0) -> dict:
    """phi between two adapters' A matrices, per module, plus a random-Gaussian baseline.

    `layer` defaults to the middle layer (the paper shows layer 48 of 96).

    Summary columns (paper Section 7.2):
      phi_11      phi(1, 1): do the two adapters' top directions coincide?
      phi_1_all   phi(1, r_b): how much of A's top direction lies inside B's whole subspace?
    each for the chosen layer, averaged over all layers, and for random Gaussian matrices.
    For random subspaces E[phi(i, j)] = max(i, j) / d_in, so phi_11 ~ 1/d and phi_1_all ~ r_b/d.
    """
    A, Bm = load_lora_A(adapter_a), load_lora_A(adapter_b)
    layers = sorted({l for l, _ in A} & {l for l, _ in Bm})
    layer = layers[len(layers) // 2] if layer is None else layer
    rng = np.random.default_rng(seed)
    out = {"layer": layer, "n_layers": len(layers), "a": str(adapter_a), "b": str(adapter_b),
           "phi": {}, "summary": []}
    for mod in modules:
        if (layer, mod) not in A:
            continue
        out["phi"][mod] = phi(A[(layer, mod)], Bm[(layer, mod)])
        every = [phi(A[(l, mod)], Bm[(l, mod)]) for l in layers]
        r1, d = A[(layer, mod)].shape
        r2 = Bm[(layer, mod)].shape[0]
        rand = phi(rng.standard_normal((r1, d)), rng.standard_normal((r2, d)))
        out["phi"][f"random_{mod}"] = rand
        P = out["phi"][mod]
        out["summary"].append({"module": mod, "r_a": r1, "r_b": r2, "d_in": d,
                               "phi_11": float(P[0, 0]),
                               "phi_11_all_layers": float(np.mean([e[0, 0] for e in every])),
                               "phi_11_random": float(rand[0, 0]),
                               "phi_1_all": float(P[0, -1]),
                               "phi_1_all_all_layers": float(np.mean([e[0, -1] for e in every])),
                               "phi_1_all_random": float(rand[0, -1]),
                               "phi_1_all_random_expected": r2 / d})
    out["summary"] = pd.DataFrame(out["summary"])
    return out


def plot_subspace(sim: dict, path: str | Path | None = None, label_a: str = "r=8", label_b: str = "r=64"):
    """Heatmaps of phi(i, j) like the paper's Figure 3: one per module plus a random baseline."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    panels = [(m, f"ΔW{m[0]} ({m})") for m in sim["phi"] if not m.startswith("random_")]
    rand_key = next((k for k in sim["phi"] if k.startswith("random_")), None)
    if rand_key:
        panels.append((rand_key, "Random Gaussian"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.9), dpi=200,
                             gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        P = sim["phi"][key]
        im = ax.imshow(P, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=(0.5, P.shape[1] + 0.5, P.shape[0] + 0.5, 0.5))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"j (top-j directions, {label_b})", fontsize=8, color=INK_2)
        ax.set_yticks(range(1, P.shape[0] + 1))
        ax.tick_params(labelsize=7, colors=INK_2, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel(f"i (top-i directions, {label_a})", fontsize=8, color=INK_2)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    cb.set_label("φ(i, j)", fontsize=8, color=INK_2)
    fig.suptitle(f"Subspace similarity, layer {sim['layer']} of {sim['n_layers']} "
                 f"(paper eq. 4)", x=0.02, ha="left", fontsize=10.5, color=INK, y=1.04)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return fig
