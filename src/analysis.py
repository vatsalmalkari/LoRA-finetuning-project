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
    def _place_labels(self, fig, ax, legend, pts: dict[str, tuple[float, float]]):
        """Direct labels that can only be read one way.

        A label next to its point must be nearer that point than any other point. If no such
        spot is free, the label moves further out with a thin leader line that passes no other
        marker. Nothing may overlap a marker, the legend or another label.
        """
        from matplotlib import patheffects
        from matplotlib.text import Text

        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        s = fig.dpi / 72                                     # points -> pixels
        disp = {n: np.asarray(ax.transData.transform(p)) for n, p in pts.items()}
        mark = 5.5 * s                                       # marker radius incl. ring
        blocked = [(x - mark, y - mark, x + mark, y + mark) for x, y in disp.values()]
        blocked.append(tuple(legend.get_window_extent(rend).extents))
        ab = ax.get_window_extent(rend).extents
        halo = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]

        def overlaps(a, b):
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        def text_box(t):
            # The label's own box. Annotation.get_window_extent would include the leader line.
            t.update_positions(rend)
            return Text.get_window_extent(t, rend).extents

        def rect_dist(bb, p):
            dx = max(bb[0] - p[0], 0, p[0] - bb[2])
            dy = max(bb[1] - p[1], 0, p[1] - bb[3])
            return math.hypot(dx, dy)

        def seg_dist(p, a, b):
            ab_, ap = b - a, p - a
            t = np.clip(ap @ ab_ / max(ab_ @ ab_, 1e-9), 0, 1)
            return float(np.linalg.norm(p - (a + t * ab_)))

        cands = []
        for dist in (7, 15, 24, 34, 46, 60):
            angles = [0, 35, -35, 180, 145, -145, 90, -90] + ([60, -60, 120, -120] if dist > 7 else [])
            for a in angles:
                c, sn = math.cos(math.radians(a)), math.sin(math.radians(a))
                ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
                va = "bottom" if sn > 0.3 else "top" if sn < -0.3 else "center"
                cands.append((dist * c, dist * sn, ha, va, dist > 7))

        def put(n, dx, dy, ha, va, leader):
            kw = {"arrowprops": {"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkA": 1,
                                 "shrinkB": 5}} if leader else {}
            return ax.annotate(self.meta.loc[n, "label"], pts[n], xytext=(dx, dy), textcoords="offset points",
                               ha=ha, va=va, fontsize=7.5, color=INK_2, zorder=5, path_effects=halo, **kw)

        # Crowded points choose first.
        crowd = {n: sum(np.linalg.norm(disp[n] - disp[m]) < 60 * s for m in disp) for n in disp}
        for n in sorted(disp, key=lambda n: (-crowd[n], -pts[n][1])):
            others = [disp[m] for m in disp if m != n]
            for dx, dy, ha, va, leader in cands:
                t = put(n, dx, dy, ha, va, leader)
                bb = text_box(t)
                ok = (bb[0] >= ab[0] and bb[2] <= ab[2] and bb[1] >= ab[1] and bb[3] <= ab[3]
                      and not any(overlaps(bb, o) for o in blocked))
                if ok and not leader:
                    own = rect_dist(bb, disp[n])
                    ok = all(rect_dist(bb, o) > own + 2 * s for o in others)
                if ok and leader:
                    near = np.array([min(max(disp[n][0], bb[0]), bb[2]), min(max(disp[n][1], bb[1]), bb[3])])
                    ok = all(seg_dist(o, disp[n], near) > mark + s for o in others)
                if ok:
                    break
                t.remove()
            else:                       # nowhere free: fall back to the first spot
                t = put(n, *cands[0])
                bb = text_box(t)
            blocked.append((bb[0] - 1.5 * s, bb[1] - 1.5 * s, bb[2] + 1.5 * s, bb[3] + 1.5 * s))

    def plot_accuracy_vs_params(self, path: str | Path | None = None, B: int = 4000):
        """Accuracy vs trainable parameters (log x), 95% item-bootstrap CIs.

        Points with the same parameter count (e.g. LoRA r=8 and QLoRA r=8) are
        nudged sideways by a few percent so their error bars don't overlap.
        """
        import matplotlib.pyplot as plt
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

        self._place_labels(fig, ax, leg, {n: (xpos[n], acc[n]) for n in trained})
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        return fig


# ====================================================================== subspace similarity (paper 7.2)
_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(\w+_proj)\.lora_([AB])(?:\.default)?\.weight$")
SIDES = {"input": "input side: row space of A (as in the paper)",
         "output": "output side: column space of ΔW = BA (learned from B = 0)"}


def load_lora(adapter_dir: str | Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """{(layer, module): {"A": (r, d_in), "B": (d_out, r)}} from a saved PEFT adapter."""
    path = Path(adapter_dir) / "adapter_model.safetensors"
    try:
        from safetensors.torch import load_file
        tensors = {k: v.float().cpu().numpy() for k, v in load_file(str(path)).items()}
    except ImportError:
        from safetensors.numpy import load_file
        tensors = load_file(str(path))
    out: dict = {}
    for k, v in tensors.items():
        m = _KEY.search(k)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = np.asarray(v, dtype=np.float64)
    if not out:
        raise ValueError(f"no LoRA weights found in {path}")
    return out


def load_lora_A(adapter_dir: str | Path) -> dict[tuple[int, str], np.ndarray]:
    """{(layer, module): A} with A of shape (r, d_in)."""
    return {k: v["A"] for k, v in load_lora(adapter_dir).items() if "A" in v}


def basis(A: np.ndarray, B: np.ndarray | None = None, side: str = "input") -> np.ndarray:
    """Orthonormal directions (rows), most important first.

    input:  right-singular vectors of A (r x d_in). This is the paper's choice. A starts as a
            random matrix, so after a short run it can still mostly reflect its initialisation.
    output: left-singular vectors of ΔW = B A (d_out x d_in). B starts at zero, so these
            directions were produced entirely by training. Computed without forming BA:
            A^T = Q R  =>  BA = (B R^T) Q^T, and Q has orthonormal columns.
    """
    if side == "input":
        return np.linalg.svd(A, full_matrices=False)[2]
    if side != "output":
        raise ValueError(f"side must be 'input' or 'output', not {side!r}")
    _, R = np.linalg.qr(A.T)
    return np.linalg.svd(B @ R.T, full_matrices=False)[0].T


def phi_from_bases(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Paper eq. (4) for orthonormal row bases: phi(i, j) = ||U1_i U2_j^T||_F^2 / min(i, j)."""
    M2 = (U1 @ U2.T) ** 2                              # (r1, r2)
    C = M2.cumsum(axis=0).cumsum(axis=1)
    i = np.arange(1, M2.shape[0] + 1)[:, None]
    j = np.arange(1, M2.shape[1] + 1)[None, :]
    return C / np.minimum(i, j)


def phi(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Paper eq. (4): phi(i, j) = ||U1_i^T U2_j||_F^2 / min(i, j), U = right-singular vectors of A.

    Returns an (r1, r2) matrix; entry [i-1, j-1] is phi(i, j), in [0, 1].
    """
    return phi_from_bases(basis(A1), basis(A2))


def subspace_similarity(adapter_a: str | Path, adapter_b: str | Path, layer: int | None = None,
                        modules=("q_proj", "v_proj"), side: str = "input", seed: int = 0) -> dict:
    """phi between two adapters, per module, plus a random-Gaussian baseline of the same shape.

    `side` chooses which subspace is compared (see `basis`). `layer` defaults to the middle
    layer (the paper shows layer 48 of 96).

    Summary columns:
      phi_11      phi(1, 1): do the two adapters' top directions coincide?
      phi_1_all   phi(1, r_b): how much of the first adapter's top direction lies in the
                  second adapter's whole subspace?
    each for the chosen layer, averaged over all layers, and for random matrices. For random
    subspaces E[phi(i, j)] = max(i, j) / d, so phi_11 ~ 1/d and phi_1_all ~ r_b/d.
      A_norm_vs_init   ||A||_F / its expected value at PEFT's initialisation (Kaiming-uniform,
                  E||A||_F^2 = r/3), first adapter, averaged over layers. Near 1 suggests A
                  is still mostly its random initialisation.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {list(SIDES)}")
    La, Lb = load_lora(adapter_a), load_lora(adapter_b)
    layers = sorted({l for l, _ in La} & {l for l, _ in Lb})
    layer = layers[len(layers) // 2] if layer is None else layer
    rng = np.random.default_rng(seed)
    out = {"layer": layer, "n_layers": len(layers), "a": str(adapter_a), "b": str(adapter_b),
           "side": side, "phi": {}, "summary": []}
    for mod in modules:
        if (layer, mod) not in La:
            continue
        bases = {l: (basis(La[(l, mod)]["A"], La[(l, mod)].get("B"), side),
                     basis(Lb[(l, mod)]["A"], Lb[(l, mod)].get("B"), side)) for l in layers}
        every = {l: phi_from_bases(*bases[l]) for l in layers}
        P = every[layer]
        out["phi"][mod] = P
        r1, r2 = bases[layer][0].shape[0], bases[layer][1].shape[0]
        d = bases[layer][0].shape[1]
        rand = phi_from_bases(basis(rng.standard_normal((r1, d))), basis(rng.standard_normal((r2, d))))
        out["phi"][f"random_{mod}"] = rand
        A_ratio = [np.linalg.norm(La[(l, mod)]["A"]) / math.sqrt(La[(l, mod)]["A"].shape[0] / 3) for l in layers]
        out["summary"].append({"module": mod, "side": side, "r_a": r1, "r_b": r2, "d": d,
                               "phi_11": float(P[0, 0]),
                               "phi_11_all_layers": float(np.mean([e[0, 0] for e in every.values()])),
                               "phi_11_random": float(rand[0, 0]),
                               "phi_1_all": float(P[0, -1]),
                               "phi_1_all_all_layers": float(np.mean([e[0, -1] for e in every.values()])),
                               "phi_1_all_random": float(rand[0, -1]),
                               "phi_1_all_random_expected": r2 / d,
                               "A_norm_vs_init": float(np.mean(A_ratio))})
    out["summary"] = pd.DataFrame(out["summary"])
    return out


def plot_subspace(sim: dict, path: str | Path | None = None, label_a: str = "r=8", label_b: str = "r=64"):
    """Heatmaps of phi(i, j) like the paper's Figure 3: one per module plus a random baseline."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    panels = [(m, f"ΔW{m[0]} ({m})") for m in sim["phi"] if not m.startswith("random_")]
    # One random baseline per distinct dimension: on the output side q (d=1024) and v (d=256)
    # differ, and random overlap is max(i, j)/d, so v needs its own reference.
    dims = dict(zip(sim["summary"]["module"], sim["summary"]["d"])) if len(sim.get("summary", [])) else {}
    seen_d = set()
    for m, _ in list(panels):
        d = dims.get(m)
        if f"random_{m}" in sim["phi"] and d not in seen_d:
            seen_d.add(d)
            panels.append((f"random_{m}", "Random Gaussian" + (f", d={d}" if d else "")))
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
    side = SIDES.get(sim.get("side", "input"), "")
    fig.suptitle(f"Subspace similarity (paper eq. 4), layer {sim['layer']} of {sim['n_layers']}, {side}",
                 x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return fig
