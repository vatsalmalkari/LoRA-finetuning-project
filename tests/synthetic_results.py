"""Fake results folders in the exact format lora_exp.run writes, for testing analysis.py.

Items come from the real frozen test set, so categories (rules vs history MC,
stats values) are realistic. Accuracies default to roughly what the Colab runs gave.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "processed" / "test.jsonl"

# Gemma 3 1B attention shapes: d_model 1152; q 1152->1024, k/v 1152->256, o 1024->1152; 26 layers.
SHAPES = {"q_proj": 1152 + 1024, "k_proj": 1152 + 256, "v_proj": 1152 + 256, "o_proj": 1024 + 1152}


def lora_params(rank, targets):
    return 26 * rank * sum(SHAPES[t] for t in targets)


# (method, rank, targets, seeds, accuracy)
DEFAULT_SPEC = [
    ("lora", 1, ("q_proj", "v_proj"), [1], 0.18),
    ("lora", 2, ("q_proj", "v_proj"), [1], 0.19),
    ("lora", 4, ("q_proj", "v_proj"), [1], 0.19),
    ("lora", 8, ("q_proj", "v_proj"), [1, 100, 1000], 0.16),
    ("lora", 64, ("q_proj", "v_proj"), [1], 0.13),
    ("lora", 8, ("q_proj",), [1], 0.19),
    ("lora", 8, ("v_proj",), [1], 0.17),
    ("lora", 2, ("q_proj", "k_proj", "v_proj", "o_proj"), [1], 0.17),
    ("qlora", 8, ("q_proj", "v_proj"), [1, 100, 1000], 0.153),
    ("full", 8, ("q_proj", "v_proj"), [1, 100, 1000], 0.107),
    ("top2", 8, ("q_proj", "v_proj"), [1, 100, 1000], 0.14),
]
LR = {"lora": 2e-4, "qlora": 2e-4, "full": 2e-5, "top2": 1e-4}


def load_items():
    with open(TEST, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_name(method, rank, targets, seed, lr, tag=""):
    t = "-".join(x.replace("_proj", "") for x in targets)
    if method in ("lora", "qlora"):
        return f"{method}_r{rank}_{t}_lr{lr:g}_s{seed}{tag}"
    return f"{method}_lr{lr:g}_s{seed}{tag}"


def _response(item, correct, rng):
    if item["qtype"] == "mc":
        wrong = [L for L in "ABCD" if L != item["answer"]]
        return item["answer"] if correct else str(rng.choice(wrong))
    if item["qtype"] == "short":
        return item["answer"] if correct else "Mickey Mantle"
    v = item["value"]
    return f"{v:.{item['decimals']}f}" if correct else f"{v * (1 + rng.normal(0, 0.08)):.3f}"


def _preds(items, acc, rng):
    out = []
    for it in items:
        # Stats are never right (as in the real runs); MC carries most of the accuracy.
        p = 0.0 if it["qtype"] == "stats" else min(1.0, acc * (2.6 if it["qtype"] == "mc" else 0.45))
        ok = bool(rng.random() < p)
        out.append({"id": it["id"], "qtype": it["qtype"], "source": it["source"], "answer": it["answer"],
                    "response": _response(it, ok, rng), "strict": ok, "lenient": ok, "extracted": None})
    return out


def _config(method, rank=8, targets=("q_proj", "v_proj"), seed=1, lr=2e-4, tag="", n_train=None):
    return {"method": method, "seed": seed, "rank": rank, "alpha": 2 * rank, "targets": list(targets),
            "lr": lr, "n_train": n_train, "tag": tag}


def _accuracy(preds):
    """The `accuracy` block lora_exp.run writes: {group: {n, strict, lenient}}."""
    out = {}
    for g in ["all", "mc", "short", "stats"]:
        sel = [p for p in preds if g == "all" or p["qtype"] == g]
        if sel:
            out[g] = {"n": len(sel), "strict": float(np.mean([p["strict"] for p in sel])),
                      "lenient": float(np.mean([p["lenient"] for p in sel]))}
    return out


def _write(out, name, result, preds):
    out.mkdir(parents=True, exist_ok=True)
    result = {**result, "accuracy": _accuracy(preds["zero_shot"])}
    if "few_shot" in preds:
        result["accuracy_fewshot"] = _accuracy(preds["few_shot"])
    (out / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")
    (out / f"{name}.predictions.json").write_text(json.dumps(preds), encoding="utf-8")


def make_results(out: Path, spec=DEFAULT_SPEC, base=(0.14, 0.17), seed=0, extras=True):
    """Write a fake results folder. `extras` adds runs analysis.py must ignore."""
    rng = np.random.default_rng(seed)
    items = load_items()
    _write(out, "base", {"name": "base", "config": _config("base"), "eval_split": "test",
                         "trainable_params": 0, "total_params": 999_885_952},
           {"zero_shot": _preds(items, base[0], rng), "few_shot": _preds(items, base[1], rng)})
    for method, rank, targets, seeds, acc in spec:
        for s in seeds:
            name = run_name(method, rank, targets, s, LR[method])
            tp = {"full": 999_885_952, "top2": 53_747_712}.get(method, lora_params(rank, targets))
            _write(out, name, {"name": name, "config": _config(method, rank, targets, s, LR[method]),
                               "eval_split": "test", "trainable_params": tp},
                   {"zero_shot": _preds(items, acc, rng)})
    if extras:
        bad = {"name": None, "eval_split": "test", "trainable_params": 1}
        _write(out, "full_lr2e-05_s7_oom", {**bad, "name": "full_lr2e-05_s7_oom", "oom": True,
                                            "config": _config("full", seed=7, tag="_oom")}, {"zero_shot": []})
        _write(out, "lora_check", {**bad, "name": "lora_check", "config": _config("lora", n_train=16)},
               {"zero_shot": _preds(items, 0.9, rng)})
        _write(out, "lora_r8_q-v_lr0.0002_s1_fp16", {**bad, "name": "lora_r8_q-v_lr0.0002_s1_fp16",
                                                      "config": _config("lora", tag="_fp16")},
               {"zero_shot": _preds(items, 0.9, rng)})
        _write(out, "lora_r8_q-v_lr0.0002_s1__val", {**bad, "name": "lora_r8_q-v_lr0.0002_s1__val",
                                                      "eval_split": "val", "config": _config("lora")},
               {"zero_shot": []})
        (out / "orphan_no_predictions.json").write_text(json.dumps(
            {"name": "orphan", "config": _config("lora", seed=5), "eval_split": "test", "trainable_params": 1,
             "accuracy": {"all": {"n": 1, "strict": 0.0, "lenient": 0.0}}}))
    return items
