"""Training, evaluation and measurement for the LoRA replication experiments.

One entry point, `run(cfg, ...)`, does a complete experiment:
load model -> (attach LoRA / quantize / unfreeze) -> train -> evaluate ->
measure latency -> write results JSON. Results are skipped if they already
exist, so a Colab disconnect never loses finished runs.

Methods
  base   no training; evaluated zero-shot and few-shot
  lora   LoRA on `targets` (paper eq. 3: h = W0 x + (alpha/r) B A x), base weights frozen in bf16
  qlora  same LoRA, base weights frozen in 4-bit NF4 (Dettmers et al., 2023)
  full   full fine-tuning: every weight trained, fp32 master weights + bf16 autocast
  top2   only the last two decoder layers trained (the paper's FT^Top2 baseline)

Loss is computed on the answer tokens only (paper eq. 1 conditions on x and
sums log-probs of y), averaged over all answer tokens in the optimizer step.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from scoring import score

MODEL_ID = "google/gemma-3-1b-it"


# ====================================================================== config
@dataclass
class RunConfig:
    method: str                       # base | lora | qlora | full | top2
    seed: int = 1
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 8               # effective batch (per optimizer step)
    micro_batch: int = 4              # per forward pass; batch_size / micro_batch = grad accumulation
    max_len: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    precision: str = "bf16"           # bf16 (L4/A100) | fp32 | fp16 (T4; Gemma 3 may overflow)
    grad_checkpointing: bool = True   # same for every method so memory numbers are comparable
    n_train: int | None = None        # use only the first n training examples (pipeline check)
    n_shot: int = 3                   # few-shot examples for the base model
    tag: str = ""                     # free-form suffix for the run name

    @property
    def name(self) -> str:
        if self.method == "base":
            return f"base{self.tag}"
        parts = [self.method]
        if self.method in ("lora", "qlora"):
            parts += [f"r{self.rank}", "-".join(t.replace("_proj", "") for t in self.targets)]
        parts += [f"lr{self.lr:g}", f"s{self.seed}"]
        if self.n_train:
            parts.append(f"n{self.n_train}")
        return "_".join(parts) + self.tag


# ====================================================================== data
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_frozen_test(processed_dir: str | Path) -> str:
    """Refuse to evaluate if test.jsonl changed since it was frozen."""
    d = Path(processed_dir)
    want = (d / "test.sha256").read_text().strip()
    got = hashlib.sha256((d / "test.jsonl").read_bytes()).hexdigest()
    if want != got:
        raise RuntimeError("test.jsonl does not match test.sha256: the frozen test set was modified")
    return got


def few_shot_pool(train: list[dict], n_shot: int, seed: int = 0) -> dict[str, list[dict]]:
    """Fixed few-shot examples per question type (same examples for every test item)."""
    rng = random.Random(seed)
    pool = {}
    for qtype in sorted({it["qtype"] for it in train}):
        items = [it for it in train if it["qtype"] == qtype]
        pool[qtype] = rng.sample(items, n_shot)
    return pool


def prompt_messages(item: dict, shots: list[dict] | None = None) -> list[dict]:
    msgs = []
    for ex in shots or []:
        msgs += ex["messages"]
    msgs.append({"role": "user", "content": item["question"]})
    return msgs


def encode_example(tok, item: dict, max_len: int) -> dict:
    """Token ids with labels masked to -100 everywhere except the answer."""
    prompt = tok.apply_chat_template(item["messages"][:1], add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(item["messages"], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: full conversation does not start with the prompt")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("tokenization of prompt is not a prefix of the full conversation")
    f_ids = f_ids[:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[:max_len]}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# ====================================================================== model
def _dtype_kw(dtype) -> dict:
    """transformers renamed torch_dtype -> dtype in 4.56."""
    import transformers
    from packaging import version
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    return {key: dtype}


def decoder_layers(model) -> torch.nn.ModuleList:
    """The transformer block list, whatever the wrapper (works for Gemma3ForCausalLM)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise ValueError("could not find decoder layers")
    return best


def count_params(model) -> tuple[int, int]:
    """(trainable, total). 4-bit weights are stored packed two per byte; count logical params."""
    trainable = total = 0
    for p in model.parameters():
        n = p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def load_model(cfg: RunConfig, model_id: str = MODEL_ID, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    compute = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.precision]
    kw = {"attn_implementation": "eager"}  # recommended for Gemma 3 training; used everywhere for consistency

    if cfg.method == "qlora":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
        kw["device_map"] = {"": 0}
        kw.update(_dtype_kw(compute))
    elif cfg.method in ("full", "top2"):
        kw.update(_dtype_kw(torch.float32))   # fp32 master weights; autocast does bf16 math
    else:                                      # base, lora: frozen weights in the compute dtype
        kw.update(_dtype_kw(compute))

    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if cfg.method != "qlora":
        model.to(device)
    model.config.use_cache = False
    return model, tok


def prepare_for_training(model, cfg: RunConfig):
    if cfg.method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        if cfg.method == "qlora":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        lcfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                          target_modules=list(cfg.targets), bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)  # B = 0, A ~ random: Delta W = BA = 0 at start
    elif cfg.method == "top2":
        for p in model.parameters():
            p.requires_grad = False
        for layer in decoder_layers(model)[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif cfg.method == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"method {cfg.method!r} is not trainable")

    if cfg.grad_checkpointing:
        kw = {"use_reentrant": False}
        if hasattr(model, "activation_checkpointing_enable"):   # transformers >= 5 name
            model.activation_checkpointing_enable(activation_checkpointing_kwargs=kw)
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
    return model


# ====================================================================== train
def _autocast(cfg: RunConfig, device: str):
    if device != "cuda" or cfg.precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if cfg.precision == "bf16" else torch.float16)


def train(model, tok, train_items: list[dict], cfg: RunConfig, log=print) -> dict:
    from transformers import get_cosine_schedule_with_warmup

    device = next(p.device for p in model.parameters()).type
    examples = [encode_example(tok, it, cfg.max_len) for it in train_items]
    if cfg.batch_size % cfg.micro_batch:
        raise ValueError("batch_size must be a multiple of micro_batch")

    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            fused=(device == "cuda"))
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.precision == "fp16"))

    order_rng = random.Random(cfg.seed)
    losses, tokens_seen, step = [], 0, 0
    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        order_rng.shuffle(order)
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start: start + cfg.batch_size]
            n_target = sum(sum(l != -100 for l in examples[i]["labels"][1:]) for i in idx)
            step_loss = 0.0
            for m in range(0, len(idx), cfg.micro_batch):
                b = collate([examples[i] for i in idx[m: m + cfg.micro_batch]], tok.pad_token_id)
                b = {k: v.to(device) for k, v in b.items()}
                tokens_seen += int(b["attention_mask"].sum())
                with _autocast(cfg, device):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                # Shift so token t predicts t+1; sum over answer tokens, divide by step total.
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                       b["labels"][:, 1:].reshape(-1),
                                       ignore_index=-100, reduction="sum") / n_target
                scaler.scale(loss).backward()
                step_loss += loss.item()
            if not math.isfinite(step_loss):
                raise RuntimeError(f"non-finite loss at step {step}; try precision='fp32'")
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(step_loss)
            step += 1
            if step == 1 or step % 10 == 0 or step == total_steps:
                log(f"  step {step:4d}/{total_steps}  epoch {epoch + 1}  loss {step_loss:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}")

    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    model.eval()
    return {"train_seconds": secs, "train_steps": step, "train_tokens": tokens_seen,
            "tokens_per_second": tokens_seen / secs, "losses": losses}


# ====================================================================== eval
def _eos_ids(tok) -> list[int]:
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot != tok.unk_token_id:
        ids.add(eot)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def generate(model, tok, conversations: list[list[dict]], cfg: RunConfig,
             max_new_tokens: int = 32, batch_size: int = 16) -> list[str]:
    device = next(p.device for p in model.parameters()).type
    model.config.use_cache = True
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(conversations), batch_size):
        texts = [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                 for c in conversations[i: i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with _autocast(cfg, device):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=_eos_ids(tok), pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    model.config.use_cache = False
    return [o.strip() for o in outs]


def evaluate(model, tok, items: list[dict], cfg: RunConfig, shots: dict | None = None) -> dict:
    convs = [prompt_messages(it, shots[it["qtype"]] if shots else None) for it in items]
    responses = generate(model, tok, convs, cfg)
    preds = []
    for it, resp in zip(items, responses):
        s = score(it, resp)
        preds.append({"id": it["id"], "qtype": it["qtype"], "source": it["source"],
                      "answer": it["answer"], "response": resp, **s})
    return {"accuracy": accuracy_table(preds), "predictions": preds}


def accuracy_table(preds: list[dict]) -> dict:
    out = {}
    for group in ["all"] + sorted({p["qtype"] for p in preds}):
        sub = preds if group == "all" else [p for p in preds if p["qtype"] == group]
        out[group] = {"n": len(sub),
                      "strict": sum(p["strict"] for p in sub) / len(sub),
                      "lenient": sum(p["lenient"] for p in sub) / len(sub)}
    return out


@torch.no_grad()
def measure_latency(model, tok, items: list[dict], cfg: RunConfig,
                    n_prompts: int = 20, new_tokens: int = 64, warmup: int = 2) -> dict:
    """Wall time to generate exactly `new_tokens` tokens at batch size 1."""
    device = next(p.device for p in model.parameters()).type
    model.config.use_cache = True
    times = []
    for k, it in enumerate(items[: n_prompts + warmup]):
        text = tok.apply_chat_template(prompt_messages(it), add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _autocast(cfg, device):
            model.generate(**enc, min_new_tokens=new_tokens, max_new_tokens=new_tokens,
                           do_sample=False, pad_token_id=tok.pad_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - t0)
    model.config.use_cache = False
    mean = sum(times) / len(times)
    sd = (sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
    return {"latency_ms_mean": 1000 * mean, "latency_ms_sd": 1000 * sd, "n_prompts": len(times)}


# ====================================================================== orchestration
def _gpu_gb(fn) -> float | None:
    return fn() / 2**30 if torch.cuda.is_available() else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg: RunConfig, data: dict[str, list[dict]], out_dir: str | Path,
        eval_split: str = "test", model_id: str = MODEL_ID, latency: bool = True,
        save_adapter: bool = True, overwrite: bool = False, log=print) -> dict:
    """Train (unless base) and evaluate one configuration. Returns the result dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.name if eval_split == "test" else f"{cfg.name}__{eval_split}"
    result_path = out_dir / f"{name}.json"
    if result_path.exists() and not overwrite:
        log(f"[skip] {name}: result exists")
        return json.loads(result_path.read_text())

    log(f"=== {name} ===")
    set_seed(cfg.seed)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    res = {"name": name, "config": asdict(cfg), "gpu": gpu, "eval_split": eval_split}
    model = None
    try:
        model, tok = load_model(cfg, model_id)
        res["model_load_mem_gb"] = _gpu_gb(torch.cuda.memory_allocated)
        model, preds = _run_body(model, tok, cfg, data, data[eval_split], res, out_dir, name,
                                 latency, save_adapter, log)
    except torch.cuda.OutOfMemoryError as e:
        # An OOM is a result too (e.g. full fine-tuning on a 16 GB T4).
        res["oom"] = True
        res["oom_message"] = str(e).splitlines()[0]
        res["peak_mem_gb"] = _gpu_gb(torch.cuda.max_memory_allocated)
        log(f"  OUT OF MEMORY on {gpu}: recorded as a result")
        result_path.write_text(json.dumps(res, indent=1))
        return res
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / f"{name}.predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)
    result_path.write_text(json.dumps(res, indent=1))
    return res


def _run_body(model, tok, cfg, data, items, res, out_dir, name, latency, save_adapter, log):
    if cfg.method == "base":
        res["trainable_params"], res["total_params"] = 0, count_params(model)[1]
        zero = evaluate(model, tok, items, cfg)
        few = evaluate(model, tok, items, cfg, shots=few_shot_pool(data["train"], cfg.n_shot))
        res["accuracy"], res["accuracy_fewshot"] = zero["accuracy"], few["accuracy"]
        preds = {"zero_shot": zero["predictions"], "few_shot": few["predictions"]}
        if latency:
            res["latency"] = measure_latency(model, tok, items, cfg)
        res["peak_mem_gb"] = _gpu_gb(torch.cuda.max_memory_allocated)
    else:
        model = prepare_for_training(model, cfg)
        res["trainable_params"], res["total_params"] = count_params(model)
        # Relative to the pre-trained model, as in the paper (LoRA's total includes the adapter).
        res["base_params"] = res["total_params"] - (res["trainable_params"] if cfg.method in ("lora", "qlora") else 0)
        res["trainable_pct"] = 100 * res["trainable_params"] / res["base_params"]
        log(f"  trainable {res['trainable_params']:,} / {res['total_params']:,} "
            f"({res['trainable_pct']:.4f}%)")
        train_items = data["train"][: cfg.n_train] if cfg.n_train else data["train"]
        res["train"] = train(model, tok, train_items, cfg, log)
        res["peak_train_mem_gb"] = _gpu_gb(torch.cuda.max_memory_allocated)
        res["peak_train_reserved_gb"] = _gpu_gb(torch.cuda.max_memory_reserved)

        ev = evaluate(model, tok, items, cfg)
        res["accuracy"], preds = ev["accuracy"], {"zero_shot": ev["predictions"]}

        if save_adapter and cfg.method in ("lora", "qlora"):
            adir = out_dir / "adapters" / name
            model.save_pretrained(adir)
            res["checkpoint_mb"] = sum(f.stat().st_size for f in adir.rglob("*") if f.is_file()) / 2**20
        else:  # full / top2: size of the trained weights if saved in bf16
            res["checkpoint_mb"] = res["trainable_params"] * 2 / 2**20
            res["checkpoint_mb_is_estimate"] = True

        if latency:
            # Paper claim: merging BA into W0 removes any inference latency.
            res["latency"] = measure_latency(model, tok, items, cfg)
            if cfg.method == "lora":
                model = model.merge_and_unload()
                res["latency_merged"] = measure_latency(model, tok, items, cfg)

    log(f"  accuracy (strict/lenient): " + "  ".join(
        f"{k} {v['strict']:.2f}/{v['lenient']:.2f}" for k, v in res["accuracy"].items()))
    return model, preds


# ====================================================================== analysis
def load_results(out_dir: str | Path, split: str = "test") -> list[dict]:
    rs = [json.loads(p.read_text()) for p in Path(out_dir).glob("*.json")
          if not p.name.endswith(".predictions.json")]
    return [r for r in rs if r.get("eval_split", "test") == split]


def summarize(out_dir: str | Path, split: str = "test"):
    """One row per configuration (seeds pooled): mean and sd of each metric."""
    import pandas as pd

    rows = []
    for r in load_results(out_dir, split):
        c = r["config"]
        is_lora = c["method"] in ("lora", "qlora")
        rank = c["rank"] if is_lora else "-"            # rank/targets only mean something for LoRA
        targets = "+".join(c["targets"]) if is_lora else "-"
        if r.get("oom"):
            rows.append({"config": r["name"], "method": c["method"], "prompt": "OOM",
                         "rank": rank, "targets": targets, "seed": c["seed"],
                         "peak_mem_gb": r.get("peak_mem_gb")})
            continue
        settings = [("zero_shot", r["accuracy"])]
        if "accuracy_fewshot" in r:
            settings.append(("few_shot", r["accuracy_fewshot"]))
        for prompt, acc in settings:
            row = {"config": r["name"].replace(f"_s{c['seed']}", "") if c["method"] != "base" else "base",
                   "method": c["method"], "prompt": prompt, "rank": rank,
                   "targets": targets, "seed": c["seed"],
                   "trainable_params": r["trainable_params"],
                   "trainable_pct": r.get("trainable_pct", 0.0),
                   "train_min": r.get("train", {}).get("train_seconds", float("nan")) / 60,
                   "tok_per_s": r.get("train", {}).get("tokens_per_second", float("nan")),
                   "peak_mem_gb": r.get("peak_train_mem_gb", r.get("peak_mem_gb")),
                   "checkpoint_mb": r.get("checkpoint_mb", float("nan")),
                   "latency_ms": r.get("latency", {}).get("latency_ms_mean", float("nan")),
                   "latency_merged_ms": r.get("latency_merged", {}).get("latency_ms_mean", float("nan"))}
            for g, v in acc.items():
                row[f"{g}_strict"], row[f"{g}_lenient"] = v["strict"], v["lenient"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["config", "method", "prompt", "rank", "targets"]
    num = [c for c in df.columns if c not in keys + ["seed"]]
    agg = df.groupby(keys)[num].agg(["mean", "std"])
    agg.insert(0, ("n_seeds", ""), df.groupby(keys).seed.nunique())
    return agg


def mcnemar(preds_a: list[dict], preds_b: list[dict], metric: str = "strict") -> dict:
    """Exact McNemar test on the same test items: does B differ from A?"""
    from scipy.stats import binomtest

    a = {p["id"]: p[metric] for p in preds_a}
    b = {p["id"]: p[metric] for p in preds_b}
    ids = sorted(set(a) & set(b))
    only_b = sum(b[i] and not a[i] for i in ids)
    only_a = sum(a[i] and not b[i] for i in ids)
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    return {"n_items": len(ids), "b_right_a_wrong": only_b, "a_right_b_wrong": only_a,
            "acc_a": sum(a[i] for i in ids) / len(ids), "acc_b": sum(b[i] for i in ids) / len(ids),
            "p_value": p}
