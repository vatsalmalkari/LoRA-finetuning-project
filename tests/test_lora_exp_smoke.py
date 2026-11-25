"""End-to-end smoke test of src/lora_exp.py on a tiny randomly initialised Gemma 3.

No model download: the model is built from a small Gemma3TextConfig and the
tokenizer is a small BPE trained on the dataset text, with Gemma's chat
template. Exercises every method, masking, generation, scoring, latency,
saving, OOM handling and the analysis helpers. Skipped when torch is absent;
QLoRA and fp16 also need CUDA (+ bitsandbytes).

    python -m pytest tests/test_lora_exp_smoke.py -q
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import lora_exp as lx  # noqa: E402

GEMMA_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)
CUDA = torch.cuda.is_available()


def tiny_items(tmp: Path) -> dict:
    """Small split built by build_dataset.py from the synthetic Lahman fixture."""
    import build_dataset as bd
    import test_build_dataset as fixture

    (tmp / "lahman").mkdir()
    fixture.make_lahman(tmp / "lahman")
    fixture.make_rules(tmp / "rules.json")
    bd.main(["--lahman", str(tmp / "lahman"), "--rules", str(tmp / "rules.json"), "--out", str(tmp / "data")])
    data = {s: lx.load_jsonl(tmp / "data" / f"{s}.jsonl") for s in ["train", "val", "test"]}
    # Keep every question type in each small split.
    def take(items, n):
        out = []
        for q in ["mc", "short", "stats"]:
            out += [it for it in items if it["qtype"] == q][:n]
        return out
    return {"train": take(data["train"], 8), "val": take(data["val"], 3), "test": take(data["test"], 4)}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

    tmp = tmp_path_factory.mktemp("smoke")
    data = tiny_items(tmp)

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    text = [m["content"] for s in data.values() for it in s for m in it["messages"]] + ["user\n", "model\n"]
    bpe.train_from_iterator(text, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, pad_token="<pad>", eos_token="<eos>",
                                  bos_token="<bos>", unk_token="<unk>")
    tok.chat_template = GEMMA_TEMPLATE

    cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=32,
                           max_position_embeddings=1024, sliding_window=256,
                           pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                           eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    model_dir = tmp / "tiny-gemma3"
    Gemma3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return {"data": data, "model_id": str(model_dir), "out": tmp / "results", "tok": tok}


def _run(setup, cfg, **kw):
    return lx.run(cfg, setup["data"], setup["out"], model_id=setup["model_id"], log=lambda *_: None, **kw)


def test_answer_masking(setup):
    tok, item = setup["tok"], setup["data"]["train"][0]
    enc = lx.encode_example(tok, item, 512)
    kept = [t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100]
    assert tok.decode(kept) == item["answer"] + "<end_of_turn>\n"
    assert len(enc["labels"]) == len(enc["input_ids"])


def test_base(setup):
    r = _run(setup, lx.RunConfig("base", precision="fp32", n_shot=2))
    assert set(r["accuracy"]) == {"all", "mc", "short", "stats"}
    assert "accuracy_fewshot" in r and r["latency"]["n_prompts"] > 0
    preds = json.loads((setup["out"] / "base.predictions.json").read_text())
    assert len(preds["few_shot"]) == len(setup["data"]["test"])


@pytest.mark.parametrize("method,lr", [("lora", 5e-3), ("full", 1e-3), ("top2", 1e-3)])
def test_trained_methods(setup, method, lr):
    cfg = lx.RunConfig(method, lr=lr, epochs=3, batch_size=8, micro_batch=4, precision="fp32")
    r = _run(setup, cfg)
    losses = r["train"]["losses"]
    assert all(map(lambda x: x == x, losses))                    # no NaN
    assert losses[-1] < losses[0], losses                          # it learns
    assert 0 < r["trainable_params"] <= r["total_params"]
    if method == "full":
        assert r["trainable_params"] == r["total_params"]
    if method == "lora":
        assert r["trainable_pct"] < 50
        assert r["base_params"] + r["trainable_params"] == r["total_params"]
        assert (setup["out"] / "adapters" / r["name"] / "adapter_config.json").exists()
        assert "latency_merged" in r
    if CUDA:
        assert r["peak_train_mem_gb"] > 0


def test_lora_param_count_matches_formula(setup):
    """r * (d_in + d_out) per adapted matrix per layer (paper's count, without square-matrix assumption)."""
    model, _ = lx.load_model(lx.RunConfig("lora", precision="fp32"), setup["model_id"])
    model = lx.prepare_for_training(model, lx.RunConfig("lora", rank=8, targets=("q_proj", "v_proj")))
    c = model.config
    d, hd = c.hidden_size, c.head_dim
    want = c.num_hidden_layers * 8 * ((d + c.num_attention_heads * hd) + (d + c.num_key_value_heads * hd))
    assert lx.count_params(model)[0] == want


@pytest.mark.skipif(not CUDA, reason="QLoRA needs CUDA")
def test_qlora(setup):
    pytest.importorskip("bitsandbytes")
    r = _run(setup, lx.RunConfig("qlora", lr=5e-3, precision="fp32"))
    assert r["train"]["losses"][-1] < r["train"]["losses"][0]
    # Logical parameter count matches the unquantised model.
    base = json.loads((setup["out"] / "base.json").read_text())
    assert r["total_params"] - r["trainable_params"] == base["total_params"]


@pytest.mark.skipif(not CUDA, reason="fp16 autocast needs CUDA")
def test_lora_fp16_path(setup):
    r = _run(setup, lx.RunConfig("lora", lr=5e-3, precision="fp16", tag="_fp16"), latency=False)
    assert r["train"]["losses"][-1] < r["train"]["losses"][0]


def test_memorizes_and_answers(setup):
    """Train until 6 items are memorised, then generate: proves the training format,
    chat template at generation time, decoding and scoring all agree."""
    tr = setup["data"]["train"]
    items = [next(it for it in tr if it["qtype"] == q) for q in ["mc", "short", "stats"]]
    mem = {"train": items, "val": items, "test": items}
    r = lx.run(lx.RunConfig("full", lr=3e-3, epochs=150, batch_size=3, micro_batch=3, precision="fp32",
                            warmup_ratio=0.0, tag="_memorize"),
               mem, setup["out"] / "memorize", model_id=setup["model_id"], log=lambda *_: None, latency=False)
    assert r["accuracy"]["all"]["strict"] == 1.0, r["accuracy"]


def test_oom_is_recorded(setup, monkeypatch):
    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20 GiB")
    monkeypatch.setattr(lx, "train", boom)
    r = _run(setup, lx.RunConfig("full", precision="fp32", tag="_oom"))
    assert r["oom"] and "out of memory" in r["oom_message"]


def test_skip_existing(setup):
    calls = []
    r = lx.run(lx.RunConfig("base", precision="fp32", n_shot=2), setup["data"], setup["out"],
               model_id=setup["model_id"], log=calls.append)
    assert any("skip" in c for c in calls) and r["name"] == "base"


def test_val_split_runs_separately(setup):
    r = _run(setup, lx.RunConfig("lora", lr=5e-3, precision="fp32"), eval_split="val",
             latency=False, save_adapter=False)
    assert r["name"].endswith("__val") and r["accuracy"]["all"]["n"] == len(setup["data"]["val"])


def test_summarize_and_mcnemar(setup):
    # Needs results from the tests above (module-scoped results folder).
    table = lx.summarize(setup["out"])
    methods = set(table.index.get_level_values("method"))
    assert {"base", "lora", "full", "top2"} <= methods
    a = json.loads((setup["out"] / "base.predictions.json").read_text())["few_shot"]
    lora_name = next(r["name"] for r in lx.load_results(setup["out"])
                     if r["config"]["method"] == "lora" and not r["config"]["tag"])
    b = json.loads((setup["out"] / f"{lora_name}.predictions.json").read_text())["zero_shot"]
    t = lx.mcnemar(a, b)
    assert t["n_items"] == len(setup["data"]["test"]) and 0 <= t["p_value"] <= 1


def test_adapter_loads_for_subspace_analysis(setup, tmp_path):
    """analysis.py reads lora_A from the adapters lora_exp saves (real PEFT key names)."""
    import analysis as an
    r8 = lx.run(lx.RunConfig("lora", lr=5e-3, precision="fp32", epochs=1), setup["data"], tmp_path,
                model_id=setup["model_id"], log=lambda *_: None, latency=False)
    r16 = lx.run(lx.RunConfig("lora", rank=16, alpha=32, lr=5e-3, precision="fp32", epochs=1), setup["data"],
                 tmp_path, model_id=setup["model_id"], log=lambda *_: None, latency=False)
    a8, a16 = tmp_path / "adapters" / r8["name"], tmp_path / "adapters" / r16["name"]
    A = an.load_lora_A(a8)
    n_layers = 4                                   # tiny model in `setup`
    assert set(A) == {(l, m) for l in range(n_layers) for m in ("q_proj", "v_proj")}
    assert A[(0, "q_proj")].shape == (8, 128)      # (rank, hidden_size)
    sim = an.subspace_similarity(a8, a16)
    P = sim["phi"]["q_proj"]
    assert P.shape == (8, 16) and P.min() >= -1e-9 and P.max() <= 1 + 1e-9
