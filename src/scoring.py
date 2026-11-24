"""Answer extraction and scoring for the baseball QA test set.

Every item is scored two ways:

* strict  -- the response must *be* the answer (letter / name / number),
             which is what the prompt asks for. Penalises verbose output.
* lenient -- the answer only has to appear somewhere in the response.

The gap between the two separates "learned the answer format" from
"knows the answer", which matters because LoRA on 300 examples is
expected to mostly teach format.

Item fields used here (see build_dataset.py for the full schema):
    qtype            "mc" | "short" | "stats"
    answer           gold answer string (MC: the letter)
    aliases          short: accepted strings for strict matching
    lenient_aliases  short: extra strings accepted only by lenient scoring
    value, decimals  stats: exact (unrounded) value and display precision
"""

from __future__ import annotations

import re
import unicodedata

MC_LETTERS = "ABCD"

# Leading phrases models put before an answer ("Answer: B", "The answer is B").
_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is)?\s*[:\-]?\s*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?(?:\d+\.\d+|\.\d+|\d+)")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, drop a leading 'the'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def first_line(response: str) -> str:
    for line in response.strip().splitlines():
        line = line.strip().strip("*").strip()
        if line:
            return _PREFIX.sub("", line).strip()
    return ""


# ---------------------------------------------------------------- MC
def extract_mc_strict(response: str) -> str | None:
    """'B', 'B)', '(B)', 'B.', 'Answer: B', '**B**' -> 'B'. Anything longer fails."""
    line = first_line(response)
    m = re.fullmatch(r"\(?([A-Da-d])[\).:]?", line)
    return m.group(1).upper() if m else None


def extract_mc_lenient(response: str) -> str | None:
    strict = extract_mc_strict(response)
    if strict:
        return strict
    text = response.strip()
    patterns = [
        r"answer\s*(?:is|:)\s*\(?([A-D])\b",
        r"^\(?([A-D])[\).:]\s",          # "B) Four balls" / "B. Four balls"
        r"\(([A-D])\)",
        r"\*\*([A-D])[\).:]?\*\*",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------- stats
def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in _NUMBER.findall(text.replace(",", ""))]


def number_ok(pred: float, value: float, decimals: int) -> bool:
    """Correct if within one unit in the last displayed place of the exact value.

    This accepts both proper rounding and truncation (0.2867 -> 0.287 or 0.286)
    but rejects anything further off.
    """
    return abs(pred - value) <= 10 ** (-decimals) + 1e-9


# ---------------------------------------------------------------- scoring
def score(item: dict, response: str) -> dict:
    """Return {'strict': bool, 'lenient': bool, 'extracted': str | None}."""
    qtype = item["qtype"]

    if qtype == "mc":
        s = extract_mc_strict(response)
        l = extract_mc_lenient(response)
        return {"strict": s == item["answer"], "lenient": l == item["answer"],
                "extracted": l}

    if qtype == "short":
        strict_set = {normalize(a) for a in item["aliases"]}
        lenient_set = strict_set | {normalize(a) for a in item.get("lenient_aliases", [])}
        line = normalize(first_line(response))
        full = f" {normalize(response)} "
        strict = line in strict_set
        lenient = strict or any(f" {a} " in full for a in lenient_set if a)
        return {"strict": strict, "lenient": lenient, "extracted": line}

    if qtype == "stats":
        value, d = item["value"], item["decimals"]
        line_nums = parse_numbers(first_line(response))
        all_nums = parse_numbers(response)
        # Strict: the first line is a single number.
        strict = len(line_nums) == 1 and re.fullmatch(
            r"-?\d*\.?\d+", first_line(response).replace(",", "").strip()
        ) is not None and number_ok(line_nums[0], value, d)
        lenient = any(number_ok(x, value, d) for x in all_nums)
        return {"strict": strict, "lenient": lenient,
                "extracted": str(line_nums[0]) if line_nums else None}

    raise ValueError(f"unknown qtype {qtype!r}")
