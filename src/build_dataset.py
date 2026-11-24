"""Build the train / val / test splits for the baseball QA project.

Three question types, balanced across splits:

  stats  Computed from Lahman Batting/Pitching. The prompt gives the raw
         counts and the model must compute the rate stat (AVG, OBP, SLG,
         ERA, WHIP, K/9). Tests a learned calculation, not recall.
  short  Short factual recall generated from Lahman Teams/SeriesPost/
         AwardsPlayers (World Series winners, MVP, Cy Young, Rookie of the
         Year). Answers come straight from the database, so no hand checking.
  mc     Multiple choice, 4 options: hand-drafted rules questions
         (data/raw/rules_mc.json, must be verified by the team) plus history
         questions generated from Lahman with distractors from the same pool.

Leakage control: every item has a `group` (playerID for stats, the fact
key for history, the item id for rules). Splitting is done by group, so
no player or fact appears in two splits. A history fact is used in only one
form (short OR mc). Hand-drafted rules items are additionally checked for
near-duplicate wording across splits.

Usage:
  python src/build_dataset.py --lahman data/raw/lahman --rules data/raw/rules_mc.json
  python src/build_dataset.py ... --allow-unverified   # pipeline checks only

Outputs data/processed/{train,val,test}.jsonl and test.sha256 (the frozen
test set fingerprint; check it before every evaluation).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import MC_LETTERS, normalize  # noqa: E402

SEED = 534
SPLIT_SIZES = {  # per question type: (train, val, test)
    "stats": (100, 10, 34),
    "short": (100, 10, 33),
    "mc": (100, 10, 33),
}
YEAR_MIN = 1960          # SF / HBP recorded consistently; names more familiar
HISTORY_YEAR_MAX = 2023  # Gemma 3's training data ends ~mid-2024; later facts are unknowable to it
MIN_AB = 200
MIN_IPOUTS = 300         # 100 innings
FUZZY_THRESHOLD = 0.85   # difflib ratio for near-duplicate rules questions

INSTR = {
    "mc": "Answer with the letter only.",
    "short_team": "Answer with the team name only.",
    "short_player": "Answer with the player's full name only.",
    "stats": "Answer with the number only, rounded to {d} decimal places.",
}

TWO_WORD_NICKNAMES = {"red sox", "white sox", "blue jays"}


# ------------------------------------------------------------------ io
def read_csv(lahman: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    path = lahman / f"{name}.csv"
    if not path.exists():
        if required:
            sys.exit(f"missing {path}; download the SABR Lahman CSVs (see README)")
        return None
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def player_names(people: pd.DataFrame) -> dict[str, tuple[str, str]]:
    people = people.dropna(subset=["nameFirst", "nameLast"])
    return {r.playerID: (r.nameFirst, r.nameLast) for r in people.itertuples()}


def count(n: int, singular: str, plural: str | None = None) -> str:
    """count(1, 'triple') -> '1 triple'; count(2, 'triple') -> '2 triples'."""
    n = int(n)
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def fmt_innings(ipouts: int) -> str:
    whole, rem = divmod(int(ipouts), 3)
    return f"{whole}" if rem == 0 else f"{whole} {rem}/3"


# ------------------------------------------------------------------ stats
def stats_items(lahman: Path, names: dict, rng: random.Random) -> list[dict]:
    bat = read_csv(lahman, "Batting")
    pit = read_csv(lahman, "Pitching")
    count_cols_b = ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SF"]
    count_cols_p = ["IPouts", "ER", "H", "BB", "SO"]

    bat = bat[bat.lgID.isin(["AL", "NL"]) & (bat.yearID >= YEAR_MIN)]
    # A blank count (e.g. unrecorded HBP) would silently become 0 in the sum and
    # give a wrong answer, so drop player-seasons with any missing count.
    bat = bat.dropna(subset=count_cols_b)
    bat = bat.groupby(["playerID", "yearID"], as_index=False)[count_cols_b].sum()
    bat[count_cols_b] = bat[count_cols_b].astype(int)  # blanks load as float: "8.0 walks"
    bat = bat[bat.AB >= MIN_AB]
    pit = pit[pit.lgID.isin(["AL", "NL"]) & (pit.yearID >= YEAR_MIN)]
    pit = pit.dropna(subset=count_cols_p)
    pit = pit.groupby(["playerID", "yearID"], as_index=False)[count_cols_p].sum()
    pit[count_cols_p] = pit[count_cols_p].astype(int)
    pit = pit[pit.IPouts >= MIN_IPOUTS]

    def who(pid, year):
        first, last = names[pid]
        return f"In {year}, {first} {last}"

    def make_avg(r):
        v = r.H / r.AB
        q = (f"{who(r.playerID, r.yearID)} had {count(r.H, 'hit')} in {count(r.AB, 'at-bat')}. "
             f"What was his batting average?")
        return "AVG", q, v, 3

    def make_obp(r):
        denom = r.AB + r.BB + r.HBP + r.SF
        v = (r.H + r.BB + r.HBP) / denom
        q = (f"{who(r.playerID, r.yearID)} had {count(r.H, 'hit')}, {count(r.BB, 'walk')}, "
             f"{count(r.HBP, 'hit-by-pitch', 'hit-by-pitches')}, {count(r.AB, 'at-bat')} and "
             f"{count(r.SF, 'sacrifice fly', 'sacrifice flies')}. What was his on-base percentage?")
        return "OBP", q, v, 3

    def make_slg(r):
        singles = r.H - r["2B"] - r["3B"] - r.HR
        v = (singles + 2 * r["2B"] + 3 * r["3B"] + 4 * r.HR) / r.AB
        q = (f"{who(r.playerID, r.yearID)} had {count(singles, 'single')}, {count(r['2B'], 'double')}, "
             f"{count(r['3B'], 'triple')} and {count(r.HR, 'home run')} in {count(r.AB, 'at-bat')}. "
             f"What was his slugging percentage?")
        return "SLG", q, v, 3

    def make_era(r):
        v = 27 * r.ER / r.IPouts
        q = (f"{who(r.playerID, r.yearID)} allowed {count(r.ER, 'earned run')} in "
             f"{fmt_innings(r.IPouts)} innings pitched. What was his ERA?")
        return "ERA", q, v, 2

    def make_whip(r):
        v = 3 * (r.BB + r.H) / r.IPouts
        q = (f"{who(r.playerID, r.yearID)} allowed {count(r.H, 'hit')} and {count(r.BB, 'walk')} in "
             f"{fmt_innings(r.IPouts)} innings pitched. What was his WHIP?")
        return "WHIP", q, v, 2

    def make_k9(r):
        v = 27 * r.SO / r.IPouts
        q = (f"{who(r.playerID, r.yearID)} struck out {count(r.SO, 'batter')} in "
             f"{fmt_innings(r.IPouts)} innings pitched. What was his strikeouts per nine innings (K/9)?")
        return "K9", q, v, 2

    makers = [(bat, make_avg), (bat, make_obp), (bat, make_slg),
              (pit, make_era), (pit, make_whip), (pit, make_k9)]
    total = sum(SPLIT_SIZES["stats"])
    per_type = -(-total // len(makers))  # ceil
    # Oversample so the group split has room; extra items are dropped later.
    items, used = [], set()
    for df, make in makers:
        rows = df[df.playerID.isin(names)].sample(frac=1, random_state=rng.randrange(10**9))
        n = 0
        for _, r in rows.iterrows():
            if r.playerID in used:  # one stats item per player keeps groups small
                continue
            stat, q, v, d = make(r)
            used.add(r.playerID)
            items.append({
                "qtype": "stats", "source": "lahman_calc", "subtype": stat,
                "question": f"{q} {INSTR['stats'].format(d=d)}",
                "answer": f"{v:.{d}f}", "value": v, "decimals": d,
                "group": f"player:{r.playerID}",
            })
            n += 1
            if n >= per_type * 2:
                break
    return items


# ------------------------------------------------------------------ history facts
def team_aliases(full_name: str) -> list[str]:
    words = full_name.split()
    tail2 = " ".join(words[-2:]).lower()
    nick = " ".join(words[-2:]) if tail2 in TWO_WORD_NICKNAMES else words[-1]
    return [full_name, nick] if nick != full_name else [full_name]


def history_facts(lahman: Path, names: dict) -> list[dict]:
    """Each fact: question stem, answer, aliases, pool key for MC distractors."""
    facts = []
    teams = read_csv(lahman, "Teams", required=False)
    series = read_csv(lahman, "SeriesPost", required=False)
    awards = read_csv(lahman, "AwardsPlayers", required=False)

    if teams is not None and series is not None:
        tname = {(r.yearID, r.teamID): r.name for r in teams.itertuples()}
        # Lahman also labels the 1884-1890 championship series "WS"; the modern
        # World Series starts in 1903.
        ws = series[(series["round"] == "WS") & (series.yearID >= 1903)
                    & (series.yearID <= HISTORY_YEAR_MAX)]
        for r in ws.itertuples():
            name = tname.get((r.yearID, r.teamIDwinner))
            if not name:
                continue
            facts.append({
                "key": f"ws:{r.yearID}", "kind": "team", "pool": "ws",
                "stem": f"Which team won the {r.yearID} World Series?",
                "answer": name, "aliases": team_aliases(name), "lenient_aliases": [],
            })

    if awards is not None:
        # Exact (normalized) spellings only: the 2024 release added look-alikes
        # such as "TSN Rookie of the Year" that a substring match would catch.
        award_spellings = {
            "Most Valuable Player (MVP) award": {"most valuable player", "most valuable player award", "mvp"},
            "Cy Young Award": {"cy young award", "cy young"},
            "Rookie of the Year award": {"rookie of the year", "rookie of the year award"},
        }
        canonical = {}
        for aid in awards.awardID.dropna().unique():
            for text, spellings in award_spellings.items():
                if normalize(aid) in spellings:
                    canonical[aid] = text
        missing = set(award_spellings) - set(canonical.values())
        if missing:
            near = sorted(a for a in awards.awardID.dropna().unique()
                          if any(w in a.lower() for w in ("valuable", "young", "rookie")))
            sys.exit(f"AwardsPlayers has no award matching {sorted(missing)}. "
                     f"Similar awardIDs present: {near}. Add the right spelling to award_spellings.")
        award_text = canonical
        league_text = {"AL": "American League", "NL": "National League"}
        aw = awards[awards.awardID.isin(award_text) & awards.lgID.isin(league_text)
                    & (awards.yearID <= HISTORY_YEAR_MAX)]
        # Drop ties / shared awards: keep (award, year, league) with exactly one winner.
        counts = aw.groupby(["awardID", "yearID", "lgID"]).playerID.transform("count")
        aw = aw[counts == 1]
        for r in aw.itertuples():
            if r.playerID not in names:
                continue
            first, last = names[r.playerID]
            full = f"{first} {last}"
            facts.append({
                "key": f"award:{r.awardID}:{r.yearID}:{r.lgID}", "kind": "player",
                "pool": f"award:{r.awardID}:{r.lgID}",
                "stem": f"Who won the {r.yearID} {league_text[r.lgID]} {award_text[r.awardID]}?",
                "answer": full, "aliases": [full], "lenient_aliases": [last],
            })
    return facts


def mc_question(stem: str, correct: str, distractors: list[str], rng: random.Random) -> tuple[str, str, dict]:
    options = [correct] + distractors[:3]
    rng.shuffle(options)
    choices = dict(zip(MC_LETTERS, options))
    letter = next(k for k, v in choices.items() if v == correct)
    body = "\n".join(f"{k}) {v}" for k, v in choices.items())
    return f"{stem}\n{body}\n{INSTR['mc']}", letter, choices


def balance_mc_letters(items: list[dict], rng: random.Random) -> None:
    """Re-place each MC answer so correct letters are as even as possible in this split.

    Otherwise a model that always says one letter could beat chance by luck
    of the shuffle (with 33 test items the skew can be large).
    """
    mc = [it for it in items if it["qtype"] == "mc"]
    letters = [MC_LETTERS[i % 4] for i in range(len(mc))]
    rng.shuffle(letters)
    for it, target in zip(mc, letters):
        correct = it["choices"][it["answer"]]
        others = [v for k, v in it["choices"].items() if k != it["answer"]]
        rng.shuffle(others)
        others.insert(MC_LETTERS.index(target), correct)
        it["choices"] = dict(zip(MC_LETTERS, others))
        it["answer"] = target
        stem = it["question"].split("\n")[0]
        body = "\n".join(f"{k}) {v}" for k, v in it["choices"].items())
        it["question"] = f"{stem}\n{body}\n{INSTR['mc']}"


def history_items(facts: list[dict], n_mc: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Split facts into disjoint short-answer and MC sets."""
    facts = facts[:]
    rng.shuffle(facts)
    pools = defaultdict(set)
    for f in facts:
        pools[f["pool"]].add(f["answer"])

    mc, short = [], []
    for f in facts:
        others = sorted(pools[f["pool"]] - {f["answer"]})
        if len(mc) < n_mc and len(others) >= 3:
            q, letter, choices = mc_question(f["stem"], f["answer"], rng.sample(others, 3), rng)
            mc.append({"qtype": "mc", "source": "history", "subtype": f["pool"].split(":")[0],
                       "question": q, "answer": letter, "choices": choices,
                       "group": f["key"]})
        else:
            instr = INSTR["short_team"] if f["kind"] == "team" else INSTR["short_player"]
            short.append({"qtype": "short", "source": "history", "subtype": f["pool"].split(":")[0],
                          "question": f"{f['stem']} {instr}", "answer": f["answer"],
                          "aliases": f["aliases"], "lenient_aliases": f["lenient_aliases"],
                          "group": f["key"]})
    return mc, short


# ------------------------------------------------------------------ rules
def rules_items(path: Path, rng: random.Random, allow_unverified: bool) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    unverified = [r["id"] for r in raw if not r.get("verified")]
    if unverified and not allow_unverified:
        sys.exit(f"{len(unverified)} rules items not verified (e.g. {unverified[:5]}). "
                 "Verify them in rules_mc.json or pass --allow-unverified for a pipeline check.")
    items = []
    for r in raw:
        if len(r["distractors"]) != 3:
            sys.exit(f"{r['id']}: needs exactly 3 distractors")
        q, letter, choices = mc_question(r["question"], r["correct"], r["distractors"], rng)
        items.append({"qtype": "mc", "source": "rules", "subtype": r.get("topic", "rules"),
                      "question": q, "answer": letter, "choices": choices,
                      # Near-twin questions share a `group` so they stay in one split.
                      "group": f"rule:{r.get('group', r['id'])}", "verified": bool(r.get("verified"))})
    return items


# ------------------------------------------------------------------ split + checks
def group_split(items: list[dict], sizes: tuple[int, int, int], rng: random.Random) -> dict[str, list[dict]]:
    by_group = defaultdict(list)
    for it in items:
        by_group[it["group"]].append(it)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out = {"test": [], "val": [], "train": []}
    targets = dict(zip(["train", "val", "test"], sizes))
    # Fill test first so it is never short; then val; then train.
    for split in ["test", "val", "train"]:
        while groups and len(out[split]) < targets[split]:
            g = groups.pop()
            if len(out[split]) + len(by_group[g]) <= targets[split]:
                out[split].extend(by_group[g])
    for split, n in targets.items():
        if len(out[split]) < n:
            sys.exit(f"not enough items for {split}: {len(out[split])} < {n}")
    return out


def allocate(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Split n across strata in proportion to weights (largest remainder)."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def stratified_split(items: list[dict], sizes: tuple[int, int, int], stratum: str,
                     weights: dict[str, float], rng: random.Random) -> dict[str, list[dict]]:
    """Group split within each stratum, so every split has the same mix."""
    per_split = [allocate(n, weights) for n in sizes]
    out = {"train": [], "val": [], "test": []}
    for s in sorted(weights):
        sub = [it for it in items if it[stratum] == s]
        part = group_split(sub, tuple(p[s] for p in per_split), rng)
        for k in out:
            out[k].extend(part[k])
    return out


def check_overlap(splits: dict[str, list[dict]]) -> list[str]:
    problems = []
    seen_q, seen_g = {}, {}
    for split, items in splits.items():
        for it in items:
            nq = normalize(it["question"])
            if nq in seen_q and seen_q[nq] != split:
                problems.append(f"exact duplicate question across {seen_q[nq]}/{split}: {it['question'][:60]}")
            seen_q[nq] = split
            g = it["group"]
            if g in seen_g and seen_g[g] != split:
                problems.append(f"group {g} in both {seen_g[g]} and {split}")
            seen_g[g] = split

    # Near-duplicate wording for hand-written rules items (generated items are
    # templated, so their wording is intentionally similar; groups cover them).
    def stem(it):
        return normalize(it["question"].split("\n")[0])

    held_out = [it for s in ("val", "test") for it in splits[s] if it["source"] == "rules"]
    train = [it for it in splits["train"] if it["source"] == "rules"]
    for a in held_out:
        for b in train:
            r = difflib.SequenceMatcher(None, stem(a), stem(b)).ratio()
            if r >= FUZZY_THRESHOLD:
                problems.append(f"near-duplicate rules wording ({r:.2f}): {stem(a)[:50]} | {stem(b)[:50]}")
    return problems


def to_record(it: dict, idx: int, split: str) -> dict:
    rec = dict(it)
    rec["id"] = f"{split}-{it['qtype']}-{idx:03d}"
    rec["messages"] = [
        {"role": "user", "content": it["question"]},
        {"role": "assistant", "content": it["answer"]},
    ]
    return rec


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lahman", type=Path, default=Path("data/raw/lahman"))
    ap.add_argument("--rules", type=Path, default=Path("data/raw/rules_mc.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen test set")
    args = ap.parse_args(argv)

    if args.allow_unverified:
        # Pipeline-check builds never touch the real (frozen) split.
        args.out = args.out / "unverified"
    elif (args.out / "test.sha256").exists() and not args.force:
        sys.exit(f"{args.out}/test.sha256 exists: the test set is frozen. Use --force only if "
                 "no model has been trained or evaluated on the current split.")

    rng = random.Random(SEED)
    names = player_names(read_csv(args.lahman, "People"))

    stats = stats_items(args.lahman, names, rng)
    rules = rules_items(args.rules, rng, args.allow_unverified)
    n_mc_total = sum(SPLIT_SIZES["mc"])
    n_rules = min(len(rules), n_mc_total // 2)  # at most half of MC is rules
    n_mc_history = n_mc_total - n_rules + 20      # spare for the group split
    hist_mc, hist_short = history_items(history_facts(args.lahman, names), n_mc_history, rng)

    splits = {"train": [], "val": [], "test": []}
    parts = [
        stratified_split(stats, SPLIT_SIZES["stats"], "subtype",
                         {k: 1 for k in sorted({it["subtype"] for it in stats})}, rng),
        group_split(hist_short, SPLIT_SIZES["short"], rng),
        stratified_split(rules + hist_mc, SPLIT_SIZES["mc"], "source",
                         {"rules": n_rules, "history": n_mc_total - n_rules}, rng),
    ]
    for part in parts:
        for s in splits:
            splits[s].extend(part[s])
    for s in splits:
        balance_mc_letters(splits[s], rng)

    problems = check_overlap(splits)
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} overlap problems; not writing splits")

    args.out.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        records = []
        counters = Counter()
        for it in sorted(items, key=lambda x: (x["qtype"], x["group"])):
            counters[it["qtype"]] += 1
            records.append(to_record(it, counters[it["qtype"]], split))
        rng.shuffle(records)
        write_jsonl(args.out / f"{split}.jsonl", records)
        mix = Counter((r["qtype"], r["source"]) for r in records)
        letters = Counter(r["answer"] for r in records if r["qtype"] == "mc")
        print(f"{split:5s} n={len(records):3d}  {dict(sorted(mix.items()))}  mc letters={dict(sorted(letters.items()))}")

    if args.allow_unverified:
        print(f"WARNING: unverified rules items; wrote a pipeline-check split to {args.out}, "
              "not the real test set.")
        return
    digest = hashlib.sha256((args.out / "test.jsonl").read_bytes()).hexdigest()
    (args.out / "test.sha256").write_text(digest + "\n")
    print(f"test.jsonl sha256 {digest[:16]}... (frozen)")


if __name__ == "__main__":
    main()
