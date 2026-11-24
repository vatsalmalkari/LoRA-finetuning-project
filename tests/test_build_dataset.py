"""End-to-end test of build_dataset.py on a small synthetic Lahman fixture."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

import build_dataset as bd
from scoring import score

TEAMS = ["Boston Red Sox", "New York Yankees", "Chicago White Sox", "Toronto Blue Jays",
         "New York Mets", "Los Angeles Dodgers", "St. Louis Cardinals", "Detroit Tigers"]


def make_lahman(root: Path) -> None:
    rng = random.Random(0)
    n = 600
    pd.DataFrame({"playerID": [f"p{i:04d}" for i in range(n)],
                  "nameFirst": [f"First{i}" for i in range(n)],
                  "nameLast": [f"Last{i}" for i in range(n)]}).to_csv(root / "People.csv", index=False)

    bat = []
    for i in range(0, 300):
        ab = rng.randint(200, 650)
        h = rng.randint(40, ab // 3)
        d2, d3, hr = rng.randint(0, h // 4), rng.randint(0, 5), rng.randint(0, 40)
        hr = min(hr, h - d2 - d3)
        bat.append(dict(playerID=f"p{i:04d}", yearID=rng.randint(1960, 2025), stint=1, lgID="AL",
                        AB=ab, H=h, **{"2B": d2, "3B": d3}, HR=hr, BB=rng.randint(10, 100),
                        HBP=rng.randint(0, 15), SF=rng.randint(0, 10)))
    # A pre-1960 row and a low-AB row that must be filtered out.
    bat.append(dict(playerID="p0000", yearID=1950, stint=1, lgID="AL", AB=500, H=150,
                    **{"2B": 20, "3B": 2}, HR=10, BB=50, HBP=1, SF=0))
    pd.DataFrame(bat).to_csv(root / "Batting.csv", index=False)

    pit = []
    for i in range(300, 600):
        ip = rng.randint(300, 750)
        pit.append(dict(playerID=f"p{i:04d}", yearID=rng.randint(1960, 2025), stint=1, lgID="NL",
                        IPouts=ip, ER=rng.randint(20, 120), H=rng.randint(100, 250),
                        BB=rng.randint(20, 100), SO=rng.randint(50, 300)))
    pd.DataFrame(pit).to_csv(root / "Pitching.csv", index=False)

    years = list(range(1903, 2026))
    teams = [dict(yearID=y, lgID="AL", teamID=f"T{j}", name=TEAMS[j]) for y in years for j in range(len(TEAMS))]
    pd.DataFrame(teams).to_csv(root / "Teams.csv", index=False)
    pd.DataFrame([dict(yearID=y, round="WS", teamIDwinner=f"T{y % len(TEAMS)}", lgIDwinner="AL",
                       teamIDloser=f"T{(y + 1) % len(TEAMS)}", lgIDloser="NL") for y in years]
                 ).to_csv(root / "SeriesPost.csv", index=False)

    awards, pid = [], 0
    for y in range(1960, 2026):
        for aw in ["Most Valuable Player", "Cy Young Award", "Rookie of the Year"]:
            for lg in ["AL", "NL"]:
                awards.append(dict(playerID=f"p{pid % n:04d}", awardID=aw, yearID=y, lgID=lg))
                pid += 1
    # A tie that must be dropped.
    awards.append(dict(playerID="p0001", awardID="Most Valuable Player", yearID=1979, lgID="NL"))
    # A look-alike award (added in the 2024 release) that must NOT be used.
    for y in range(1960, 2026):
        # p0599 wins no real award in this fixture (real awards cycle p0000-p0395).
        awards.append(dict(playerID="p0599", awardID="TSN Rookie of the Year", yearID=y, lgID="AL"))
    pd.DataFrame(awards).to_csv(root / "AwardsPlayers.csv", index=False)


def make_rules(path: Path, n: int = 72, verified: bool = True) -> None:
    topics = ["count", "field", "scoring", "postseason", "equipment", "pitching"]
    rng = random.Random(1)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    def gibberish():  # random words, so no two questions look alike
        return " ".join("".join(rng.choices(alphabet, k=rng.randint(3, 9))) for _ in range(8)) + "?"

    rules = [{"id": f"r{i:03d}", "topic": topics[i % len(topics)], "verified": verified,
              "question": gibberish(),
              "correct": f"right{i}", "distractors": [f"w{i}a", f"w{i}b", f"w{i}c"]} for i in range(n)]
    path.write_text(json.dumps(rules))


@pytest.fixture
def env(tmp_path):
    lahman = tmp_path / "lahman"
    lahman.mkdir()
    make_lahman(lahman)
    rules = tmp_path / "rules.json"
    make_rules(rules)
    return lahman, rules, tmp_path / "out"


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_end_to_end(env, capsys):
    lahman, rules, out = env
    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    splits = {s: load(out / f"{s}.jsonl") for s in ["train", "val", "test"]}

    assert [len(splits[s]) for s in ["train", "val", "test"]] == [300, 30, 100]
    for s, (tr, va, te) in [("train", (100, 100, 100)), ("val", (10, 10, 10)), ("test", (34, 33, 33))]:
        counts = [sum(r["qtype"] == q for r in splits[s]) for q in ["stats", "short", "mc"]]
        assert counts == [tr, va, te], (s, counts)

    # No group (player / fact / rule) shared between splits.
    groups = {s: {r["group"] for r in rs} for s, rs in splits.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]

    for r in splits["train"] + splits["test"]:
        assert r["messages"][0] == {"role": "user", "content": r["question"]}
        assert r["messages"][1] == {"role": "assistant", "content": r["answer"]}
        # The gold answer must score as correct under both metrics.
        assert score(r, r["answer"]) | {"extracted": None} == {"strict": True, "lenient": True, "extracted": None}
        if r["qtype"] == "mc":
            assert r["answer"] in r["choices"] and len(set(r["choices"].values())) == 4
        assert "1950" not in r["question"]

    # Test set has both rules and history MC, and several stat types.
    test_mc_sources = {r["source"] for r in splits["test"] if r["qtype"] == "mc"}
    assert test_mc_sources == {"rules", "history"}
    assert len({r["subtype"] for r in splits["test"] if r["qtype"] == "stats"}) == 6

    # Correct MC letters are balanced in every split (max - min <= 1).
    for rs in splits.values():
        counts = [sum(r["qtype"] == "mc" and r["answer"] == L for r in rs) for L in "ABCD"]
        assert max(counts) - min(counts) <= 1, counts

    assert (out / "test.sha256").exists()


def test_stats_values_are_correct(env):
    lahman, rules, out = env
    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    bat = pd.read_csv(lahman / "Batting.csv").set_index("playerID")
    for r in load(out / "train.jsonl"):
        if r.get("subtype") == "AVG":
            row = bat.loc[r["group"].split(":")[1]]
            assert r["value"] == pytest.approx(row.H / row.AB)
            assert f"{row.H} hits in {row.AB} at-bats" in r["question"]


def test_test_set_is_frozen(env):
    lahman, rules, out = env
    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    with pytest.raises(SystemExit, match="frozen"):
        bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])


def test_unverified_rules_refused_then_go_to_side_folder(env, tmp_path):
    lahman, _, out = env
    rules = tmp_path / "unverified.json"
    make_rules(rules, verified=False)
    with pytest.raises(SystemExit, match="not verified"):
        bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out), "--allow-unverified"])
    assert (out / "unverified" / "test.jsonl").exists()
    assert not (out / "test.jsonl").exists()


def test_lookalike_awards_excluded(env):
    lahman, rules, out = env
    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    allrecs = [r for s in ["train", "val", "test"] for r in load(out / f"{s}.jsonl")]
    history = [r for r in allrecs if r["source"] == "history"]
    assert not any("Last599" in r["answer"] or "Last599" in r["question"] for r in history)
    assert not any("TSN" in r["question"] for r in allrecs)


def test_missing_award_spelling_fails_loudly(env):
    lahman, rules, out = env
    aw = pd.read_csv(lahman / "AwardsPlayers.csv")
    aw.loc[aw.awardID == "Cy Young Award", "awardID"] = "Cy Young Memorial Trophy"
    aw.to_csv(lahman / "AwardsPlayers.csv", index=False)
    with pytest.raises(SystemExit, match="Cy Young"):
        bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])


def test_team_aliases():
    assert bd.team_aliases("Boston Red Sox") == ["Boston Red Sox", "Red Sox"]
    assert bd.team_aliases("New York Mets") == ["New York Mets", "Mets"]


def test_count_pluralizes():
    assert bd.count(1, "triple") == "1 triple"
    assert bd.count(0, "triple") == "0 triples"
    assert bd.count(2.0, "sacrifice fly", "sacrifice flies") == "2 sacrifice flies"


def test_no_float_counts_or_pre_1903_series(env):
    lahman, rules, out = env
    # Blank HBP on one row loads the whole column as float.
    bat = pd.read_csv(lahman / "Batting.csv")
    bat.loc[5, "HBP"] = None
    bat.to_csv(lahman / "Batting.csv", index=False)
    sp = pd.read_csv(lahman / "SeriesPost.csv")
    sp = pd.concat([sp, pd.DataFrame([dict(yearID=1888, round="WS", teamIDwinner="T0",
                                           lgIDwinner="NL", teamIDloser="T1", lgIDloser="AA")])])
    sp.to_csv(lahman / "SeriesPost.csv", index=False)
    tm = pd.read_csv(lahman / "Teams.csv")
    pd.concat([tm, pd.DataFrame([dict(yearID=1888, lgID="NL", teamID="T0", name="Boston Red Sox")])]
              ).to_csv(lahman / "Teams.csv", index=False)

    bd.main(["--lahman", str(lahman), "--rules", str(rules), "--out", str(out)])
    recs = [r for s in ["train", "val", "test"] for r in load(out / f"{s}.jsonl")]
    import re
    assert not any(re.search(r"\d\.0 ", r["question"]) for r in recs if r["qtype"] == "stats")
    assert not any("1888" in r["question"] for r in recs)
    assert not any(r["group"] == "player:p0005" for r in recs)  # row with blank HBP dropped


def test_fmt_innings():
    assert bd.fmt_innings(631) == "210 1/3"
    assert bd.fmt_innings(630) == "210"
