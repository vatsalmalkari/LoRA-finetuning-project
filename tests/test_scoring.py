import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from scoring import normalize, number_ok, score

MC = {"qtype": "mc", "answer": "B"}
SHORT = {"qtype": "short", "answer": "New York Mets",
         "aliases": ["New York Mets", "Mets"], "lenient_aliases": []}
PLAYER = {"qtype": "short", "answer": "Barry Larkin",
          "aliases": ["Barry Larkin"], "lenient_aliases": ["Larkin"]}
STATS = {"qtype": "stats", "answer": "0.287", "value": 0.28667, "decimals": 3}


@pytest.mark.parametrize("resp", ["B", "B)", "(B)", "b.", "Answer: B", "**B**", "The answer is B"])
def test_mc_strict_accepts_bare_letter(resp):
    assert score(MC, resp)["strict"]


@pytest.mark.parametrize("resp", [
    "B) Four balls",
    "The correct answer is (B) because four balls make a walk.",
    "**B.** Four",
])
def test_mc_lenient_only(resp):
    r = score(MC, resp)
    assert not r["strict"] and r["lenient"]


def test_mc_article_a_is_not_an_answer():
    r = score(MC, "A walk happens after four balls.")
    assert not r["strict"] and not r["lenient"]


def test_mc_wrong_letter():
    assert not score(MC, "C")["lenient"]


def test_normalize():
    assert normalize("  The Boston Red Sox! ") == "boston red sox"
    assert normalize("José Altuve") == "jose altuve"
    assert normalize("St. Louis Cardinals") == "st louis cardinals"


@pytest.mark.parametrize("resp", ["New York Mets", "Mets", "the Mets.", "Answer: New York Mets"])
def test_short_strict(resp):
    assert score(SHORT, resp)["strict"]


def test_short_lenient_in_sentence():
    r = score(SHORT, "The 1986 World Series was won by the New York Mets.")
    assert not r["strict"] and r["lenient"]


def test_short_last_name_is_lenient_only():
    r = score(PLAYER, "Larkin")
    assert not r["strict"] and r["lenient"]


def test_short_no_partial_word_match():
    # "mets" must not match inside another word
    assert not score(SHORT, "Metsville Tigers")["lenient"]


@pytest.mark.parametrize("resp", ["0.287", ".287", "0.286", "0.2867", "Answer: 0.287"])
def test_stats_strict(resp):
    assert score(STATS, resp)["strict"]


def test_stats_lenient_in_sentence():
    r = score(STATS, "His batting average was 0.287.")
    assert not r["strict"] and r["lenient"]


@pytest.mark.parametrize("resp", ["0.284", "0.29", "287"])
def test_stats_wrong(resp):
    assert not score(STATS, resp)["lenient"]


def test_number_ok_two_decimals():
    assert number_ok(3.46, 3.4567, 2)
    assert not number_ok(3.48, 3.4567, 2)
