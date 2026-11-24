import json
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules_review as rr  # noqa: E402

COL = rr.COL


@pytest.fixture
def files(tmp_path):
    # Start from an unreviewed copy so the tests don't depend on the real review state.
    j = tmp_path / "rules.json"
    items = json.loads((ROOT / "data/raw/rules_mc.json").read_text(encoding="utf-8"))
    review_keys = {"review_status", "reviewer", "review_note"}
    items = [{**{k: v for k, v in it.items() if k not in review_keys}, "verified": False} for it in items]
    j.write_text(json.dumps(items), encoding="utf-8")
    x = tmp_path / "review.xlsx"
    rr.export(j, x)
    return j, x


def edit(x, changes):
    """changes: {(row, header): value}"""
    wb = load_workbook(x)
    ws = wb[rr.SHEET]
    for (r, h), v in changes.items():
        ws.cell(row=r, column=COL[h]).value = v  # cell(value=None) would be a no-op
    wb.save(x)


def test_round_trip_without_edits_keeps_content(files):
    j, x = files
    before = json.loads(j.read_text(encoding="utf-8"))
    counts = rr.import_(j, x)
    after = json.loads(j.read_text(encoding="utf-8"))
    assert counts["Not checked"] == len(before) and counts["Delete"] == 0
    strip = lambda it: {k: v for k, v in it.items() if k != "review_status"}
    assert [strip(a) for a in after] == before


def test_export_layout(files):
    _, x = files
    wb = load_workbook(x)
    ws = wb[rr.SHEET]
    assert [c.value for c in ws[1]] == rr.HEADERS
    assert ws.cell(row=2, column=COL["Question"]).value.startswith("How many balls")
    assert ws.freeze_panes == "D2"
    assert "How to review" in wb.sheetnames
    # Time-sensitive flag shown
    flags = [ws.cell(row=r, column=COL["Time-sensitive"]).value for r in range(2, ws.max_row + 1)]
    want = sum(it.get("time_sensitive", False) for it in json.loads((ROOT / "data/raw/rules_mc.json")
                                                                      .read_text(encoding="utf-8")))
    assert flags.count("Yes") == want


def test_statuses_edits_and_new_row(files):
    j, x = files
    n = len(json.loads(j.read_text(encoding="utf-8")))
    new_row = n + 2
    edit(x, {
        (2, "Status"): "Correct", (2, "Reviewer"): "Avi", (2, "Notes"): "Rule 5.05(b)(1)",
        (3, "Status"): "Fixed", (3, "Correct answer"): "Three strikes",
        (4, "Status"): "Delete",
        (new_row, "Question"): "What does a 1-3 groundout mean?", (new_row, "Correct answer"): "Pitcher to first",
        (new_row, "Wrong answer 1"): "First to pitcher", (new_row, "Wrong answer 2"): "Catcher to first",
        (new_row, "Wrong answer 3"): "Pitcher to third", (new_row, "Status"): "Correct",
    })
    c = rr.import_(j, x)
    out = {it["id"]: it for it in json.loads(j.read_text(encoding="utf-8"))}
    assert c["Correct"] == 2 and c["Fixed"] == 1 and c["Delete"] == 1
    assert out["r001"]["verified"] and out["r001"]["reviewer"] == "Avi"
    assert out["r002"]["correct"] == "Three strikes" and out["r002"]["review_status"] == "Fixed"
    assert "r003" not in out
    new = [it for it in out.values() if it["question"].startswith("What does a 1-3")]
    assert len(new) == 1 and new[0]["id"] == f"r{n + 1:03d}" and new[0]["verified"]
    assert "group" in out["r031"]  # fields not shown in the sheet survive


def test_re_export_keeps_review_state(files, tmp_path):
    j, x = files
    edit(x, {(3, "Status"): "Fixed", (2, "Status"): "Correct"})
    rr.import_(j, x)
    x2 = tmp_path / "again.xlsx"
    rr.export(j, x2)
    ws = load_workbook(x2)[rr.SHEET]
    assert ws.cell(row=2, column=COL["Status"]).value == "Correct"
    assert ws.cell(row=3, column=COL["Status"]).value == "Fixed"


@pytest.mark.parametrize("changes,msg", [
    ({(2, "Status"): "Done"}, "not one of"),
    ({(2, "Wrong answer 2"): None}, "three wrong answers"),
    ({(2, "Wrong answer 1"): "four"}, "identical"),
    ({(2, "ID"): "r999"}, "unknown ID"),
    ({(2, "Question"): None, (2, "ID"): None, (2, "Correct answer"): None}, "missing from the sheet"),
])
def test_bad_sheet_rejected_and_json_untouched(files, changes, msg, capsys):
    j, x = files
    before = j.read_text(encoding="utf-8")
    edit(x, changes)
    with pytest.raises(SystemExit, match="NOT changed"):
        rr.import_(j, x)
    assert msg in capsys.readouterr().out
    assert j.read_text(encoding="utf-8") == before


def test_imported_file_still_builds(files, tmp_path):
    """After review, build_dataset.py accepts the JSON."""
    j, x = files
    ws_rows = len(json.loads(j.read_text(encoding="utf-8")))
    edit(x, {(r, "Status"): "Correct" for r in range(2, ws_rows + 2)})
    rr.import_(j, x)
    import build_dataset as bd
    import random
    items = bd.rules_items(j, random.Random(0), allow_unverified=False)
    assert len(items) == ws_rows
