"""Review the rules multiple-choice questions in a spreadsheet instead of JSON.

data/raw/rules_mc.json stays the source of truth; the spreadsheet is a
round-trip view of it.

    python src/rules_review.py export     # rules_mc.json -> data/raw/rules_review.xlsx
    (open the .xlsx in Excel or Google Sheets, review, save / download as .xlsx)
    python src/rules_review.py import     # rules_review.xlsx -> rules_mc.json (validated)

In the sheet, set Status for each row:
    Not checked   not reviewed yet                 -> verified: false
    Correct       question and answers are right   -> verified: true
    Fixed         you edited the text to fix it    -> verified: true
    Delete        drop the question                -> removed from the JSON
You may edit the question and answer text directly. To add a question, fill
in a new row and leave ID blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

JSON_PATH = Path("data/raw/rules_mc.json")
XLSX_PATH = Path("data/raw/rules_review.xlsx")
SHEET = "Questions"

STATUSES = ["Not checked", "Correct", "Fixed", "Delete"]
VERIFIED_STATUSES = {"Correct", "Fixed"}

# (header, json key or None, width, editable)
COLUMNS = [
    ("ID", "id", 7, False),
    ("Topic", "topic", 11, False),
    ("Time-sensitive", "time_sensitive", 10, False),
    ("Question", "question", 55, True),
    ("Correct answer", "correct", 32, True),
    ("Wrong answer 1", None, 26, True),
    ("Wrong answer 2", None, 26, True),
    ("Wrong answer 3", None, 26, True),
    ("Status", "review_status", 13, True),
    ("Reviewer", "reviewer", 12, True),
    ("Notes", "review_note", 34, True),
]
HEADERS = [c[0] for c in COLUMNS]
COL = {h: i + 1 for i, h in enumerate(HEADERS)}

FONT = "Arial"
EDIT_FILL = PatternFill("solid", fgColor="FFF8DC")      # light yellow: cells to fill in
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
TIME_FILL = PatternFill("solid", fgColor="FCE4D6")      # orange tint: time-sensitive
THIN = Side(style="thin", color="D9D9D9")


def status_of(item: dict) -> str:
    s = item.get("review_status")
    if s in STATUSES:
        return s
    return "Correct" if item.get("verified") else "Not checked"


# ------------------------------------------------------------------ export
def export(json_path: Path = JSON_PATH, xlsx_path: Path = XLSX_PATH) -> int:
    items = json.loads(json_path.read_text(encoding="utf-8"))
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET

    for c, (head, _, width, editable) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=head)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 30

    for r, it in enumerate(items, start=2):
        d = it["distractors"]
        values = [it["id"], it.get("topic", ""), "Yes" if it.get("time_sensitive") else "",
                  it["question"], it["correct"], d[0], d[1], d[2],
                  status_of(it), it.get("reviewer", ""), it.get("review_note", "")]
        for c, v in enumerate(values, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN)
        if it.get("time_sensitive"):
            ws.cell(row=r, column=COL["Time-sensitive"]).fill = TIME_FILL
            ws.cell(row=r, column=COL["Time-sensitive"]).font = Font(name=FONT, size=10, bold=True,
                                                                     color="C55A11")
        for h in ("Status", "Reviewer", "Notes"):
            ws.cell(row=r, column=COL[h]).fill = EDIT_FILL

    last = len(items) + 1
    extra = last + 50  # room for added questions
    status_col = ws.cell(row=1, column=COL["Status"]).column_letter

    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Status",
                        error="Choose Not checked, Correct, Fixed or Delete.")
    ws.add_data_validation(dv)
    dv.add(f"{status_col}2:{status_col}{extra}")

    # Colour each row by its status so progress is visible at a glance.
    whole = f"A2:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{extra}"
    for status, color in [("Correct", "E2EFDA"), ("Fixed", "DDEBF7"), ("Delete", "F8CBAD")]:
        ws.conditional_formatting.add(
            whole, FormulaRule(formula=[f'${status_col}2="{status}"'], fill=PatternFill("solid", bgColor=color)))

    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{last}"

    _write_guide(wb)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    return len(items)


def _write_guide(wb: Workbook) -> None:
    ws = wb.create_sheet("How to review")
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 95
    rows = [
        ("How to review the rules questions", None),
        ("", None),
        ("What to check", "Is the question clear? Is the correct answer actually correct under current MLB "
                         "rules? Is every wrong answer actually wrong? Could two options both be right?"),
        ("Time-sensitive", "Orange rows depend on recent rule changes (pitch clock, shift limits, roster "
                           "size...). Check these against the current MLB rulebook."),
        ("Cells to edit", "Status (required), Reviewer and Notes, in yellow. You may also edit the "
                          "Question and answer text directly to fix them."),
        ("Do not edit", "ID, Topic and Time-sensitive. They link each row back to rules_mc.json."),
        ("", None),
        ("Status values", None),
        ("Not checked", "Not reviewed yet. These are left out of the dataset until checked."),
        ("Correct", "Reviewed, right as written."),
        ("Fixed", "You edited the text to fix it; now right."),
        ("Delete", "Drop it (wrong, ambiguous, or not worth fixing)."),
        ("", None),
        ("Example row", None),
        ("Question", "How many balls does it take for a batter to be awarded a walk?"),
        ("Correct answer", "Four"),
        ("Wrong answers", "Three  |  Five  |  Six"),
        ("Status", "Correct"),
        ("Reviewer", "Avi"),
        ("Notes", "Rule 5.05(b)(1)"),
        ("", None),
        ("Splitting the work", "Filter the Topic column, or write your name in Reviewer for your rows. "
                               "To work at the same time, upload this file to Google Sheets, then "
                               "download it as .xlsx when everyone is done."),
        ("When done", "Save as rules_review.xlsx in data/raw/, then run:  "
                      "python src/rules_review.py import   (it checks every row and reports progress)"),
    ]
    for r, (a, b) in enumerate(rows, start=1):
        ca = ws.cell(row=r, column=1, value=a)
        ca.font = Font(name=FONT, bold=True, size=14 if r == 1 else 10)
        ca.alignment = Alignment(vertical="top")
        if b is not None:
            cb = ws.cell(row=r, column=2, value=b)
            cb.font = Font(name=FONT, size=10)
            cb.alignment = Alignment(vertical="top", wrap_text=True)
        elif a and r > 1:
            ca.font = Font(name=FONT, bold=True, size=11, color="1F3864")
    for r in range(15, 21):  # the example row, shaded like the edit cells where relevant
        ws.cell(row=r, column=2).fill = EDIT_FILL if r >= 18 else PatternFill()


# ------------------------------------------------------------------ import
def _text(v) -> str:
    return "" if v is None else str(v).strip()


def import_(json_path: Path = JSON_PATH, xlsx_path: Path = XLSX_PATH) -> Counter:
    items = json.loads(json_path.read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in items}
    wb = load_workbook(xlsx_path)
    if SHEET not in wb.sheetnames:
        sys.exit(f"{xlsx_path}: no sheet named {SHEET!r}")
    ws = wb[SHEET]
    header = [_text(c.value) for c in ws[1]][: len(HEADERS)]
    if header != HEADERS:
        sys.exit(f"{xlsx_path}: columns changed. Expected {HEADERS}, found {header}")

    errors, out, seen = [], [], set()
    next_num = max((int(i[1:]) for i in by_id if i[1:].isdigit()), default=0) + 1
    for r, row in enumerate(ws.iter_rows(min_row=2, max_col=len(HEADERS), values_only=True), start=2):
        v = dict(zip(HEADERS, (_text(x) for x in row)))
        if not any(v[h] for h in ("ID", "Question", "Correct answer")):
            continue  # blank row
        status = v["Status"] or "Not checked"
        if status not in STATUSES:
            errors.append(f"row {r}: Status {status!r} is not one of {STATUSES}")
            continue
        rid = v["ID"]
        if rid:
            if rid not in by_id:
                errors.append(f"row {r}: unknown ID {rid!r} (leave ID blank for a new question)")
                continue
            if rid in seen:
                errors.append(f"row {r}: ID {rid} appears twice")
                continue
            item = dict(by_id[rid])
        else:
            rid = f"r{next_num:03d}"
            next_num += 1
            item = {"id": rid, "topic": "new", "time_sensitive": False}
        seen.add(rid)
        if status == "Delete":
            continue

        wrong = [v[f"Wrong answer {k}"] for k in (1, 2, 3)]
        q, correct = v["Question"], v["Correct answer"]
        problems = []
        if not q:
            problems.append("question is empty")
        if not correct:
            problems.append("correct answer is empty")
        if any(not w for w in wrong):
            problems.append("needs three wrong answers")
        norm = [s.lower() for s in [correct] + wrong]
        if len(set(norm)) < 4 and all(norm):
            problems.append("two options are identical")
        if problems:
            errors.append(f"row {r} ({rid}): " + "; ".join(problems))
            continue

        item.update(question=q, correct=correct, distractors=wrong,
                    verified=status in VERIFIED_STATUSES, review_status=status)
        for key, h in (("reviewer", "Reviewer"), ("review_note", "Notes")):
            if v[h]:
                item[key] = v[h]
            else:
                item.pop(key, None)
        out.append(item)

    missing = sorted(set(by_id) - seen)
    if missing:
        errors.append(f"rows missing from the sheet (use Status 'Delete' to remove): {missing}")
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} problem(s); rules_mc.json NOT changed. Fix the sheet and run import again.")

    json_path.write_text(_dump(out), encoding="utf-8", newline="\n")
    counts = Counter(it["review_status"] for it in out)
    counts["Delete"] = len(seen) - len(out)
    return counts


def _dump(items: list[dict]) -> str:
    """One question per line, like the hand-written file, so git diffs stay readable."""
    return "[\n" + ",\n".join("  " + json.dumps(it, ensure_ascii=False) for it in items) + "\n]\n"


# ------------------------------------------------------------------ cli
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "import"])
    ap.add_argument("--json", type=Path, default=JSON_PATH)
    ap.add_argument("--xlsx", type=Path, default=XLSX_PATH)
    args = ap.parse_args(argv)
    if args.command == "export":
        n = export(args.json, args.xlsx)
        print(f"wrote {args.xlsx} ({n} questions)")
    else:
        c = import_(args.json, args.xlsx)
        kept = c["Correct"] + c["Fixed"] + c["Not checked"]
        print(f"updated {args.json}: {kept} questions "
              f"({c['Correct']} correct, {c['Fixed']} fixed, {c['Not checked']} not checked), "
              f"{c['Delete']} deleted")
        if c["Not checked"]:
            print(f"{c['Not checked']} still to review; build_dataset.py will refuse to build until they are.")
        else:
            print("All reviewed. Next: python src/build_dataset.py")


if __name__ == "__main__":
    main()
