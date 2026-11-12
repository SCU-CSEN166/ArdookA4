"""Load grants from the team spreadsheet or from a JSON file of {program, text} records."""

import json
from pathlib import Path

import pandas as pd

from .profile import REPO_ROOT

SPREADSHEET = REPO_ROOT / "Copy of Ardook_Data_All_Grant_Recommendations_UPDATED_v3.xlsx.xlsm"

# Columns that describe the grant itself. Fit, Why It Fits, Key Concern, Recommended Action
# and Current Status hold staff judgments, so they stay out of the model input (no label leakage).
INPUT_COLUMNS = ["Program", "Type", "Amount / Benefit", "Deadline", "Core Requirements"]


def grant_text(fields: dict) -> str:
    return "\n".join(f"{name}: {str(value).strip()}" for name, value in fields.items())


def load_spreadsheet(path: Path = SPREADSHEET) -> list[dict]:
    df = pd.read_excel(path, sheet_name="Grant Tracker", engine="openpyxl")
    grants = []
    for i, row in df.iterrows():
        status = str(row["Current Status (as last reported)"])
        grants.append(
            {
                "id": i + 1,
                "program": str(row["Program"]).strip(),
                "text": grant_text({col: row[col] for col in INPUT_COLUMNS}),
                "sheet_fit": str(row["Fit"]).strip(),
                "closed": status.lower().startswith("closed"),
            }
        )
    return grants


def load_json(path: Path) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {"id": i + 1, "program": r["program"], "text": r["text"], "sheet_fit": r.get("expected", ""), "closed": False}
        for i, r in enumerate(records)
    ]
