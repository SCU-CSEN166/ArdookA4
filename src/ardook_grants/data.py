"""Load grants from the team spreadsheet, a CSV export of it, the synthetic set, or JSON.

Every loader returns a list of dicts with at least: id, program, text, sheet_fit, closed.
The tracker and synthetic loaders also add gold labels (gold_fit, and for synthetic data
gold_theme / gold_entity) so the pipeline can be scored.
"""

import json
import re
from pathlib import Path

import pandas as pd

from .profile import REPO_ROOT

SPREADSHEET = REPO_ROOT / "Copy of Ardook_Data_All_Grant_Recommendations_UPDATED_v3.xlsx.xlsm"

# Columns that describe the grant itself. Fit, Why It Fits, Key Concern, Recommended Action
# and Current Status hold staff judgments, so they stay out of the model input (no label leakage).
INPUT_COLUMNS = ["Program", "Type", "Amount / Benefit", "Deadline", "Core Requirements"]

# Rows 1-37 of the Grant Tracker were rated for "Ardook Data" (a separate AI venture, see the
# profile's Open items); Timothy rated rows 38+ for Ardook itself. Only those are Fit labels.
FIRST_ARDOOK_ROW = 38

# Timothy's Fit ratings -> the three tiers (same mapping as notebooks/real_grant_eval.ipynb).
FIT_MAP = {
    "excellent": "High",
    "strong": "High",
    "moderate": "Moderate",
    "weak-moderate": "Moderate",
    "weak": "Low",
}
TIERS = ("Low", "Moderate", "High")

# Marker the sheet uses (in an unnamed column) for how much of a row Timothy made up.
SYNTHETIC_MARKER = "How Synthetic is the Data"


def grant_text(fields: dict) -> str:
    return "\n".join(f"{name}: {str(value).strip()}" for name, value in fields.items())


def _clean(value) -> str:
    """Collapse whitespace; pandas NaN and blanks become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def map_fit(raw) -> str | None:
    """'Strong' -> 'High'. Conditional or mixed ratings ('Excellent only if nonprofit') -> None."""
    key = _clean(raw).lower().replace("–", "-").replace("—", "-").replace(" - ", "-")
    if key in FIT_MAP:
        return FIT_MAP[key]
    if key.title() in TIERS:  # already a tier (e.g. a hidden test set labeled High / Moderate / Low)
        return key.title()
    return None


def check_no_leakage(grants: list[dict]) -> None:
    """The model input must not contain the labelers' reasoning about Ardook."""
    leaked = [g["program"] for g in grants if re.search(r"ardook", g["text"], re.IGNORECASE)]
    if leaked:
        raise ValueError(f"Grant text mentions Ardook (label leakage?): {leaked[:5]}")


def load_spreadsheet(path: Path = SPREADSHEET) -> list[dict]:
    """Week 6 baseline loader for the .xlsm snapshot. Kept as-is so its cached extractions stay valid."""
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


def load_tracker(path: Path, label_rows: tuple[int, int | None] = (FIRST_ARDOOK_ROW, None)) -> list[dict]:
    """The Grant Tracker as .csv (Google Sheets export) or .xlsx/.xlsm.

    Skips separator rows (no number in the first column, e.g. "DATA ADDED"). Only rows in `label_rows`
    (first, last; None = to the end) keep their Fit as a gold label: rows 1-37 were rated for Ardook Data,
    rows 38+ for Ardook, so each organization is scored only on its own labels.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(path, sheet_name="Grant Tracker", engine="openpyxl", dtype=str).fillna("")
    order = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    synthetic_col = next((c for c in df.columns if df[c].astype(str).str.strip().eq(SYNTHETIC_MARKER).any()), None)

    grants = []
    for idx in df.index[order.notna()]:
        row = df.loc[idx]
        number = int(order[idx])
        fields = {col: _clean(row[col]) for col in INPUT_COLUMNS if _clean(row[col])}
        grants.append(
            {
                "id": number,
                "row": number,
                "program": _clean(row["Program"]),
                "text": grant_text(fields),
                "deadline": _clean(row.get("Deadline", "")),
                "sheet_fit": _clean(row.get("Fit", "")),  # absent in the TA's hidden test
                "gold_fit": map_fit(row.get("Fit", "")) if label_rows[0] <= number <= (label_rows[1] or number) else None,
                "closed": _clean(row.get("Current Status (as last reported)", "")).lower().startswith("closed"),
                "synthetic": _clean(row[synthetic_col]) if synthetic_col else "",
            }
        )
    check_no_leakage(grants)
    return grants


def load_synthetic(path: Path, split: str | None = None) -> list[dict]:
    """data/synthetic_grants.csv (Giap's generator): text, label, theme, geo, entity, split."""
    df = pd.read_csv(path)
    if split:
        df = df[df["split"] == split]
    grants = []
    for i, row in enumerate(df.itertuples(index=False)):
        text = _clean(row.text)
        grants.append(
            {
                "id": i + 1,
                "program": text[:70],  # the generator's templates don't put the name in one place
                "text": text,
                "deadline": "",
                "sheet_fit": row.label,
                "gold_fit": row.label,
                "gold_theme": row.theme,
                "gold_entity": row.entity,
                "closed": False,
            }
        )
    return grants


def load_generic_csv(path: Path, text_columns: list[str] | None = None, label_column: str | None = None,
                     program_column: str | None = None) -> list[dict]:
    """Any other CSV (e.g. the TA's hidden test set). Text comes from `text_columns`, or a 'text' column."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not text_columns:
        if "text" not in df.columns:
            raise ValueError(f"{path} has no 'text' column; pass --text-columns (columns: {list(df.columns)})")
        text_columns = ["text"]
    program_column = program_column or next((c for c in ("program", "Program", "title", "name") if c in df.columns), None)
    grants = []
    for i, row in enumerate(df.to_dict("records")):
        fields = {c: _clean(row[c]) for c in text_columns if _clean(row[c])}
        text = fields["text"] if text_columns == ["text"] else grant_text(fields)
        grants.append(
            {
                "id": i + 1,
                "program": _clean(row[program_column]) if program_column else f"grant {i + 1}",
                "text": text,
                "deadline": _clean(row.get("Deadline", row.get("deadline", ""))),
                "sheet_fit": _clean(row[label_column]) if label_column else "",
                "gold_fit": map_fit(row[label_column]) if label_column else None,
                "closed": False,
            }
        )
    return grants


def load_json(path: Path) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {"id": i + 1, "program": r["program"], "text": r["text"], "sheet_fit": r.get("expected", ""), "closed": False}
        for i, r in enumerate(records)
    ]


def load_any(path: Path, split: str | None = None, text_columns: list[str] | None = None,
             label_column: str | None = None, label_rows: tuple[int, int | None] = (FIRST_ARDOOK_ROW, None)) -> list[dict]:
    """Pick the loader from the file: JSON, the Grant Tracker, the synthetic set, or a generic CSV."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return load_tracker(path, label_rows)
    columns = pd.read_csv(path, nrows=0).columns
    if text_columns or label_column:
        return load_generic_csv(path, text_columns, label_column)
    if "Priority / Deadline Order" in columns or "Grant ID" in columns:
        return load_tracker(path, label_rows)
    if {"text", "label", "theme", "entity"} <= set(columns):
        return load_synthetic(path, split)
    return load_generic_csv(path)
