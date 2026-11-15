import pytest

from ardook_grants.data import load_any, load_tracker, map_fit

HEADER = ("Priority / Deadline Order,Applied?,Program,Type,Amount / Benefit,Deadline,"
          "Current Status (as last reported),Fit,Core Requirements,Why It Fits Ardook Data,"
          "Key Concern / Disqualifier,Recommended Action,Official Source,,")


def _write(tmp_path, rows):
    path = tmp_path / "tracker.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "raw, tier",
    [("Strong", "High"), ("Excellent", "High"), ("Moderate", "Moderate"), ("Weak–Moderate", "Moderate"),
     ("Weak-Moderate", "Moderate"), (" Weak ", "Low"), ("High", "High"), ("Excellent only if nonprofit", None), ("", None)],
)
def test_map_fit(raw, tier):
    assert map_fit(raw) == tier


def test_tracker_csv_skips_separator_rows_and_old_labels(tmp_path):
    path = _write(tmp_path, [
        "1,Yes,Old AI Prize,Prize,$5k,2026-09-01,Open,Strong,AI startups,fits,none,apply,http://x,,",
        "DATA ADDED ,,,,,,,,,,,,,Who found this,How Synthetic is the Data",
        "38,Not Yet,Care Fund,Grant,$1k,Not Listed,Open,Excellent,Mothers of disabled children,fits,none,apply,http://y,Timothy,None",
        "39,Not Yet,Arts Fund,Grant,$1k,11-12-26,Closed,Weak,Artists only,no,none,skip,http://z,Timothy,Partial",
    ])
    grants = load_tracker(path)
    assert [g["id"] for g in grants] == [1, 38, 39]
    assert [g["gold_fit"] for g in grants] == [None, "High", "Low"]  # row 1 was rated for "Ardook Data"
    assert grants[1]["synthetic"] == "None" and grants[2]["closed"]
    assert "Why It Fits" not in grants[1]["text"] and "fits" not in grants[1]["text"]
    assert grants[1]["text"].startswith("Program: Care Fund\n")
    assert load_any(path)[1]["program"] == "Care Fund"


def test_label_leakage_is_caught(tmp_path):
    path = _write(tmp_path, ["38,No,Fund,Grant,$1k,,Open,Weak,Ardook is not eligible,,,,,,"])
    with pytest.raises(ValueError, match="Ardook"):
        load_tracker(path)


def test_generic_csv_for_a_hidden_test_set(tmp_path):
    path = tmp_path / "hidden.csv"
    path.write_text("name,details,label\nCare Fund,Jobs for mothers,High\n", encoding="utf-8")
    g = load_any(path, text_columns=["details"], label_column="label")[0]
    assert g["text"] == "details: Jobs for mothers" and g["gold_fit"] == "High"
