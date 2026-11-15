"""The end-to-end pipeline: checklist answers -> checks -> eligibility rules -> Fit -> explanation.

`process()` is a pure function of a grant and the model's raw answers, so it can be tested with
canned answers and re-run on cached answers without loading any model. `run()` adds the model
calls and the cache. `write_outputs()` saves results.csv, shortlist.csv and trace.jsonl.
"""

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .backends import AnswerCache, ask_cached, direct_version
from .checklist import clean
from .data import map_fit
from .fit import entity_rating, explain, fit_label, theme_rating
from .rules import location_result, screen

GROUP_ORDER = ("applicants", "ai", "requirements", "theme", "limits")
ELIGIBILITY_GROUPS = ("applicants", "ai", "requirements")
# Added after the first runs. A run whose cache lacks these still works, just without their checks.
OPTIONAL_GROUPS = ("limits",)
# When "limits" was answered, its answers replace the older "applicants" ones it was added to correct.
LIMIT_OVERRIDES = {"max_years_in_operation": "max_company_age_years", "max_annual_revenue_usd": "max_annual_revenue_usd"}
FIT_ORDER = {"High": 0, "Moderate": 1, "Low": 2}


@dataclass
class Result:
    id: int
    program: str
    fit: str  # High / Moderate / Low
    eligibility: str  # Lead / Conditional lead / Not a lead
    why: str  # why this Fit
    eligibility_why: str  # why this eligibility
    check: str  # unknown facts to confirm before applying
    notes: str  # already applied, closed, needs review, ...
    theme: str  # good / partial / bad / unclear
    theme_areas: str
    entity: str  # good / partial / bad
    location: str  # ok / conditional / excluded / not stated / ignored
    evidence: str  # quotes with ✓ (found in the grant text) or ✗
    needs_review: bool
    deadline: str = ""
    gold_fit: str | None = None
    sheet_fit: str = ""
    gold_theme: str | None = None
    gold_entity: str | None = None
    synthetic: str = ""
    seconds: float = 0.0
    trace: dict = field(default_factory=dict, repr=False)


def _grant_fields(grant: dict) -> dict:
    return {
        "deadline": grant.get("deadline", ""),
        "gold_fit": grant.get("gold_fit"),
        "sheet_fit": grant.get("sheet_fit", ""),
        "gold_theme": grant.get("gold_theme"),
        "gold_entity": grant.get("gold_entity"),
        "synthetic": grant.get("synthetic", ""),
    }


def process(grant: dict, raw: dict, groups: dict, profile: dict, use_geography: bool = True,
            today: date | None = None) -> Result:
    """Turn one grant's raw model answers ({group name: answer}) into a Result."""
    facts, reports = {}, {}
    for name in GROUP_ORDER:
        if name in OPTIONAL_GROUPS and raw.get(name) is None:
            reports[name] = {"error": None, "problems": [], "quotes": [], "unverified": [], "skipped": True}
            continue
        group_facts, reports[name] = clean(groups[name], raw.get(name), grant["text"])
        if name == "limits" and not reports[name]["error"]:
            for new, old in LIMIT_OVERRIDES.items():
                group_facts[old] = group_facts.pop(new)
        facts.update(group_facts)

    failed = [n for n in (*ELIGIBILITY_GROUPS, *OPTIONAL_GROUPS) if reports[n]["error"]]
    screen_input = dict(facts)
    if failed:
        screen_input["_error"] = f"extraction failed ({', '.join(failed)})"
    v = screen(grant["program"], screen_input, profile, today, use_geography)

    unverified = [f for n in (*ELIGIBILITY_GROUPS, *OPTIONAL_GROUPS) for f in reports[n]["unverified"]]
    if unverified:
        v.conditions.append(f"check by hand: no matching quote for {', '.join(unverified)}")
        v.update_status()

    theme, areas = theme_rating(facts.get("focus_areas") or [], profile)
    entity = entity_rating(v)
    fit = fit_label(theme or "partial", entity)  # unknown theme: assume related, and flag for review

    notes = list(v.notes)
    if grant.get("closed"):
        notes.append("closed")
    if failed or reports["theme"]["error"]:
        notes.append("extraction failed")
    if unverified or reports["theme"]["unverified"]:
        notes.append("unverified answer")
    if theme is None:
        notes.append("focus unclear")
    needs_review = bool(failed or reports["theme"]["error"] or unverified or reports["theme"]["unverified"] or theme is None)
    if needs_review:
        notes.append("needs review")

    evidence = [f"{'✓' if q['found'] else '✗'} \"{q['quote']}\"" for n in GROUP_ORDER for q in reports[n]["quotes"]]
    evidence = list(dict.fromkeys(evidence))
    return Result(
        id=grant["id"],
        program=grant["program"],
        fit=fit,
        eligibility=v.status,
        why=explain(fit, theme, areas, entity, v),
        eligibility_why="; ".join(v.reasons or v.conditions or v.matches or ["no restrictions found"]),
        check="; ".join(v.conditions),
        notes=", ".join(dict.fromkeys(notes)),
        theme=theme or "unclear",
        theme_areas=", ".join(a for r in ("good", "partial", "bad") for a in areas.get(r, [])),
        entity=entity,
        location=location_result(v) if use_geography else "ignored",
        evidence=" | ".join(evidence),
        needs_review=needs_review,
        trace={"facts": facts, "reports": reports, "verdict": asdict(v)},
        **_grant_fields(grant),
    )


def process_direct(grant: dict, raw: dict | None) -> Result:
    """Baseline: the model rates Fit in one answer, with no checklist or rules."""
    raw = raw if isinstance(raw, dict) else {"_error": "no answer"}
    fit = map_fit(raw.get("fit")) if "_error" not in raw else None
    return Result(
        id=grant["id"], program=grant["program"], fit=fit or "Moderate", eligibility="", why=str(raw.get("reason", "")),
        eligibility_why="", check="", notes="" if fit else "invalid answer, needs review", theme="", theme_areas="",
        entity="", location="", evidence="", needs_review=fit is None, trace={"raw": raw}, **_grant_fields(grant),
    )


def run(grants: list[dict], groups: dict, profile: dict, backend, cache: AnswerCache, *, rules_only: bool = False,
        use_geography: bool = True, direct: bool = False, profile_text: str = "", verbose: bool = True) -> list[Result]:
    results = []
    for g in grants:
        if direct:
            hit = cache.get("direct", direct_version(profile_text), g["text"])
            if hit is None and not rules_only:
                start = time.perf_counter()
                answer = backend.direct(g["text"], profile_text)
                cache.put("direct", direct_version(profile_text), g["text"], answer, time.perf_counter() - start)
                cache.save()
                hit = cache.get("direct", direct_version(profile_text), g["text"])
            if hit is None:
                print(f"skip (not cached): {g['program']}")
                continue
            r = process_direct(g, hit["answer"])
            r.seconds = hit["seconds"]
        else:
            raw, seconds, missing = {}, 0.0, False
            for name in GROUP_ORDER:
                raw[name], s = ask_cached(backend, cache, groups[name], g["text"], rules_only)
                seconds += s
                missing |= raw[name] is None and name not in OPTIONAL_GROUPS
            if missing:
                print(f"skip (not cached): {g['program']}")
                continue
            if not rules_only:
                cache.save()  # after every grant, so an interrupted run keeps its work
            r = process(g, raw, groups, profile, use_geography)
            r.seconds = round(seconds, 1)
        results.append(r)
        if verbose:
            print(f"[{r.id:>3}] {r.fit:<8} {r.eligibility:<16} {r.program[:48]:<48} {r.seconds:>5}s  {r.why[:90]}")
    return results


def prefetch(grants: list[dict], groups: dict, backend, cache: AnswerCache, workers: int, *,
             direct: bool = False, profile_text: str = "") -> int:
    """Ask the model all uncached questions in parallel (for API backends, whose server batches requests).

    Answers go into the cache; run() then works from the cache as usual. Returns how many were asked.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if direct:
        todo = [(g, None) for g in grants if cache.get("direct", direct_version(profile_text), g["text"]) is None]
    else:
        todo = [(g, groups[n]) for g in grants for n in GROUP_ORDER
                if cache.get(n, groups[n].version, g["text"]) is None]

    def ask(grant, group):
        start = time.perf_counter()
        answer = backend.direct(grant["text"], profile_text) if group is None else backend.ask(grant["text"], group)
        return grant, group, answer, time.perf_counter() - start

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, g, grp) for g, grp in todo]
        try:
            for future in as_completed(futures):
                grant, group, answer, seconds = future.result()
                name, version = ("direct", direct_version(profile_text)) if group is None else (group.name, group.version)
                cache.put(name, version, grant["text"], answer, seconds)
                done += 1
                if done % 20 == 0 or done == len(todo):
                    cache.save()
                    print(f"  asked {done}/{len(todo)}")
        except BaseException:
            for f in futures:
                f.cancel()
            cache.save()
            raise
    return done


def parse_deadline(s: str) -> date | None:
    """2026-09-16, 11-12-26, 1/15/2027 -> date; 'Rolling', 'Not Listed' -> None."""
    s = (s or "").strip()
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})", s):
            year = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
            return date(year, int(m[1]), int(m[2]))
    except ValueError:
        return None
    return None


def shortlist(results: list[Result]) -> list[Result]:
    """Leads Ardook could act on: not ruled out, not closed; best Fit first, then soonest deadline."""
    keep = [r for r in results if r.eligibility != "Not a lead" and "closed" not in r.notes]
    return sorted(keep, key=lambda r: (FIT_ORDER.get(r.fit, 3), parse_deadline(r.deadline) or date.max, r.id))


def write_outputs(results: list[Result], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in asdict(r).items() if k != "trace"} for r in results]
    row_of = {id(r): row for r, row in zip(results, rows)}
    for name, data in (("results.csv", rows), ("shortlist.csv", [row_of[id(r)] for r in shortlist(results)])):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
            writer.writeheader()
            writer.writerows(data)
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"id": r.id, "program": r.program, **r.trace}, ensure_ascii=False) + "\n")
