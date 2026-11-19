"""Retrieval-augmented Fit judge: rate a grant with the most similar already-rated grants as worked examples.

The checklist (checklist.py + rules.py) is good at hard eligibility facts but rates Fit from coarse theme
categories. The direct judge (backends.DIRECT_INSTRUCTIONS) judges Fit better but has no reference for how
the organization's staff rate grants. This judge adds that reference:

  1. pool      grants with a known rating and a one-line reason: the sponsor's rated rows (Fit plus their
               "Why It Fits" / "Key Concern" notes) and the synthetic pool split (label plus its deciding fact)
  2. retrieve  the k pool grants most similar to the new one (MiniLM sentence embeddings, cosine similarity),
               never the grant itself (leave-one-out, so dev scores aren't inflated)
  3. judge     the model rates the grant given the profile, the lead rules and those k examples
  4. vote      optionally n samples at temperature > 0; the majority wins and its share is the confidence

Answers are cached like the checklist's, keyed by the prompt, so re-scoring costs nothing.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import AnswerCache
from .checklist import parse_json_answer
from .data import _clean, grant_text, map_fit
from .profile import REPO_ROOT, profile_summary

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TIERS = ("Low", "Moderate", "High")

JUDGE_INSTRUCTIONS = """You screen grant listings for one organization and rate how good a lead each grant is.

Rate with one of:
- High: what the grant funds is at the core of the organization's mission, and nothing stated in the listing rules it out.
- Moderate: related but not core, or the decision depends on a fact the profile leaves unknown.
- Low: off-mission, or a requirement stated in the listing clearly rules the organization out (applicant type,
  a location it doesn't work in, a revenue, age or size minimum a new venture can't meet).

Only use requirements the listing actually states. An unknown fact is a reason for Moderate, not Low.
The examples are similar grants the organization's staff already rated; follow how they rate, but judge this grant on its own text.

Return only JSON: {"fit": "High" | "Moderate" | "Low", "reason": "<one short sentence naming the deciding fact>"}"""


@dataclass
class Example:
    text: str
    label: str
    reason: str
    source: str


def judge_profile_text(profile_path: Path) -> str:
    """The profile's descriptive sections plus its lead rules (the direct baseline leaves the rules out)."""
    text = Path(profile_path).read_text(encoding="utf-8")
    rules = re.search(r"^## Lead rules\n(.*?)(?=^#+ )", text, re.DOTALL | re.MULTILINE)
    extra = f"\n\n## Lead rules\n{rules.group(1).strip()}" if rules else ""
    return re.sub(r"\s*\[S\] rows? [\d, –-]+", "", profile_summary(profile_path) + extra)


def sponsor_examples(path: Path, rows: tuple[int, int]) -> list[Example]:
    """The sponsor's rated rows, with their own reasoning as the example's reason."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    order = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    out = []
    for idx in df.index[order.notna()]:
        row, number = df.loc[idx], int(order[idx])
        label = map_fit(row.get("Fit", ""))
        if not (rows[0] <= number <= rows[1]) or not label:
            continue
        fields = {c: _clean(row[c]) for c in ("Program", "Type", "Amount / Benefit", "Deadline", "Core Requirements")
                  if _clean(row[c])}
        why = _clean(next((row[c] for c in row.index if c.startswith("Why It Fits")), ""))
        concern = _clean(row.get("Key Concern / Disqualifier", ""))
        reason = "; ".join(p for p in (why, f"concern: {concern}" if concern else "") if p)
        out.append(Example(grant_text(fields), label, reason[:300], "sponsor"))
    return out


def synthetic_examples(path: Path, split: str = "pool") -> list[Example]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["split"] == split]
    return [Example(r.text, r.label, f"deciding fact: {r.decisive}", "synthetic") for r in df.itertuples(index=False)]


class Retriever:
    """Cosine similarity over mean-pooled MiniLM embeddings (loaded lazily; runs on CPU in seconds)."""

    def __init__(self, examples: list[Example]):
        self.examples = examples
        self._model = None
        self.matrix = self.embed([e.text for e in examples]) if examples else np.zeros((0, 384))

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._model is None:
            self._tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
            self._model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        vecs = []
        for i in range(0, len(texts), 64):
            batch = self._tok(texts[i : i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            v = (out * mask).sum(1) / mask.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=1).numpy())
        return np.vstack(vecs)

    def top_k(self, text: str, k: int) -> list[Example]:
        if k <= 0 or not self.examples:
            return []
        sims = self.matrix @ self.embed([text])[0]
        order = [i for i in np.argsort(-sims) if self.examples[i].text.strip() != text.strip()]  # leave-one-out
        return [self.examples[i] for i in order[:k]]


def judge_prompt(text: str, profile_text: str, examples: list[Example], facts: str = "") -> str:
    parts = [f"Organization profile:\n{profile_text}"]
    if examples:
        shots = "\n\n".join(f"Listing:\n{e.text}\nRating: {e.label} ({e.reason})" for e in examples)
        parts.append(f"Similar grants already rated:\n\n{shots}")
    parts.append(f"Grant listing to rate:\n{text}")
    if facts:
        parts.append("An automatic requirement check of this listing found the following. It can be wrong, "
                     f"so confirm each point against the listing text:\n{facts}")
    return "\n\n".join(parts)


def checklist_facts(row: dict) -> str:
    """One checklist result row (results.csv) as hints for the judge."""
    lines = []
    if row.get("eligibility") == "Not a lead":
        lines.append(f"- Looks ruled out: {row.get('eligibility_why', '')}")
    elif row.get("check"):
        lines.append(f"- Depends on: {row['check']}")
    else:
        lines.append(f"- No hard restriction found ({row.get('eligibility_why', '')})")
    if row.get("theme_areas"):
        lines.append(f"- Funds: {row['theme_areas']}")
    return "\n".join(lines)


def vote(answers: list[dict]) -> tuple[str | None, float, str]:
    """(majority tier, its share of valid answers, the reason of the first answer that gave it)."""
    tiers = [map_fit(a.get("fit")) for a in answers if isinstance(a, dict) and "_error" not in a]
    tiers = [t for t in tiers if t]
    if not tiers:
        return None, 0.0, ""
    counts = Counter(tiers)
    best = max(counts.values())
    # Ties go to the middle tier (Moderate), then to the lower one: the cautious reading of a split vote.
    tied = sorted((t for t in counts if counts[t] == best), key=lambda t: (t != "Moderate", TIERS.index(t)))
    winner = tied[0]
    reason = next(str(a.get("reason", "")) for a in answers
                  if isinstance(a, dict) and map_fit(a.get("fit")) == winner)
    return winner, best / len(tiers), reason


class Judge:
    def __init__(self, backend, cache: AnswerCache, profile_text: str, retriever: Retriever | None, k: int = 4,
                 votes: int = 1, temperature: float = 0.6, thinking: bool = False):
        self.backend, self.cache, self.profile_text = backend, cache, profile_text
        self.retriever, self.k, self.votes, self.thinking = retriever, k, votes, thinking
        self.temperature = temperature if votes > 1 else 0.0

    def requests(self, text: str, facts: str = "") -> list[tuple[str, str, int]]:
        """(cache version, prompt, sample index) for each answer this grant needs."""
        shots = self.retriever.top_k(text, self.k) if self.retriever else []
        user = judge_prompt(text, self.profile_text, shots, facts)
        base = hashlib.sha256((JUDGE_INSTRUCTIONS + user).encode()).hexdigest()[:12]
        mode = "-think" if self.thinking else ""
        return [(f"{base}-t{self.temperature}-s{i}{mode}", user, i) for i in range(self.votes)]

    def ask(self, version: str, user: str, sample: int) -> dict:
        # Thinking mode reasons step by step before the JSON (parse_json_answer drops the <think> block).
        raw = self.backend.generate(JUDGE_INSTRUCTIONS, user, max_new_tokens=2500 if self.thinking else 150,
                                    temperature=self.temperature, seed=sample if self.temperature else None)
        return parse_json_answer(raw)


def default_pool(profile_name: str, sponsor_csv: Path, sponsor_rows: tuple[int, int] | None,
                 synthetic_csv: Path | None) -> list[Example]:
    pool = []
    if sponsor_rows and Path(sponsor_csv).exists():
        pool += sponsor_examples(sponsor_csv, sponsor_rows)
    if synthetic_csv and Path(synthetic_csv).exists():
        pool += synthetic_examples(synthetic_csv)
    return pool


SPONSOR_CSV = REPO_ROOT / "data" / "real_grants.csv"
SYNTHETIC_ARDOOK_DATA = REPO_ROOT / "data" / "synthetic_ardook_data.csv"
