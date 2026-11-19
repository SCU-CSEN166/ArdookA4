"""Models that answer the checklist, and a cache of their answers.

  NuExtractBackend  numind/NuExtract-2.0-2B, the extraction model from Week 6 (laptop, ~4 GB)
  LLMBackend        any chat model on Hugging Face; default Qwen/Qwen3-4B (~8 GB in bf16, fits an 18 GB Mac)
  APIBackend        any OpenAI-compatible API; default Qwen3-32B on our Akash server (the pipeline's default)

Both are loaded lazily, so cached runs (--rules-only) and tests never import torch.
The cache is a JSON file per backend+model, keyed by question group, its version and the grant text,
so changing a question or a grant re-asks only what changed.
"""

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .checklist import Group, parse_json_answer
from .extract import MODEL_NAME as NUEXTRACT_MODEL
from .profile import REPO_ROOT

DEFAULT_LLM = "Qwen/Qwen3-4B"
AKASHML_URL = "https://api.akashml.com/v1"
# The project's model: Qwen3-32B on our own Akash GPU (deploy/akash/akash_llm.py writes its URL and key to .env).
# It replaced NuExtract-2B as the default in Week 9: NuExtract invented "nonprofit only" and misread year limits.
DEFAULT_API_MODEL = "Qwen/Qwen3-32B"
OWN_SERVER_URL_ENV, OWN_SERVER_KEY_ENV = "AKASH_LLM_URL", "AKASH_LLM_KEY"


def default_api_endpoint() -> tuple[str, str]:
    """(base URL, env var holding the key): our Akash server if .env has its URL, else AkashML."""
    url = read_secret(OWN_SERVER_URL_ENV)
    return (url, OWN_SERVER_KEY_ENV) if url else (AKASHML_URL, "AKASHML_API_KEY")

LLM_INSTRUCTIONS = """Fill in the JSON template using only the grant listing. Don't guess or use outside knowledge.

How to read the template:
- ["a", "b", ...]: pick exactly one of these values, or null if the listing doesn't say.
- [["a", "b", ...]]: pick every value that applies, as a list; [] if none apply or the listing doesn't say.
- ["string"]: a list of short texts taken from the listing; [] if none.
- "verbatim-string" or ["verbatim-string"]: copy text from the listing exactly, word for word.
- "number" / "integer": a number, or null.
Fields ending in _evidence: copy the exact phrases from the listing that support your other answers.
Return only the filled JSON object, nothing else."""

DIRECT_INSTRUCTIONS = """You help a social enterprise find grants. Read its profile and one grant listing, then rate how well the grant fits it.

Judge two things only: whether what the grant funds matches the organization's mission, and whether an organization like this one can apply (legal form, size, stage, who the applicants must be). Ignore the grant's location.

Answer with one rating:
- High: the grant funds work at the core of the mission, and the organization can apply.
- Moderate: related but not core, or it depends on an unknown fact.
- Low: off-mission, or the organization clearly can't apply.

Return only JSON: {"fit": "High" | "Moderate" | "Low", "reason": "<one short sentence>"}"""


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


class NuExtractBackend:
    name = "nuextract"

    def __init__(self, model_name: str = NUEXTRACT_MODEL):
        self.model_name = model_name
        self._model = None

    def ask(self, text: str, group: Group) -> dict:
        if self._model is None:
            from .extract import NuExtractor

            print(f"Loading {self.model_name} ...")
            self._model = NuExtractor(self.model_name)
        return self._model.run(text, group.template, group.examples)


class LLMBackend:
    name = "llm"

    def __init__(self, model_name: str = DEFAULT_LLM):
        self.model_name = model_name
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .extract import pick_device, pick_dtype

            print(f"Loading {self.model_name} ...")
            device = pick_device()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=pick_dtype(device))
            model.to(device).eval()
            self._loaded = (torch, tokenizer, model, device)
        return self._loaded

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        torch, tokenizer, model, device = self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # enable_thinking=False turns off Qwen3's reasoning mode; other chat templates ignore it.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def ask(self, text: str, group: Group) -> dict:
        return parse_json_answer(self.generate(LLM_INSTRUCTIONS, group_prompt(group, text)))

    def direct(self, text: str, profile_summary: str) -> dict:
        user = f"Organization profile:\n{profile_summary}\n\nGrant listing:\n{text}"
        return parse_json_answer(self.generate(DIRECT_INSTRUCTIONS, user, max_new_tokens=150))


class APIError(RuntimeError):
    """The API call itself failed (no credits, bad key, network). Stops the run; nothing is cached."""


def read_secret(name: str) -> str | None:
    """An environment variable, or a NAME=value line in the repo's .env file (git-ignored)."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("'\"")
    return None


class APIBackend:
    """Any OpenAI-compatible chat API. Default: AkashML (Akash Network's hosted open models).

    The key comes from $AKASHML_API_KEY (or --api-key-env NAME), else from a NAME=... line in .env.
    It is only ever sent in the Authorization header, never printed or cached.
    """

    name = "api"

    def __init__(self, model_name: str = DEFAULT_API_MODEL, base_url: str = AKASHML_URL,
                 api_key_env: str = "AKASHML_API_KEY", max_retries: int = 6, no_thinking: bool = False):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        # vLLM extension: turns off Qwen3's reasoning mode, like enable_thinking=False in LLMBackend.
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else {}

    def _key(self) -> str:
        key = read_secret(self.api_key_env)
        if not key:
            raise APIError(f"No API key: set ${self.api_key_env} or add a {self.api_key_env}=... line to .env")
        return key

    def generate(self, system: str, user: str, max_new_tokens: int = 512, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        body = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            **({"seed": seed} if seed is not None else {}),
            "max_completion_tokens": max_new_tokens,
            **self.extra,
        }).encode()
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json",
                   "User-Agent": "ardook-grants-pipeline/1.0"}  # Cloudflare blocks Python's default client name
        for attempt in range(self.max_retries):
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
                return (data["choices"][0]["message"].get("content") or "").strip()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 402:
                    raise APIError("Out of API credits (402). Top up the account, then re-run: cached answers are kept.")
                if e.code in (401, 403):
                    raise APIError(f"API key rejected ({e.code}). Check ${self.api_key_env}.")
                if e.code == 429 or e.code >= 500:
                    wait = float(e.headers.get("Retry-After") or 2 ** attempt)
                    print(f"  API {e.code}, retrying in {wait:.0f}s ...")
                    time.sleep(min(wait, 60))
                    continue
                raise APIError(f"API error {e.code}: {detail}")
            # OSError covers URLError, dropped connections and "Can't assign requested address" after the laptop sleeps.
            except (OSError, http.client.HTTPException) as e:
                print(f"  network error ({e}), retrying ...")
                time.sleep(2 ** attempt)
        raise APIError(f"API still failing after {self.max_retries} tries")

    def ask(self, text: str, group: Group) -> dict:
        return parse_json_answer(self.generate(LLM_INSTRUCTIONS, group_prompt(group, text)))

    def direct(self, text: str, profile_summary: str) -> dict:
        user = f"Organization profile:\n{profile_summary}\n\nGrant listing:\n{text}"
        return parse_json_answer(self.generate(DIRECT_INSTRUCTIONS, user, max_new_tokens=150))


def group_prompt(group: Group, text: str) -> str:
    parts = [f"Template:\n{json.dumps(group.template, indent=2, ensure_ascii=False)}", "Examples:"]
    for ex in group.examples:
        parts.append(f"Listing:\n{ex['input']}\nAnswer:\n{ex['output']}")
    parts.append(f"Now this listing.\nListing:\n{text}\nAnswer:")
    return "\n\n".join(parts)


def direct_version(profile_text: str = "") -> str:
    """Version of a direct answer: the instructions plus the profile the model was shown."""
    return hashlib.sha256((DIRECT_INSTRUCTIONS + "\n" + profile_text).encode()).hexdigest()[:12]


class AnswerCache:
    """{key: {"answer": <raw model answer>, "seconds": float}} in one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @staticmethod
    def key(group_name: str, version: str, text: str) -> str:
        return f"{group_name}|{version}|{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    def get(self, group_name: str, version: str, text: str) -> dict | None:
        return self.data.get(self.key(group_name, version, text))

    def put(self, group_name: str, version: str, text: str, answer: dict, seconds: float) -> None:
        self.data[self.key(group_name, version, text)] = {"answer": answer, "seconds": round(seconds, 2)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def ask_cached(backend, cache: AnswerCache, group: Group, text: str, rules_only: bool) -> tuple[dict | None, float]:
    """(raw answer, seconds). None when --rules-only and the answer isn't cached."""
    hit = cache.get(group.name, group.version, text)
    if hit:
        return hit["answer"], hit["seconds"]
    if rules_only:
        return None, 0.0
    start = time.perf_counter()
    answer = backend.ask(text, group)
    seconds = time.perf_counter() - start
    cache.put(group.name, group.version, text, answer, seconds)
    return answer, seconds
