"""Helpers for reproduce.ipynb and the WAVE job scripts: pick a model for the GPU, serve it with vLLM,
run the final system on the shared datasets and score it.

The final system (Week 9) is two Qwen3-32B steps over the Ardook Data profile:
  1. checklist   src/run_pipeline.py  -> eligibility (can Ardook Data apply at all?)
  2. RAG judge   src/run_judge.py --k 16 -> Fit (High / Moderate / Low), shown the 16 most similar rated grants
  Final Fit = Low if the checklist rules the grant out, else the judge's Fit.  Relevant = Fit is not Low.

Nothing here imports torch or vLLM: the model runs in its own process (its own Python env) and the
pipeline talks to it over vLLM's OpenAI-compatible API, exactly like the team's Akash server.
"""

import csv
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reproduce" / "bundle"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

PROFILE = "ardook-data"
JUDGE_K = 16
VLLM_VERSION = "0.18.0"  # the version the team's Akash server ran (deploy/akash/vllm-qwen3-32b.yaml)
VLLM_IMAGE = f"docker://vllm/vllm-openai:v{VLLM_VERSION}"
KEY_ENV = "REPRO_LLM_KEY"

# The labeled sets anyone can rerun. (The TA's hidden test isn't shared; see the notebook's optional step.)
DATASETS = [
    {"key": "real", "title": "Real grants the sponsor rated for Ardook Data", "input": "data/real_grants.csv",
     "split": None, "labeled_only": True},
    {"key": "synthetic_test", "title": "Synthetic Ardook Data grants, test split", "input": "data/synthetic_ardook_data.csv",
     "split": "test", "labeled_only": False},
    {"key": "synthetic_alt_test", "title": "Synthetic, held-out wording (no overlap with the example pool)",
     "input": "data/synthetic_ardook_data_alt.csv", "split": "test", "labeled_only": False},
]
DATA_FILES = ["real_grants.csv", "synthetic_ardook_data.csv", "synthetic_ardook_data_alt.csv"]


# ---------------------------------------------------------------- environment

def in_colab() -> bool:
    return "google.colab" in sys.modules


def gpu() -> dict | None:
    """{"name", "memory_gb", "capability"} of GPU 0, or None without an NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError):
        return None
    name, mem, cap = [x.strip() for x in out.split(",")]
    return {"name": name, "memory_gb": round(float(mem) / 1024, 1), "capability": float(cap)}


def pick_model(memory_gb: float | None, capability: float = 8.0) -> dict:
    """The model a GPU can hold, best first. Qwen3-32B in bf16 is exactly what produced the reported results."""
    if memory_gb is None:
        return {"model": None, "note": "No NVIDIA GPU found: use replay mode, or point BASE_URL at a server."}
    if memory_gb >= 70:
        return {"model": "Qwen/Qwen3-32B",
                "note": "Qwen3-32B in bf16: the exact model and precision behind the reported results."}
    if memory_gb >= 38 and capability >= 8.0:
        return {"model": "Qwen/Qwen3-32B-AWQ",
                "note": "Qwen3-32B, Qwen's official 4-bit AWQ build (bf16 needs ~70 GB). Same model, so expect "
                        "the same results within a few grants, not bit-for-bit."}
    if memory_gb >= 14:
        return {"model": "Qwen/Qwen3-4B",
                "note": "Only Qwen3-4B fits this GPU. The pipeline runs end to end, but this is the smaller "
                        "model, so scores will be lower than the reported Qwen3-32B results."}
    return {"model": None, "note": f"{memory_gb} GB is too small for these models: use replay mode."}


def free_disk_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


MODEL_DISK_GB = {"Qwen/Qwen3-32B": 66, "Qwen/Qwen3-32B-AWQ": 20, "Qwen/Qwen3-4B": 9}


# ---------------------------------------------------------------- data and caches

def prepare_data() -> list[str]:
    """Copy the shared datasets into data/ (git-ignored) unless they are already there. Returns what was copied."""
    copied = []
    (ROOT / "data").mkdir(exist_ok=True)
    for name in DATA_FILES:
        dest = ROOT / "data" / name
        if not dest.exists():
            shutil.copy(BUNDLE / "data" / name, dest)
            copied.append(name)
    return copied


def replay_caches(dest: Path) -> dict:
    """Fresh copies of the bundled Qwen3-32B answers (copies, so a run never edits the bundle)."""
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for step in ("checklist", "judge"):
        src = BUNDLE / "cache" / f"{step}__qwen-qwen3-32b.json"
        shutil.copy(src, dest / src.name)
        out[step] = dest / src.name
    return out


# ---------------------------------------------------------------- the model server

def install_vllm(env_dir: Path) -> Path:
    """vLLM in its own virtualenv (via uv), so it can't clash with the notebook's torch. Returns its python."""
    python = Path(env_dir) / "bin" / "python"
    if python.exists() and subprocess.run([python, "-c", "import vllm"], capture_output=True).returncode == 0:
        print(f"vLLM already installed in {env_dir}")
        return python
    if not shutil.which("uv"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    # uv's own Python build, which ships the headers vLLM's Triton kernels compile against.
    subprocess.run([uv, "venv", "--python", "3.12", "--python-preference", "only-managed", str(env_dir)], check=True)
    print(f"Installing vllm=={VLLM_VERSION} (a few minutes; it brings its own PyTorch) ...")
    subprocess.run([uv, "pip", "install", "--python", str(python), f"vllm=={VLLM_VERSION}"], check=True)
    return python


class Server:
    """`vllm serve` in the background, with a random API key, logging to a file."""

    def __init__(self, model: str, extra_args: list[str] | None = None, port: int = 8000, launcher: str = "pip",
                 env_dir: Path | None = None, log: Path | None = None, max_model_len: int = 8192):
        self.model, self.port, self.launcher = model, port, launcher
        self.extra = list(extra_args or [])
        self.env_dir = Path(env_dir or ROOT / ".vllm-env")
        self.log = Path(log or ROOT / "outputs" / "reproduce" / "vllm.log")
        self.max_model_len = max_model_len
        self.proc = None
        self.key = os.environ.get(KEY_ENV) or secrets.token_hex(16)
        os.environ[KEY_ENV] = self.key  # child processes (the pipeline) read it from here
        self.base_url = f"http://127.0.0.1:{port}/v1"

    def command(self) -> list[str]:
        # Same settings as the Akash server: 8192-token context, model served under its own name.
        args = ["serve", self.model, "--served-model-name", self.model, "--port", str(self.port),
                "--max-model-len", str(self.max_model_len), "--gpu-memory-utilization", "0.90",
                "--api-key", self.key, *self.extra]
        if self.launcher == "apptainer":
            return ["apptainer", "exec", "--nv", VLLM_IMAGE, "vllm", *args]
        return [str(self.env_dir / "bin" / "vllm"), *args]

    def start(self) -> "Server":
        self.log.parent.mkdir(parents=True, exist_ok=True)
        print("Starting:", " ".join(c if c != self.key else "<key>" for c in self.command()))
        self.proc = subprocess.Popen(self.command(), stdout=self.log.open("w"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        return self

    def ready(self) -> bool:
        req = urllib.request.Request(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except OSError:
            return False

    def wait(self, timeout_s: int = 3600) -> None:
        """Block until the server answers (the first start downloads the weights: 20-66 GB)."""
        start, last = time.time(), ""
        while time.time() - start < timeout_s:
            if self.ready():
                print(f"\nServer ready after {time.time() - start:.0f}s at {self.base_url}")
                return
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited (code {self.proc.returncode}). Last lines of {self.log}:\n{self.tail()}")
            line = self.tail(1)
            if line and line != last:
                print(f"[{time.time() - start:5.0f}s] {line[:150]}")
                last = line
            time.sleep(10)
        raise TimeoutError(f"vLLM not ready after {timeout_s}s. See {self.log}")

    def tail(self, n: int = 30) -> str:
        if not self.log.exists():
            return ""
        lines = self.log.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]).strip()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, 15)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        print("Server stopped.")


# ---------------------------------------------------------------- running the system

def sh(cmd: list, quiet_prefixes: tuple = ()) -> None:
    """Run a command from the repo root, streaming its output; raise if it fails."""
    print("$", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    for line in proc.stdout:
        if not line.startswith(quiet_prefixes):
            print(line, end="")
    if proc.wait():
        raise RuntimeError(f"command failed with exit code {proc.returncode}")


def dataset_args(ds: dict, limit: int | None) -> list[str]:
    args = ["--input", ds["input"], "--profile", PROFILE]
    args += ["--split", ds["split"]] if ds["split"] else []
    args += ["--labeled-only"] if ds["labeled_only"] else []
    args += ["--limit", str(limit)] if limit else []
    return args


def run_system(ds: dict, out_dir: Path, caches: dict, *, replay: bool, model: str = "Qwen/Qwen3-32B",
               base_url: str | None = None, key_env: str = KEY_ENV, workers: int = 16,
               limit: int | None = None) -> dict:
    """Checklist + judge on one dataset. Returns {"checklist": dir, "judge": dir}."""
    dirs = {"checklist": Path(out_dir) / ds["key"] / "checklist", "judge": Path(out_dir) / ds["key"] / "judge"}
    server = ["--model", model] + ([] if replay else ["--base-url", base_url, "--api-key-env", key_env])
    mode = ["--rules-only"] if replay else ["--workers", str(workers)]
    sh([sys.executable, "src/run_pipeline.py", "--backend", "api", *dataset_args(ds, limit), *server, *mode,
        "--cache", caches["checklist"], "--out", dirs["checklist"], "--quiet"])
    sh([sys.executable, "src/run_judge.py", *dataset_args(ds, limit), "--k", str(JUDGE_K), *server, *mode,
        "--cache", caches["judge"], "--out", dirs["judge"]])
    return dirs


# ---------------------------------------------------------------- scoring

def read_results(run_dir: Path) -> dict[str, dict]:
    with (Path(run_dir) / "results.csv").open(encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def final_predictions(dirs: dict) -> dict[str, dict]:
    """{id: {"program", "gold", "fit", "relevant"}}: judge Fit, forced to Low when the checklist rules it out."""
    elig, fit = read_results(dirs["checklist"]), read_results(dirs["judge"])
    out = {}
    for gid, e in elig.items():
        if gid not in fit:
            continue
        final = "Low" if e["eligibility"] == "Not a lead" else fit[gid]["fit"]
        out[gid] = {"program": e["program"], "gold": e["gold_fit"], "fit": final, "relevant": final != "Low",
                    "why": e["eligibility_why"] if e["eligibility"] == "Not a lead" else fit[gid]["why"]}
    return out


def score(dirs: dict) -> list[dict]:
    """Metrics for the checklist alone, the judge alone, and the final system."""
    from combined_eval import binary
    from ardook_grants.metrics import fit_scores

    elig, fit = read_results(dirs["checklist"]), read_results(dirs["judge"])
    final = final_predictions(dirs)
    ids = [i for i in final if final[i]["gold"]]
    gold = [final[i]["gold"] for i in ids]
    rows = []
    for system, pred in (("checklist alone", [elig[i]["fit"] for i in ids]),
                         ("RAG judge alone", [fit[i]["fit"] for i in ids]),
                         ("final system", [final[i]["fit"] for i in ids])):
        s, b = fit_scores(gold, pred), binary(gold, pred)
        rows.append({"system": system, "n": len(ids), "3-tier accuracy": round(s["accuracy"], 3),
                     "3-tier macro F1": round(s["f1"], 3), "relevant accuracy": round(b["acc"], 3),
                     "relevant precision": round(b["p"], 3), "relevant recall": round(b["r"], 3),
                     "relevant F1": round(b["f1"], 3)})
    return rows


def expected() -> list[dict]:
    with (BUNDLE / "expected_results.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agreement(a: dict, b: dict) -> dict:
    """How often two runs give the same final Fit and the same Relevant call, on the grants both have."""
    pa, pb = final_predictions(a), final_predictions(b)
    ids = [i for i in pa if i in pb]
    if not ids:
        return {"n": 0}
    return {"n": len(ids), "same Fit": round(sum(pa[i]["fit"] == pb[i]["fit"] for i in ids) / len(ids), 3),
            "same Relevant": round(sum(pa[i]["relevant"] == pb[i]["relevant"] for i in ids) / len(ids), 3),
            "differs": [(i, pa[i]["program"][:40], pa[i]["fit"], pb[i]["fit"]) for i in ids if pa[i]["fit"] != pb[i]["fit"]]}


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
