"""Rent a GPU on Akash (through the Console API) and serve an LLM there for the pipeline's --backend api.

Uses the Console account's credits: needs a Console API key in $AKASH_API_KEY (or an AKASH_API_KEY=...
line in the repo's git-ignored .env). Steps, from the repo root:

  python deploy/akash/akash_llm.py up        # create the deployment, wait for bids, list them with $/hour
  python deploy/akash/akash_llm.py lease     # rent the cheapest bid (or --provider akash1...)
  python deploy/akash/akash_llm.py wait      # wait until the model answers; saves its URL and key to .env
  python src/run_pipeline.py --input data/real_grants.csv --labeled-only --backend api \
      --model Qwen/Qwen3-32B --base-url <url printed by wait> --api-key-env AKASH_LLM_KEY --no-thinking --workers 8
  python deploy/akash/akash_llm.py down      # close the deployment; unspent credit returns to the account
  python deploy/akash/akash_llm.py logs      # container logs (--events for Kubernetes events), to debug startup

`up` alone costs nothing but holds a small refundable deposit; renting starts at `lease`. As a safety net,
Console closes the deployment by itself RUNTIME_LIMIT_HOURS after the lease starts.
State (the deployment's dseq) is kept in outputs/akash/state.json, which git ignores.
"""

import argparse
import json
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ardook_grants.backends import read_secret  # noqa: E402

API = "https://console-api.akash.network"
SDL_PATH = Path(__file__).with_name("vllm-qwen3-32b.yaml")
STATE = REPO_ROOT / "outputs" / "akash" / "state.json"
ENV_FILE = REPO_ROOT / ".env"
RUNTIME_LIMIT_HOURS = 3
BLOCKS_PER_HOUR = 600  # ~6 s per block
SERVICE = "llm"
USER_AGENT = "ardook-grants-pipeline/1.0"


def call(method: str, path: str, body: dict | None = None, auth: bool = True) -> dict:
    # A plain client name: Cloudflare in front of the Console API blocks Python's default one (error 1010).
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if auth:
        key = read_secret("AKASH_API_KEY")
        if not key:
            sys.exit("No Console API key: set $AKASH_API_KEY or add AKASH_API_KEY=... to .env")
        headers["x-api-key"] = key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"Console API {method} {path} -> {e.code}: {e.read().decode(errors='replace')[:500]}")


def set_env_line(name: str, value: str) -> None:
    """Add or replace NAME=value in .env (git-ignored)."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if not line.startswith(f"{name}=")] + [f"{name}={value}"]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_state() -> dict:
    if not STATE.exists():
        sys.exit("No deployment yet: run `up` first.")
    return json.loads(STATE.read_text())


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def gpu_names(bid: dict) -> str:
    text = json.dumps(bid.get("resources_offer", []))
    return ", ".join(sorted(set(re.findall(r"(a100|h100|pro6000se)", text)))) or "?"


def open_bids(dseq: str) -> list[dict]:
    bids = [b["bid"] for b in call("GET", f"/v1/bids?dseq={dseq}").get("data", []) if b["bid"]["state"] == "open"]
    return sorted(bids, key=lambda b: float(b["price"]["amount"]))


def show_bids(bids: list[dict]) -> None:
    print(f"{len(bids)} open bid(s), cheapest first (1 ACT ≈ $1 of Console credit):")
    for i, b in enumerate(bids):
        per_hour = float(b["price"]["amount"]) * BLOCKS_PER_HOUR / 1e6
        print(f"  [{i}] {b['id']['provider']}  {gpu_names(b):10}  {per_hour:6.2f} ACT/hour"
              f"{'  reclaimable with ' + b['reclamation_window'] + ' notice' if b.get('reclamation_window') else ''}")


def up(args) -> None:
    if STATE.exists() and not args.force:
        sys.exit(f"A deployment is already recorded ({load_state()['dseq']}). Run `down` first, or pass --force.")
    key = read_secret("AKASH_LLM_KEY") or secrets.token_urlsafe(32)
    set_env_line("AKASH_LLM_KEY", key)  # the pipeline reads it from .env; never printed
    sdl = SDL_PATH.read_text(encoding="utf-8").replace("__VLLM_API_KEY__", key)
    data = call("POST", "/v1/deployments", {"data": {"sdl": sdl, "name": "ardook-grants-llm",
                                                     "runtimeLimitHours": RUNTIME_LIMIT_HOURS}})["data"]
    dseq = data["dseq"]
    save_state({"dseq": dseq, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"Deployment {dseq} created (not renting anything yet). Waiting for bids ...")
    bids = []
    for _ in range(24):
        time.sleep(5)
        bids = open_bids(dseq)
        if bids:
            time.sleep(10)  # give slower providers a moment too
            bids = open_bids(dseq)
            break
    if not bids:
        print("No bids after 2 minutes: no provider matches the SDL right now. Run `down` to release the deposit.")
        return
    show_bids(bids)
    print("\nNext: `lease` (cheapest) or `lease --provider <address>`; `down` to cancel.")


def lease(args) -> None:
    state = load_state()
    bids = open_bids(state["dseq"])
    if not bids:
        sys.exit("No open bids (they expire after a few minutes). Run `down`, then `up` again.")
    bid = next((b for b in bids if b["id"]["provider"] == args.provider), None) if args.provider else bids[0]
    if bid is None:
        sys.exit(f"No open bid from {args.provider}.")
    i = bid["id"]
    call("POST", "/v1/leases", {"leases": [{"dseq": i["dseq"], "gseq": i["gseq"], "oseq": i["oseq"], "provider": i["provider"]}]})
    per_hour = float(bid["price"]["amount"]) * BLOCKS_PER_HOUR / 1e6
    state.update(provider=i["provider"], act_per_hour=round(per_hour, 3), leased=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_state(state)
    print(f"Leased {gpu_names(bid)} from {i['provider']} at {per_hour:.2f} ACT/hour. "
          f"Console closes it after {RUNTIME_LIMIT_HOURS} h at the latest. Next: `wait`.")


def deployment(dseq: str) -> dict:
    return call("GET", f"/v1/deployments/{dseq}")["data"]


def url_of(dep: dict) -> str | None:
    leases = dep.get("leases") or []
    services = ((leases[0].get("status") or {}).get("services") or {}) if leases else {}
    uris = (services.get(SERVICE) or {}).get("uris") or []
    return f"http://{uris[0]}" if uris else None


def wait(args) -> None:
    state = load_state()
    key = read_secret("AKASH_LLM_KEY")
    deadline = time.time() + args.timeout * 60
    url = None
    while time.time() < deadline:
        url = url or url_of(deployment(state["dseq"]))
        if url:
            request = urllib.request.Request(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}",
                                                                         "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=20) as r:
                    models = [m["id"] for m in json.loads(r.read())["data"]]
                set_env_line("AKASH_LLM_URL", f"{url}/v1")
                state["url"] = f"{url}/v1"
                save_state(state)
                print(f"Ready: {url}/v1 serving {models}")
                return
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
                pass  # still downloading the model (several minutes for 66 GB)
        print(f"  {time.strftime('%H:%M:%S')} not ready yet{' (' + url + ')' if url else ''} ...")
        time.sleep(30)
    sys.exit(f"Not ready after {args.timeout} minutes. Check `status`; `down` stops the charges.")


def status(args) -> None:
    state = load_state()
    dep = deployment(state["dseq"])
    leases = dep.get("leases") or []
    print(json.dumps({"dseq": state["dseq"], "state": dep["deployment"]["state"], "url": url_of(dep),
                      "provider": leases[0]["id"]["provider"] if leases else None,
                      "services": ((leases[0].get("status") or {}).get("services") if leases else None),
                      "act_per_hour": state.get("act_per_hour")}, indent=1))
    print("weekly cost (USD):", call("GET", "/v1/weekly-cost").get("data"))


def logs(args) -> None:
    """Recent container logs or Kubernetes events, read from the provider (needs `pip install websockets`)."""
    import asyncio
    import ssl

    import websockets

    state = load_state()
    lease = deployment(state["dseq"])["leases"][0]["id"]
    req = urllib.request.Request(f"{API}/v1/providers/{lease['provider']}", headers={"User-Agent": USER_AGENT})
    host = json.loads(urllib.request.urlopen(req, timeout=60).read())["hostUri"].removeprefix("https://")
    jwt = call("POST", "/v1/create-jwt-token", {"data": {"ttl": 600, "leases": {"access": "scoped",
                                                                                 "scope": ["logs", "events"]}}})["data"]["token"]
    kind, query = ("kubeevents", "follow=false") if args.events else ("logs", f"follow=false&tail={args.tail}")
    url = f"wss://{host}/lease/{state['dseq']}/{lease['gseq']}/{lease['oseq']}/{kind}?{query}"
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # providers use self-signed certs; the JWT authenticates

    async def read() -> None:
        async with websockets.connect(url, ssl=ctx, additional_headers={"Authorization": f"Bearer {jwt}"}) as ws:
            while True:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=8))
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    return
                print((msg.get("message") or msg.get("note") or str(msg)).rstrip()[:300])

    asyncio.run(read())


def down(args) -> None:
    state = load_state()
    call("DELETE", f"/v1/deployments/{state['dseq']}")
    STATE.unlink()
    print(f"Closed deployment {state['dseq']}; unspent credit returns to the account.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("up", help="create the deployment and list bids")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=up)
    p = sub.add_parser("lease", help="rent a bid")
    p.add_argument("--provider", help="provider address (default: cheapest bid)")
    p.set_defaults(fn=lease)
    p = sub.add_parser("wait", help="wait until the model answers")
    p.add_argument("--timeout", type=int, default=45, help="minutes")
    p.set_defaults(fn=wait)
    sub.add_parser("status", help="show the deployment").set_defaults(fn=status)
    p = sub.add_parser("logs", help="recent container logs (or --events); needs `pip install websockets`")
    p.add_argument("--events", action="store_true", help="Kubernetes events instead of logs")
    p.add_argument("--tail", type=int, default=30)
    p.set_defaults(fn=logs)
    sub.add_parser("down", help="close the deployment").set_defaults(fn=down)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
