#!/usr/bin/env python3
"""Wire the Raidho Pipe into a running Open WebUI via its REST API — no manual
clicks. Verified live against Open WebUI 0.9.x.

Flow: sign up the first user (becomes admin) or sign in if it exists → create
(or update) the `raidho` pipe function from integrations/openwebui_raidho.py →
set its Valves (provider + key) → enable it → verify it serves a live answer.

Idempotent: re-running updates the existing function and re-applies valves.

⚠️ The Open WebUI process must be able to `import agent` / `import vsa` — i.e.
Raidho must be installed in the SAME Python environment Open WebUI runs in
(same venv for the pip path; `pip install -e .` inside the container for Docker).
Otherwise create fails with "No module named 'agent'". The installer handles this.

Usage:
  owui_autowire.py --base URL --email E --password P \
      --provider deepseek --key sk-... [--model M] [--reason-provider P --reason-key K]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PIPE_PATH = Path(__file__).resolve().parent.parent / "integrations" / "openwebui_raidho.py"


def _call(base, method, path, token=None, body=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)              # http://localhost:3000
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--name", default="Raidho Admin")
    ap.add_argument("--provider", required=True)          # execution backend
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")             # for openai-compat
    ap.add_argument("--reason-provider", default="")      # optional split
    ap.add_argument("--reason-key", default="")
    ap.add_argument("--reason-model", default="")
    a = ap.parse_args()

    base = a.base.rstrip("/")
    pipe = PIPE_PATH.read_text(encoding="utf-8")

    # 1) admin: signup (first user) or signin
    st, res = _call(base, "POST", "/api/v1/auths/signup",
                    body={"name": a.name, "email": a.email, "password": a.password})
    if st != 200:
        st, res = _call(base, "POST", "/api/v1/auths/signin",
                        body={"email": a.email, "password": a.password})
    token = res.get("token") if isinstance(res, dict) else None
    if not token:
        print(f"[owui] auth failed: HTTP {st} {res}", file=sys.stderr)
        return 1

    # 2) create the pipe function, or update if it already exists
    fn = {"id": "raidho", "name": "Raidho", "content": pipe,
          "meta": {"description": "Raidho coder agent — chat / council / code"}}
    st, res = _call(base, "POST", "/api/v1/functions/create", token=token, body=fn)
    if st != 200 and "already" in str(res).lower():
        st, res = _call(base, "POST", "/api/v1/functions/id/raidho/update",
                        token=token, body=fn)
    if st != 200:
        print(f"[owui] function create/update failed: HTTP {st} {res}", file=sys.stderr)
        return 1

    # 3) valves: provider + key (+ optional reasoning split)
    valves = {"provider": a.provider, "api_key": a.key, "enable_code": False}
    if a.model:
        valves["model"] = a.model
    if a.base_url:
        valves["base_url"] = a.base_url
    if a.reason_provider:
        valves["reason_provider"] = a.reason_provider
        valves["reason_api_key"] = a.reason_key
        if a.reason_model:
            valves["reason_model"] = a.reason_model
    st, res = _call(base, "POST", "/api/v1/functions/id/raidho/valves/update",
                    token=token, body=valves)
    if st != 200:
        print(f"[owui] valves update failed: HTTP {st} {res}", file=sys.stderr)
        return 1

    # 4) ensure ENABLED — /toggle flips state, so toggling blindly would turn an
    #    already-active function OFF on a re-run. Read state first, flip only if off.
    st, lst = _call(base, "GET", "/api/v1/functions/", token=token)
    active = next((f.get("is_active") for f in lst if isinstance(f, dict)
                   and f.get("id") == "raidho"), None) if isinstance(lst, list) else None
    if active is not True:
        _call(base, "POST", "/api/v1/functions/id/raidho/toggle", token=token, body={})

    # 5) verify — model present + live answer through the OWUI stack
    st, res = _call(base, "GET", "/api/models", token=token)
    models = [m.get("id") for m in res.get("data", [])] if isinstance(res, dict) else []
    raidho = [m for m in models if "raidho" in str(m).lower()]
    if not raidho:
        print(f"[owui] wired, but no Raidho model is listed yet: {models}", file=sys.stderr)
        return 1
    st, res = _call(base, "POST", "/api/chat/completions", token=token,
                    body={"model": raidho[0], "stream": False,
                          "messages": [{"role": "user", "content": "Reply exactly: OWUI OK"}]})
    answer = ""
    if isinstance(res, dict):
        answer = (res.get("choices", [{}])[0].get("message", {}).get("content", ""))
    ok = "OWUI OK" in answer
    print(f"[owui] models={raidho} live_answer={'ok' if ok else repr(answer)[:80]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
