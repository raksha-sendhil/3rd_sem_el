"""Is Gemini reachable right now, and is a problem the keys' or Google's?

    python tools/gemini_check.py                 # asks for the keys (typed text is hidden)
    python tools/gemini_check.py --rounds 10     # more requests per key and model
    GEMINI_API_KEYS=key1,key2 python tools/gemini_check.py

For every key and every model it sends the same small request several times, plus one about
the size of an Agent 2 call, and counts what came back:

* 200: answered;
* 503 / 500: the model is overloaded on Google's side (every key sees it alike);
* 429: this key's quota (per minute or per day) is used up;
* 401 / 403: the key is invalid or blocked;
* 404: this key cannot use this model.

Keys are never printed or saved: each is shown by a short fingerprint. Run it with your keys and
with a friend's: if every key gets 503 on a model at the same time, it is Google's capacity, not
the account and not this project's code.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
SMALL = "Reply with the single word OK."
# about the size of one Agent 2 call: a short medical record and an instruction
LARGE = ("List every fact in this record as JSON {\"facts\": [...]}.\n\n" + (
    "The patient, a 54 year old teacher, reported a persistent cough and shortness of breath "
    "for three weeks. Chest X-ray: lungs clear, no consolidation, infiltrate or effusion. "
    "Impression: no acute cardiopulmonary process. Plan: review in two weeks. ") * 12)


def fingerprint(key: str) -> str:
    return "key-" + hashlib.sha256(key.encode()).hexdigest()[:6]


def ask(key: str, model: str, prompt: str) -> tuple[int, float, str]:
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 64}}).encode()  # fmt: skip
    req = urllib.request.Request(URL.format(model=model), data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})  # fmt: skip
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
            return resp.status, time.perf_counter() - t, ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        return exc.code, time.perf_counter() - t, detail[:120]
    except Exception as exc:  # network
        return 0, time.perf_counter() - t, str(exc)[:120]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, default=5, help="small requests per key and model")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--pause", type=float, default=4.5, help="seconds between requests")
    args = ap.parse_args()
    raw = os.environ.get("GEMINI_API_KEYS") or getpass.getpass(
        "Gemini API key(s), comma-separated (hidden): ")
    keys = list(dict.fromkeys(k.strip() for k in raw.split(",") if k.strip()))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"{len(keys)} key(s), {len(models)} model(s), {args.rounds} small + 1 large request "
          f"each, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    rows = []
    for key in keys:
        for model in models:
            codes, times, notes = [], [], set()
            for i in range(args.rounds + 1):
                code, secs, note = ask(key, model, LARGE if i == args.rounds else SMALL)
                codes.append(code)
                if code == 200:
                    times.append(secs)
                elif note:
                    notes.add(f"{code}: {note}")
                time.sleep(args.pause)
                if code in (401, 403, 404):
                    break  # the same answer every time: no need to ask again
            ok = codes.count(200)
            busy = sum(c in (500, 503) for c in codes)
            limit = codes.count(429)
            refused = sum(c in (401, 403) for c in codes)
            missing = codes.count(404)
            med = f"{statistics.median(times):.1f} s" if times else "-"
            rows.append((fingerprint(key), model, ok, len(codes), busy, limit, refused, missing, med))
            print(f"{fingerprint(key)}  {model:24s} ok {ok}/{len(codes)}  busy(503) {busy}  "
                  f"quota(429) {limit}  bad key {refused}  no model {missing}  median {med}",
                  flush=True)  # fmt: skip
            for n in sorted(notes)[:2]:
                print(f"    {n}")
    print("\nHow to read it:")
    print("* busy(503) on a model for every key  -> Google's capacity for that model; try another"
          " model or later.")
    print("* quota(429) on one key only           -> that key's limit; a key from another Google"
          " project helps.")
    print("* bad key / no model                   -> the key itself, or that model is not open to"
          " it.")


if __name__ == "__main__":
    main()
