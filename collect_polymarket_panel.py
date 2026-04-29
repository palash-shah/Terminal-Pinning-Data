"""
collect_polymarket_panel.py  (v4 - cascading fidelity, broader topic coverage)
==============================================================================

CHANGES FROM V3 (the one that returned 0 ticks for 49/53 markets)
-----------------------------------------------------------------
The Polymarket CLOB endpoint /prices-history silently returns an empty
"history" array when the requested fidelity is finer than what's retained
for that market.  Specifically: for resolved/closed markets that are more
than a few weeks old, only fidelity >= 720 minutes (12 hours) is retained.
Asking for fidelity=1 (which v3 did) returns history=[] for older markets.

This is a known issue: github.com/Polymarket/py-clob-client #216.

The fix in v4: cascade through a sequence of fidelities from finest to
coarsest, and accept the first one that returns enough ticks.  We also:

- Broaden topic taxonomy (more sports keywords; nfl/nba/mlb subdivide;
  crypto altcoins; politics subdivides; macro; tech)
- Default --max-per-topic 6 (was 4) and expand topic dictionary
- Lower --min-ticks default to 50 (was 200) since coarser-fidelity
  markets will have fewer points but still be analyzable
- Use timezone-aware datetime (no deprecation warnings)
- Show which fidelity worked per market in the log
- Better progress: shows "X kept so far / Y target" in real time
- Increase scan-limit default to 8000 (was 4000) so candidate pool grows

USAGE
-----
  python3 collect_polymarket_panel.py                      # 30 markets, default
  python3 collect_polymarket_panel.py --target 50          # 50 markets
  python3 collect_polymarket_panel.py --probe              # smoke-test API

REQUIREMENTS: Python 3.8+, no external libraries.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
UA    = "Mozilla/5.0 (compatible; bridge-model-research/4.0)"

# Cascading fidelity sequence in minutes.
# Polymarket retains ~12-hour candles for old markets but ~1-minute candles
# for recent ones.  Try finest first, fall back to coarser.
FIDELITY_LADDER = [1, 10, 60, 360, 720, 1440]

DEBUG = False


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
def http_get_json(url, params=None, max_retries=4, timeout=30):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    if DEBUG:
        print(f"  GET {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return json.loads(resp.read().decode("utf-8")), resp.status, None
                return None, resp.status, "non-2xx"
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                body = ""
            return None, e.code, body
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None, None, "max retries"


def parse_iso(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def utc_iso(t_unix):
    """Timezone-aware UTC ISO string (no deprecation warning)."""
    return datetime.datetime.fromtimestamp(t_unix, tz=datetime.timezone.utc) \
        .isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------
def probe():
    print("=== Probe ===")
    data, status, err = http_get_json(f"{GAMMA}/markets",
                                       params={"limit": 1, "closed": "true"})
    if err:
        print(f"  Gamma /markets: FAILED status={status} err={err}")
        return
    print(f"  Gamma /markets: OK")
    if isinstance(data, list) and data:
        m = data[0]
        tids = m.get("clobTokenIds")
        if isinstance(tids, str):
            tids = json.loads(tids)
        if tids:
            print(f"  Trying CLOB /prices-history with each fidelity in ladder...")
            for fid in FIDELITY_LADDER:
                d, s, e = http_get_json(f"{CLOB}/prices-history",
                    params={"market": tids[0], "interval": "max",
                            "fidelity": fid})
                hist = (d or {}).get("history", []) or []
                print(f"    fidelity={fid} min: status={s}, n={len(hist)}")
                if len(hist) > 0:
                    break


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def parse_outcomes_and_tokens(m):
    outcomes = m.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            return None, None
    tids = m.get("clobTokenIds")
    if isinstance(tids, str):
        try:
            tids = json.loads(tids)
        except Exception:
            return None, None
    if not (isinstance(outcomes, list) and isinstance(tids, list)
            and len(outcomes) == 2 and len(tids) == 2):
        return None, None
    return outcomes, tids


def parse_outcome_prices(m):
    op = m.get("outcomePrices")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except Exception:
            return None
    if not isinstance(op, list) or len(op) != 2:
        return None
    try:
        return [float(x) for x in op]
    except Exception:
        return None


def discover(scan_limit=8000, page_size=100, min_volume=500_000):
    print(f"Scanning closed markets (volume desc), up to offset {scan_limit}...")
    candidates = []
    offset = 0
    while offset < scan_limit:
        data, status, err = http_get_json(f"{GAMMA}/markets",
            params={"closed": "true", "limit": page_size, "offset": offset,
                    "order": "volumeNum", "ascending": "false"})
        if err or not isinstance(data, list):
            print(f"  page offset={offset}: error {status}: {err}; stopping")
            break
        if not data:
            print(f"  page offset={offset}: empty; stopping")
            break
        for m in data:
            if not m.get("closed"):
                continue
            outcomes, tids = parse_outcomes_and_tokens(m)
            if not outcomes:
                continue
            lower = [o.strip().lower() for o in outcomes]
            if set(lower) != {"yes", "no"}:
                continue
            op = parse_outcome_prices(m)
            if not op:
                continue
            yes_idx = lower.index("yes")
            p_yes_final = op[yes_idx]
            if not (p_yes_final > 0.95 or p_yes_final < 0.05):
                continue
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0
            if vol < min_volume:
                continue
            candidates.append({
                "slug": m.get("slug"),
                "question": m.get("question") or "",
                "yes_token_id": tids[yes_idx],
                "p_yes_final": p_yes_final,
                "expected_outcome": "Yes" if p_yes_final > 0.5 else "No",
                "volume": vol,
                "start_iso": m.get("startDate"),
                "end_iso": m.get("endDate"),
                "start_unix": parse_iso(m.get("startDate")),
                "end_unix": parse_iso(m.get("endDate")),
            })
        offset += page_size
        time.sleep(0.15)
    print(f"  found {len(candidates)} closed binary markets (vol >= ${min_volume/1e6:.2f}M)")
    return candidates


# ----------------------------------------------------------------------
# Topic taxonomy (broadened)
# ----------------------------------------------------------------------
TOPIC_KEYWORDS = {
    # Politics - subdivided
    "trump":      ["trump", "donald trump"],
    "biden":      ["biden", "joe biden"],
    "harris":     ["harris", "kamala"],
    "vance":      ["vance", "j.d. vance"],
    "election":   ["election", "presidential", "senate", "house race",
                   "governor", "primary"],
    "uk_politics": ["uk election", "starmer", "labour", "tories",
                    "conservative party"],
    "eu_politics": ["macron", "scholz", "le pen", "afd", "european"],

    # Geopolitics
    "iran":       ["iran", "tehran", "iranian"],
    "israel":     ["israel", "netanyahu", "gaza", "hamas", "hezbollah",
                   "lebanon", "yemen", "houthi"],
    "russia":     ["russia", "putin", "ukraine", "kyiv", "zelensky",
                   "zelenskyy"],
    "china":      ["china", "xi jinping", "taiwan", "beijing"],

    # Macro / policy
    "fed":        ["fed", "fomc", "powell", "rate cut", "rate hike",
                   "interest rate"],
    "inflation":  ["cpi", "inflation", "ppi"],
    "gdp":        ["gdp", "recession", "unemployment"],
    "scotus":     ["scotus", "supreme court"],
    "shutdown":   ["shutdown", "debt ceiling"],

    # Crypto
    "btc":        ["bitcoin", "btc"],
    "eth":        ["ethereum", "eth "],
    "altcoin":    ["solana", "doge", "sol ", "ada", "xrp"],
    "stablecoin": ["usdc", "usdt", "tether", "stablecoin"],

    # Sports - subdivided
    "nfl":        ["super bowl", "nfl", "afc", "nfc"],
    "nba":        ["nba", "lakers", "celtics", "warriors", "knicks"],
    "mlb":        ["mlb", "world series", "yankees", "dodgers"],
    "nhl":        ["nhl", "stanley cup"],
    "soccer":     ["champions league", "premier league", "world cup",
                   "manchester", "arsenal", "real madrid", "bayern"],
    "tennis":     ["wimbledon", "us open", "australian open", "french open"],
    "f1":         ["formula 1", "f1 ", "verstappen", "hamilton race"],
    "olympics":   ["olympic"],

    # Tech / business
    "tech":       ["openai", "tesla", "spacex", "twitter", "x ", "google",
                   "apple", "microsoft", "amazon", "nvidia", "facebook",
                   "meta "],
    "ai":         ["gpt", "anthropic", "claude", "ai model", "chatgpt"],
    "tiktok":     ["tiktok"],
    "ipo":        ["ipo"],

    # Entertainment / culture
    "celebrity":  ["taylor swift", "elon", "musk", "kanye", "kardashian"],
    "oscars":     ["oscar", "academy award"],

    # Weather / climate
    "weather":    ["hurricane", "tropical storm", "typhoon", "tornado",
                   "atlantic"],
    "temperature":["heat wave", "warmest", "coldest"],
}


def topic_of(question):
    q = question.lower()
    for tag, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return tag
    return "other"


def dedupe(candidates, max_per_topic=6, max_per_week=10):
    by_topic = {}
    by_week = {}
    kept = []
    for c in candidates:
        topic = topic_of(c["question"])
        c["topic"] = topic
        if c["end_unix"]:
            week = int(c["end_unix"] // (7 * 86400))
        else:
            week = -1
        if by_topic.get(topic, 0) >= max_per_topic:
            continue
        if by_week.get(week, 0) >= max_per_week:
            continue
        by_topic[topic] = by_topic.get(topic, 0) + 1
        by_week[week] = by_week.get(week, 0) + 1
        kept.append(c)
    print(f"  after dedup (max {max_per_topic}/topic, {max_per_week}/week): "
          f"{len(kept)} markets")
    topic_breakdown = {}
    for c in kept:
        topic_breakdown[c["topic"]] = topic_breakdown.get(c["topic"], 0) + 1
    print(f"  topic breakdown: {topic_breakdown}")
    return kept


# ----------------------------------------------------------------------
# History fetch with cascading fidelity
# ----------------------------------------------------------------------
def fetch_history_cascade(token_id, min_ticks=50, fidelities=FIDELITY_LADDER):
    """Try each fidelity in turn; return (history, fidelity_used) for first
    fidelity that yields >= min_ticks data points.  Returns ([], None) if all
    fidelities fail."""
    for fid in fidelities:
        data, status, err = http_get_json(f"{CLOB}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": fid})
        if err or not isinstance(data, dict):
            continue
        history = data.get("history") or []
        if len(history) >= min_ticks:
            return history, fid
        if DEBUG:
            print(f"    fidelity={fid}: only {len(history)} ticks", file=sys.stderr)
        time.sleep(0.05)
    return [], None


def interior_fraction(history, lo=0.05, hi=0.95):
    if not history:
        return 0.0
    n = sum(1 for pt in history if lo <= pt.get("p", -1) <= hi)
    return n / len(history)


# ----------------------------------------------------------------------
# Main collection
# ----------------------------------------------------------------------
def collect(target_count=30, scan_limit=8000, min_interior=0.20,
            min_volume=500_000, min_ticks=50,
            max_per_topic=6, max_per_week=10):
    candidates = discover(scan_limit=scan_limit, min_volume=min_volume)
    if not candidates:
        print("No candidates.")
        return
    candidates = dedupe(candidates, max_per_topic=max_per_topic,
                        max_per_week=max_per_week)

    outdir = "polymarket_panel"
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, "manifest.csv")

    write_header = (not os.path.exists(manifest_path)
                    or os.path.getsize(manifest_path) == 0)
    mf = open(manifest_path, "w", newline="")
    manifest = csv.writer(mf)
    if write_header:
        manifest.writerow([
            "slug", "question", "topic", "expected_outcome", "yes_token_id",
            "n_observations", "interior_fraction", "fidelity_used_min",
            "first_t_unix", "last_t_unix",
            "first_t_iso", "last_t_iso", "p_first", "p_last",
            "volume", "end_date", "csv_path"])
        mf.flush()

    kept = 0
    rejected_for_interior = 0
    rejected_for_ticks = 0

    for ci, cand in enumerate(candidates):
        if kept >= target_count:
            break
        slug = cand["slug"]
        topic = cand.get("topic", "other")
        print(f"\n[{ci+1}/{len(candidates)}] kept={kept}/{target_count} | "
              f"{topic:10s} | {slug[:50]} (vol=${cand['volume']/1e6:.1f}M)")
        time.sleep(0.15)

        history, fid_used = fetch_history_cascade(cand["yes_token_id"],
                                                   min_ticks=min_ticks)
        if not history:
            print(f"  REJECT: no fidelity in ladder yielded >= {min_ticks} ticks")
            rejected_for_ticks += 1
            continue
        ifrac = interior_fraction(history)
        if ifrac < min_interior:
            print(f"  REJECT: interior_fraction {ifrac:.3f} < {min_interior} "
                  f"(n={len(history)} at fid={fid_used} min)")
            rejected_for_interior += 1
            continue
        print(f"  KEEP: n={len(history)} ticks at fidelity={fid_used} min, "
              f"interior_fraction={ifrac:.3f}")

        csv_path = os.path.join(outdir, f"{slug}_yes.csv")
        with open(csv_path, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["t_unix", "t_iso", "p"])
            for pt in history:
                t = pt.get("t")
                p = pt.get("p")
                if t is None or p is None:
                    continue
                w.writerow([t, utc_iso(t), f"{p:.6f}"])

        t0, tN = history[0]["t"], history[-1]["t"]
        p0, pN = history[0]["p"], history[-1]["p"]
        manifest.writerow([
            slug, cand["question"], topic, cand["expected_outcome"],
            cand["yes_token_id"], len(history), f"{ifrac:.4f}", fid_used,
            t0, tN, utc_iso(t0), utc_iso(tN),
            f"{p0:.6f}", f"{pN:.6f}",
            cand.get("volume") or "", cand.get("end_iso") or "", csv_path])
        mf.flush()
        kept += 1

    mf.close()
    print(f"\n=== Done ===")
    print(f"  Kept: {kept} markets")
    print(f"  Rejected (interior): {rejected_for_interior}")
    print(f"  Rejected (ticks): {rejected_for_ticks}")
    print(f"  Wrote {manifest_path}")
    print(f"  Send the polymarket_panel/ folder.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--scan", type=int, default=8000)
    ap.add_argument("--min-interior", type=float, default=0.20)
    ap.add_argument("--min-volume", type=float, default=500_000)
    ap.add_argument("--min-ticks", type=int, default=50)
    ap.add_argument("--max-per-topic", type=int, default=6)
    ap.add_argument("--max-per-week", type=int, default=10)
    args = ap.parse_args()
    DEBUG = args.debug
    if args.probe:
        probe()
    else:
        collect(target_count=args.target, scan_limit=args.scan,
                min_interior=args.min_interior, min_volume=args.min_volume,
                min_ticks=args.min_ticks,
                max_per_topic=args.max_per_topic,
                max_per_week=args.max_per_week)


if __name__ == "__main__":
    main()
