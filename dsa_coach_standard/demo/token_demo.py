#!/usr/bin/env python
"""
demo/token_demo.py — Token-optimisation demo for the DSA Coach agent.

Replays the SAME tutoring session (9 turns, +1 in --mock) twice:

    BASELINE   every optimisation flag off   (the original behaviour)
    OPTIMIZED  every optimisation flag on    (T1 … T9, see llm/optim.py)

and prints calls / prompt tokens / completion tokens per turn, per node and in
total, plus a quality guard-rail (answer length) so savings are not just
"the model said less".

Run from the project root:

    python -m demo.token_demo                 # real model from .env (Ollama by default)
    python -m demo.token_demo --mock          # no model needed: validates flow + prompt sizes
    python -m demo.token_demo --ablation      # add technique-by-technique waterfall
    python -m demo.token_demo --quick         # 4 turns instead of 9
    python -m demo.token_demo --trace         # show [ROUTER]/[CRITIC]/... log lines
    python -m demo.token_demo --show-answers  # print baseline vs optimized answers
    python -m demo.token_demo --json out.json # machine-readable results

Progress and conversation files are written to a temp folder, never to ./data.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_CODE = '''def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''

# id, label, session, kwargs for run(), tags
SCENARIOS = [
    dict(id=1, label="Explain BFS vs DFS",         session="alice", q="Explain BFS vs DFS with an example"),
    dict(id=2, label="How to improve at recursion", session="alice", q="How can I get better at recursion?"),
    dict(id=3, label="Give me an array problem",   session="alice", q="Give me an easy array problem to practice"),
    dict(id=4, label="I'm stuck - hint",           session="alice", q="I'm stuck, can I get a hint?"),
    dict(id=5, label="Review my code",             session="alice", q="Please review my code",
         mode="code_review", code=SAMPLE_CODE),
    dict(id=6, label="Show full solution",         session="alice", q="Show me the full solution"),
    dict(id=7, label="Greeting",                   session="alice", q="hello!"),
    dict(id=8, label="Dijkstra complexity",        session="alice", q="What is the time complexity of Dijkstra's algorithm?"),
    dict(id=9, label="Same as #1, new student",    session="bob",   q="Explain BFS vs DFS with an example"),
    dict(id=10, label="Critic rejects once (mock)", session="carol", q="Explain dynamic programming (retry-demo)",
         mock_only=True),
]
QUICK_IDS = {1, 3, 5, 9}


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="use the offline mock backend")
    p.add_argument("--ablation", action="store_true", help="also run technique-by-technique passes")
    p.add_argument("--quick", action="store_true", help="run 4 turns instead of 9")
    p.add_argument("--trace", action="store_true", help="print agent log lines")
    p.add_argument("--show-answers", action="store_true", help="print baseline vs optimized answers")
    p.add_argument("--estimate", action="store_true",
                   help="count prompt tokens as chars/4 on every backend (ignores KV-cache effects)")
    p.add_argument("--json", metavar="PATH", help="write results to a JSON file")
    return p.parse_args()


def preflight(client) -> None:
    """Fail early with a helpful message if the backend cannot be reached."""
    backend = client.LLM_BACKEND
    if backend == "mock":
        return
    if backend == "gemini":
        if not client.GEMINI_API_KEY:
            sys.exit("GEMINI_API_KEY is not set (put it in .env). Or run with --mock.")
        return
    try:
        import ollama
        listing = ollama.list()
        entries = getattr(listing, "models", None) or listing.get("models", [])
        names = set()
        for m in entries:                       # pydantic objects (>=0.4) or dicts (0.3)
            names.add(getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None) or "")
    except Exception as exc:
        sys.exit(f"Cannot reach Ollama ({exc}).\n"
                 f"  1) start it:      ollama serve\n"
                 f"  2) pull a model:  ollama pull {client.OLLAMA_MODEL}\n"
                 f"  or run the demo without a model:  python -m demo.token_demo --mock")
    wanted = {client.OLLAMA_MODEL, client.OLLAMA_SMALL_MODEL}
    missing = [m for m in wanted if m not in names and f"{m}:latest" not in names]
    if missing:
        sys.exit(f"Model(s) not pulled: {', '.join(missing)}   →  ollama pull {missing[0]}")


def pct(saved: int, base: int) -> str:
    return f"{100 * saved / base:5.1f}%" if base else "   -  "


def bar(fraction: float, width: int = 28) -> str:
    n = max(0, min(width, round(fraction * width)))
    return "█" * n + "·" * (width - n)


# ----------------------------------------------------------------
# one full pass of the session under a given flag configuration
# ----------------------------------------------------------------
def run_pass(name: str, configure, scenarios: list[dict], ctx: dict) -> dict:
    optim, tracker, client = ctx["optim"], ctx["tracker"], ctx["client"]
    store, bank = ctx["store"], ctx["bank"]
    run = ctx["run"]

    configure()
    tracker.reset()
    client.clear_cache()
    if client.LLM_BACKEND == "mock":
        from llm import mock_backend
        mock_backend.reset()

    work = Path(ctx["tmp"]) / name
    (work / "conversations").mkdir(parents=True, exist_ok=True)
    store.PROGRESS_FILE = work / "progress.json"
    store.CONV_DIR = work / "conversations"
    bank.PROBLEMS_FILE = work / "problems.json"
    random.seed(7)                       # same problem picked in every pass

    log = io.StringIO()
    sink = sys.stdout if ctx["trace"] else log
    turns = []
    t_pass = time.perf_counter()
    for sc in scenarios:
        mark = tracker.mark()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sink) if not ctx["trace"] else contextlib.nullcontext():
            if ctx["trace"]:
                print(f"\n--- [{name}] turn {sc['id']}: {sc['label']}")
            result = run(
                question=sc["q"],
                mode=sc.get("mode", ""),
                student_code=sc.get("code", ""),
                session_id=f"{name}-{sc['session']}",
            )
        recs = tracker.records[mark:]
        turns.append({
            "id": sc["id"], "label": sc["label"],
            "intent": result.get("intent", "?"),
            "answer": result.get("final_response", ""),
            "seconds": round(time.perf_counter() - t0, 2),
            **tracker.totals_for(recs),
            "by_tag": tracker.by_tag_for(recs),
            "models": sorted({r.model for r in recs if not r.cached}),
        })

    all_recs = list(tracker.records)
    small = sum(r.total for r in all_recs if not r.cached and r.model != client.model_name("main"))
    errors = [l for l in log.getvalue().splitlines() if l.startswith("[LLM] Backend")]
    return {
        "name": name,
        "turns": turns,
        "totals": {**tracker.totals_for(all_recs), "wall_s": round(time.perf_counter() - t_pass, 1),
                   "small_tier_tokens": small},
        "by_tag": tracker.by_tag_for(all_recs),
        "estimated": any(r.estimated for r in all_recs),
        "errors": errors,
        "flags": {k: getattr(optim.F, k) for k in optim.FLAG_NAMES},
    }


# ----------------------------------------------------------------
# reporting
# ----------------------------------------------------------------
def print_header(ctx: dict, scenarios: list[dict]) -> None:
    client, optim = ctx["client"], ctx["optim"]
    print("=" * 100)
    print(" DSA COACH — TOKEN OPTIMISATION DEMO")
    print("=" * 100)
    print(f" backend      : {client.LLM_BACKEND}"
          + (f"  (model {client.model_name('main')}, small tier {client.model_name('small')})"
             if client.LLM_BACKEND != "mock" else "  (offline — completions are canned)"))
    print(f" token counts : {'estimated (chars/4)' if client.LLM_BACKEND == 'mock' or client.TOKEN_COUNT_MODE == 'estimate' else 'reported by the backend'}")
    print(f" turns        : {len(scenarios)}   (same script for every pass)")
    print(f" optimizations: " + ", ".join(optim.TECHNIQUES[f].split()[0] for f in optim.FLAG_NAMES))
    print("-" * 100)


def print_turns(base: dict, opt: dict) -> None:
    print("\n1) PER TURN — LLM calls and total tokens (prompt + completion)\n")
    print(f" {'#':>2}  {'Turn':<28}{'Intent':<12}│ {'BASELINE':^16} │ {'OPTIMIZED':^16} │ {'Saved':>13}")
    print(f" {'':>2}  {'':<28}{'':<12}│ {'calls':>5} {'tokens':>10} │ {'calls':>5} {'tokens':>10} │ {'tokens':>6} {'%':>6}")
    print(" " + "─" * 98)
    for b, o in zip(base["turns"], opt["turns"]):
        saved = b["total_tokens"] - o["total_tokens"]
        cache = "  ⚡cache" if o["cache_hits"] else ""
        print(f" {b['id']:>2}  {b['label'][:27]:<28}{o['intent']:<12}│ {b['calls']:>5} {b['total_tokens']:>10,} │ "
              f"{o['calls']:>5} {o['total_tokens']:>10,} │ {saved:>6,} {pct(saved, b['total_tokens'])}{cache}")


def print_totals(base: dict, opt: dict) -> None:
    b, o = base["totals"], opt["totals"]
    print("\n2) TOTALS\n")
    print(f" {'':<28}{'BASELINE':>12}{'OPTIMIZED':>12}{'Saved':>12}{'%':>8}")
    print(" " + "─" * 72)
    rows = [("LLM calls", "calls"), ("Prompt tokens (input)", "prompt_tokens"),
            ("Completion tokens (output)", "completion_tokens"), ("TOTAL TOKENS", "total_tokens")]
    for label, key in rows:
        s = b[key] - o[key]
        print(f" {label:<28}{b[key]:>12,}{o[key]:>12,}{s:>12,}{pct(s, b[key]):>8}")
    print(f" {'Wall time (s)':<28}{b['wall_s']:>12}{o['wall_s']:>12}")
    print(f" {'Cache hits':<28}{b['cache_hits']:>12}{o['cache_hits']:>12}"
          f"    (~{o['saved_by_cache']:,} tokens avoided)")
    if o["small_tier_tokens"]:
        print(f" {'Tokens on small model tier':<28}{'-':>12}{o['small_tier_tokens']:>12,}")
    frac = 1 - (o["total_tokens"] / b["total_tokens"]) if b["total_tokens"] else 0
    print(f"\n   baseline  {bar(1.0)} {b['total_tokens']:,}")
    print(f"   optimized {bar(1 - frac)} {o['total_tokens']:,}   ({frac * 100:.0f}% fewer tokens)")


def print_by_node(base: dict, opt: dict) -> None:
    print("\n3) WHERE THE TOKENS GO — per node (calls / tokens)\n")
    print(f" {'Node':<14}{'BASELINE':>22}{'OPTIMIZED':>22}")
    print(" " + "─" * 58)
    tags = sorted(set(base["by_tag"]) | set(opt["by_tag"]),
                  key=lambda t: -(base["by_tag"].get(t, {}).get("prompt", 0) + base["by_tag"].get(t, {}).get("completion", 0)))
    for t in tags:
        def cell(d):
            x = d.get(t)
            return f"{x['calls']:>3} / {x['prompt'] + x['completion']:>7,}" if x else f"{'-':>13}"
        print(f" {t:<14}{cell(base['by_tag']):>22}{cell(opt['by_tag']):>22}")


def print_guardrail(base: dict, opt: dict) -> None:
    print("\n4) QUALITY GUARD-RAIL — answer length in characters (a huge drop = over-trimmed)\n")
    print(f" {'#':>2}  {'Turn':<28}{'baseline':>9}{'optimized':>11}   ratio")
    print(" " + "─" * 62)
    warn = False
    for b, o in zip(base["turns"], opt["turns"]):
        lb, lo = len(b["answer"]), len(o["answer"])
        ratio = lo / lb if lb else 1
        flag = "  ⚠ check" if ratio < 0.5 else ""
        warn |= ratio < 0.5
        print(f" {b['id']:>2}  {b['label'][:27]:<28}{lb:>9,}{lo:>11,}   {ratio:4.0%}{flag}")
    if warn:
        print("\n   ⚠ Some answers are less than half as long. Re-run with --show-answers and read them:"
              "\n     if quality dropped, raise the matching cap in llm/optim.py (CAPS).")


def print_ablation(rows: list[dict], base_total: int) -> None:
    print("\n5) TECHNIQUE-BY-TECHNIQUE (cumulative: each row adds one technique to the rows above)\n")
    print(f" {'Enabled up to':<26}{'calls':>6}{'tokens':>10}{'this step':>11}{'vs base':>9}   ")
    print(" " + "─" * 96)
    prev = base_total
    for r in rows:
        t = r["totals"]["total_tokens"]
        step = prev - t
        print(f" {r['label']:<26}{r['totals']['calls']:>6}{t:>10,}{step:>+11,}{pct(base_total - t, base_total):>9}   "
              f"{bar(t / base_total, 30)}")
        prev = t
    print("\n   Reading rows that show 0 or a negative number:"
          "\n   • T2 saves retrieval work (DB / embedding lookups), not LLM tokens."
          "\n   • T5 adds one 'under N words' line to prompts; its payoff is shorter REAL completions"
          "\n     (not modelled by --mock) — measure it with your model."
          "\n   • T7 only matters when the critic rejects an answer and the router would otherwise call the LLM."
          "\n   • T9 only moves tokens to a cheaper model when OLLAMA_SMALL_MODEL (or GEMINI_SMALL_MODEL) is set.")


def print_answers(base: dict, opt: dict, ids=(1, 8)) -> None:
    print("\n6) SAMPLE ANSWERS\n")
    for b, o in zip(base["turns"], opt["turns"]):
        if b["id"] in ids:
            print(f"── turn {b['id']}: {b['label']} ──")
            print(f"[baseline · {len(b['answer'])} chars]\n{b['answer'][:600]}\n")
            print(f"[optimized · {len(o['answer'])} chars]\n{o['answer'][:600]}\n")


# ----------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.mock:
        os.environ["LLM_BACKEND"] = "mock"
    if args.estimate:
        os.environ["TOKEN_COUNT_MODE"] = "estimate"

    # imported AFTER env vars are set (llm.client reads them at import time)
    from llm import client, optim
    from llm.usage import tracker
    from memory import store
    from problems import bank
    from agents.graph import run

    preflight(client)

    scenarios = [s for s in SCENARIOS
                 if (client.LLM_BACKEND == "mock" or not s.get("mock_only"))
                 and (not args.quick or s["id"] in QUICK_IDS)]
    ctx = dict(optim=optim, tracker=tracker, client=client, store=store, bank=bank,
               run=run, trace=args.trace, tmp=tempfile.mkdtemp(prefix="dsa_demo_"))

    print_header(ctx, scenarios)

    print("\n▶ Pass 1/2  BASELINE  (all flags off) ...", flush=True)
    base = run_pass("baseline", lambda: optim.set_profile("baseline"), scenarios, ctx)
    print(f"  done — {base['totals']['calls']} calls, {base['totals']['total_tokens']:,} tokens, {base['totals']['wall_s']}s")
    print("▶ Pass 2/2  OPTIMIZED (all flags on) ...", flush=True)
    opt = run_pass("optimized", lambda: optim.set_profile("optimized"), scenarios, ctx)
    print(f"  done — {opt['totals']['calls']} calls, {opt['totals']['total_tokens']:,} tokens, {opt['totals']['wall_s']}s")

    for p in (base, opt):
        if p["errors"]:
            print(f"\n⚠ {p['name']}: backend errors, numbers are not meaningful.\n  {p['errors'][0]}")
    if base["totals"]["calls"] == 0:
        sys.exit("\nNo LLM calls were recorded — check the backend (or use --mock).")

    print_turns(base, opt)
    print_totals(base, opt)
    print_by_node(base, opt)
    print_guardrail(base, opt)

    ablation = []
    if args.ablation:
        print(f"\n▶ Ablation: {len(optim.FLAG_NAMES)} extra passes ...", flush=True)
        for i, flag in enumerate(optim.FLAG_NAMES, start=1):
            enabled = optim.FLAG_NAMES[:i]
            r = run_pass(f"abl{i}", lambda e=enabled: optim.set_only(e), scenarios, ctx)
            r["label"] = "+ " + optim.TECHNIQUES[flag]
            ablation.append(r)
            print(f"  {r['label']:<28} {r['totals']['total_tokens']:>8,} tokens", flush=True)
        print_ablation(ablation, base["totals"]["total_tokens"])

    if args.show_answers:
        print_answers(base, opt)

    if client.LLM_BACKEND == "mock":
        print("\nNOTE  mock backend: prompt sizes and call counts are real (built by the real nodes);\n"
              "      completions are canned and identical in both passes, so output-token savings\n"
              "      from T5 are NOT shown here. Run without --mock against Ollama/Gemini to measure them.")
    else:
        print("\nNOTE  LLM outputs vary run to run (and the critic may or may not ask for a retry), so\n"
              "      treat small differences as noise; call counts and prompt tokens are the stable signal.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"backend": client.LLM_BACKEND, "baseline": base, "optimized": opt, "ablation": ablation},
            indent=2, default=str), encoding="utf-8")
        print(f"\nResults written to {args.json}")

    optim.set_profile("optimized")   # leave the process in the default state


if __name__ == "__main__":
    main()
