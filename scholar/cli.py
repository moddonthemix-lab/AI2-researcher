"""Command line: talk to Scholar, run research, or browse his library directly."""

from __future__ import annotations

import argparse
import json
import sys

from . import calc
from .agent import CARD
from .assistant import Assistant
from .config import Config
from .credibility import TIER_LABELS
from .db import STATUSES, Store
from .llm import LLMError, Ollama
from .research import DEPTHS, Researcher, to_json
from .tools import Toolbox
from .web import Web, WebError


def _print_rows(rows: list[dict], cols: list[str], width: int = 70) -> None:
    if not rows:
        print("(none)")
        return

    def cell(r, c):
        v = r.get(c)
        v = "" if v is None else ", ".join(f"S{i}" for i in v) if isinstance(v, list) else str(v)
        return v if len(v) <= width else v[:width - 1] + "…"

    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(cell(r, c).ljust(widths[c]) for c in cols))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scholar", description="Scholar — the Researcher of your AI family.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("chat", help="Talk to Scholar in the terminal")
    for name, helptext in (("research", "Investigate a question and write a cited report"),
                           ("plan", "Turn a business idea into a researched plan with numbers")):
        r = sub.add_parser(name, help=helptext)
        r.add_argument("text", nargs="+")
        r.add_argument("--depth", choices=list(DEPTHS), default="standard")
        r.add_argument("--json", action="store_true", help="print the structured data instead of markdown")

    s = sub.add_parser("search", help="Search without the model (web, papers or wiki)")
    s.add_argument("query", nargs="+")
    s.add_argument("--in", dest="where", choices=["web", "papers", "wiki"], default="web")
    s.add_argument("--limit", type=int, default=8)
    s = sub.add_parser("read", help="Read a URL into the library and print its text")
    s.add_argument("url")
    s.add_argument("--chars", type=int, default=3000)

    f = sub.add_parser("finding", help="Findings in the library").add_subparsers(dest="action", required=True)
    a = f.add_parser("list")
    a.add_argument("query", nargs="*")
    a.add_argument("--status", choices=STATUSES)
    a = f.add_parser("show")
    a.add_argument("id", type=int)
    a = f.add_parser("add")
    a.add_argument("claim", nargs="+")
    a.add_argument("--status", choices=STATUSES, default="assumption")
    a.add_argument("--confidence", type=float, default=0.5)
    a.add_argument("--topic", default="general")
    a.add_argument("--source", type=int, action="append", default=[], help="source id (repeatable)")
    a = f.add_parser("update")
    a.add_argument("id", type=int)
    a.add_argument("--status", choices=STATUSES)
    a.add_argument("--confidence", type=float)
    a.add_argument("--source", type=int, action="append", default=[])
    a.add_argument("--reason", required=True)

    s = sub.add_parser("sources", help="Sources Scholar has read")
    s.add_argument("query", nargs="*")
    sub.add_parser("reports", help="List research reports")
    s = sub.add_parser("report", help="Print a report")
    s.add_argument("id", type=int)

    r = sub.add_parser("remember", help="Save a note")
    r.add_argument("text", nargs="+")
    r.add_argument("--topic", default="general")
    r = sub.add_parser("recall", help="Search notes")
    r.add_argument("query", nargs="*")
    r = sub.add_parser("forget", help="Delete a note by id")
    r.add_argument("id", type=int)

    c = sub.add_parser("calc", help="Exact arithmetic, e.g. scholar calc '(320-210)/320*100'")
    c.add_argument("expression", nargs="+")
    e = sub.add_parser("economics", help="Unit economics for a product")
    e.add_argument("--price", type=float, required=True)
    e.add_argument("--cost", type=float, required=True)
    e.add_argument("--fees-pct", type=float, default=0)
    e.add_argument("--fee-fixed", type=float, default=0)
    e.add_argument("--shipping", type=float, default=0)
    e.add_argument("--fixed", type=float, default=0, help="fixed costs per month")

    sub.add_parser("card", help="Scholar's identity card and library stats")
    s = sub.add_parser("serve", help="Run the JSON API for the rest of the family")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    sub.add_parser("doctor", help="Check Ollama, the model and internet access")
    return p


def chat_loop(assistant: Assistant) -> None:
    name = assistant.config.assistant_name
    print(f"{name} is at his desk. /reset clears the conversation, /quit exits.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return
        if text == "/reset":
            assistant.reset()
            print("(conversation cleared — the library is kept)")
            continue
        try:
            print(f"\n{name.lower()}> {assistant.ask(text)}")
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)


def _progress(msg: str) -> None:
    print(f"  · {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()
    config.ensure_home()
    db = Store(config.db_path)
    web = Web(config.searxng_url, config.tor_proxy)
    tb = Toolbox(db, web)
    llm = Ollama(config.ollama_url, config.model)
    cmd, action = args.cmd, getattr(args, "action", None)

    try:
        if cmd in (None, "chat"):
            chat_loop(Assistant(tb, llm, config))
        elif cmd in ("research", "plan"):
            result = Researcher(tb, llm, config, log=_progress).investigate(
                " ".join(args.text), args.depth, cmd)
            print(to_json(result) if args.json else result["markdown"])
            if result["path"]:
                print(f"(saved as report #{result['report_id']}: {result['path']})", file=sys.stderr)
        elif cmd == "search":
            q = " ".join(args.query)
            fn = {"web": web.search, "papers": web.papers, "wiki": web.wikipedia}[args.where]
            for r in fn(q, args.limit):
                sid = db.add_source(r["url"], r.get("title", ""), r.get("domain", ""),
                                    {"web": "web", "papers": "paper", "wiki": "reference"}[args.where],
                                    r.get("tier", 3), r.get("snippet", ""), r.get("published", ""))
                print(f"[S{sid}] tier {r['tier']}  {r.get('title', '')}\n      {r['url']}")
                if r.get("snippet"):
                    print(f"      {r['snippet'][:200]}")
        elif cmd == "read":
            doc = web.read(args.url)
            sid = db.add_source(doc["url"], doc["title"], doc["domain"], doc["kind"], doc["tier"],
                                doc["text"], doc["published"])
            print(f"[S{sid}] {doc['title']}\n{doc['url']}\ntier {doc['tier']}: {doc['tier_label']}"
                  + (f" · published {doc['published']}" if doc["published"] else "") + "\n")
            print(doc["text"][:args.chars])
        elif cmd == "finding":
            if action == "list":
                _print_rows(db.findings(" ".join(args.query), 50, args.status),
                            ["id", "status", "confidence", "claim", "source_ids", "updated_at"])
            elif action == "show":
                f = db.get_finding(args.id)
                if not f:
                    print("not found")
                    return 1
                print(f"F{f['id']}  [{f['status']}, {f['confidence']:.0%}]  {f['claim']}\n")
                for i in f["source_ids"]:
                    s = db.get_source(i)
                    if s:
                        print(f"  [S{i}] tier {s['tier']}  {s['title']}  {s['url']}")
                print("\nHistory:")
                for h in db.finding_history(f["id"]):
                    print(f"  {h['changed_at']}  {h['status']:<10} {h['confidence']:.0%}  {h['reason']}")
            elif action == "add":
                print(f"finding #{db.add_finding(' '.join(args.claim), args.status, args.confidence, args.topic, args.source)}")
            else:
                f = db.update_finding(args.id, args.status, args.confidence, args.source, args.reason)
                print(f"F{f['id']} is now {f['status']} ({f['confidence']:.0%})" if f else "not found")
        elif cmd == "sources":
            _print_rows(db.sources(" ".join(args.query), 50), ["id", "tier", "domain", "title", "fetched_at"])
        elif cmd == "reports":
            _print_rows(db.reports(50), ["id", "mode", "created_at", "question"])
        elif cmd == "report":
            r = db.get_report(args.id)
            print(r["body"] if r else "not found")
        elif cmd == "remember":
            print(f"saved #{db.remember(' '.join(args.text), args.topic)}")
        elif cmd == "recall":
            _print_rows(db.recall(" ".join(args.query), limit=20), ["id", "topic", "created_at", "content"])
        elif cmd == "forget":
            print("deleted" if db.forget(args.id) else "not found")
        elif cmd == "calc":
            print(calc.calculate(" ".join(args.expression)))
        elif cmd == "economics":
            print(json.dumps(calc.unit_economics(args.price, args.cost, args.fees_pct, args.fee_fixed,
                                                 args.shipping, args.fixed), indent=2))
        elif cmd == "card":
            print(json.dumps({**CARD, "library": db.stats()}, indent=2))
        elif cmd == "serve":
            from .server import serve
            serve(db, Assistant(tb, llm, config), Researcher(tb, llm, config, log=_progress),
                  args.host or config.api_host, args.port or config.api_port, config.api_token)
        elif cmd == "doctor":
            return doctor(config, llm, web)
    except (LLMError, WebError, ValueError, SyntaxError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def doctor(config: Config, llm: Ollama, web: Web) -> int:
    ok = True
    print(f"library: {config.db_path}")
    print(f"reports: {config.reports_dir}")
    print(f"model:   {config.model} via {config.ollama_url}")
    if llm.available():
        print("  ✓ Ollama is reachable and the model is installed")
    else:
        ok = False
        print(f"  ✗ not ready — install Ollama (https://ollama.com), then: ollama pull {config.model}")
    print(f"search:  {'SearXNG at ' + config.searxng_url if config.searxng_url else 'DuckDuckGo (set SCHOLAR_SEARXNG_URL for your own SearXNG)'}")
    for label, fn in (("web search", lambda: web.search("open source research tools", 3)),
                      ("papers", lambda: web.papers("information retrieval", 2)),
                      ("wikipedia", lambda: web.wikipedia("Library science", 1))):
        try:
            print(f"  {'✓' if fn() else '?'} {label}")
        except WebError as e:
            print(f"  ✗ {label}: {e}")
    if config.tor_proxy:
        print(f"tor:     SOCKS proxy {config.tor_proxy} (.onion sources: tier 4 — {TIER_LABELS[4]})")
    else:
        print("tor:     off (set SCHOLAR_TOR_PROXY=127.0.0.1:9050 to let Scholar read .onion sources)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
