"""Tiny JSON HTTP API (standard library only).

This is how the rest of the AI family — and later the Family Hub forum —
talks to Scholar: structured requests in, structured results out.
Set SCHOLAR_API_TOKEN to require ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agent import CARD
from .assistant import Assistant
from .db import Store
from .llm import LLMError
from .research import DEPTHS, Researcher


def make_handler(store: Store, assistant: Assistant | None, researcher: Researcher | None,
                 token: str = ""):
    lock = threading.Lock()  # one SQLite connection / one line of thought at a time

    class Handler(BaseHTTPRequestHandler):
        server_version = "Scholar/0.1"

        def log_message(self, fmt, *args):  # quieter default logging
            pass

        def _send(self, status: int, body) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not token:
                return True
            if self.headers.get("Authorization", "") == f"Bearer {token}":
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            return body

        def do_GET(self):
            if not self._authorized():
                return
            url = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            with lock:
                if url.path == "/api/health":
                    return self._send(200, {"ok": True, "agent": CARD["id"]})
                if url.path == "/api/agent":
                    return self._send(200, {**CARD, "library": store.stats()})
                if url.path == "/api/findings":
                    return self._send(200, store.findings(qs.get("q", ""), int(qs.get("limit", 20)),
                                                          qs.get("status")))
                if m := re.fullmatch(r"/api/findings/(\d+)", url.path):
                    f = store.get_finding(int(m.group(1)))
                    if f:
                        f["history"] = store.finding_history(f["id"])
                        f["sources"] = [_brief(store.get_source(i)) for i in f["source_ids"]]
                    return self._send(200 if f else 404, f or {"error": "not found"})
                if url.path == "/api/sources":
                    return self._send(200, [_brief(s) for s in store.sources(qs.get("q", ""))])
                if url.path == "/api/reports":
                    return self._send(200, store.reports())
                if m := re.fullmatch(r"/api/reports/(\d+)", url.path):
                    r = store.get_report(int(m.group(1)))
                    return self._send(200 if r else 404, r or {"error": "not found"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
            except ValueError:
                return self._send(400, {"error": "invalid JSON"})
            sender = str(body.get("from", ""))
            with lock:
                try:
                    if path == "/api/ask" and body.get("text"):
                        if assistant is None:
                            return self._send(503, {"error": "assistant not configured"})
                        return self._send(200, {"from": "scholar", "reply": assistant.ask(body["text"], sender)})
                    if path in ("/api/research", "/api/plan") and (body.get("question") or body.get("idea")):
                        if researcher is None:
                            return self._send(503, {"error": "researcher not configured"})
                        depth = body.get("depth", "standard")
                        if depth not in DEPTHS:
                            return self._send(400, {"error": f"depth must be one of {', '.join(DEPTHS)}"})
                        mode = "plan" if path == "/api/plan" else "research"
                        r = researcher.investigate(body.get("question") or body["idea"], depth, mode)
                        return self._send(200, {"from": "scholar", "report_id": r["report_id"],
                                                "markdown": r["markdown"], "data": r["data"]})
                except LLMError as e:
                    return self._send(502, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return Handler


def _brief(s: dict | None) -> dict | None:
    if s is None:
        return None
    return {k: s[k] for k in ("id", "title", "url", "domain", "kind", "tier", "published", "fetched_at")}


def serve(store: Store, assistant: Assistant | None, researcher: Researcher | None, host: str, port: int,
          token: str = "") -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(store, assistant, researcher, token))
    print(f"Scholar API on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
