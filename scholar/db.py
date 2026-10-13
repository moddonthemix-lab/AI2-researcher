"""SQLite storage: Scholar's research library.

One file (``~/.scholar/scholar.db``) holds everything — memory, every source
he has read, every finding with its evidence and history, and his reports.
Moving him to the Pi is just copying that file. Standard library only.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'note',    -- note | preference | fact
    topic       TEXT NOT NULL DEFAULT 'general',
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, topic, content='memories', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, topic) VALUES (new.id, new.content, new.topic);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, topic) VALUES ('delete', old.id, old.content, old.topic);
END;

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    title       TEXT,
    domain      TEXT,
    kind        TEXT NOT NULL DEFAULT 'web',     -- web | paper | reference | onion
    tier        INTEGER NOT NULL DEFAULT 3,      -- 1 primary/peer-reviewed ... 4 unvetted
    published   TEXT,
    text        TEXT,                            -- readable text, truncated
    fetched_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
    title, text, content='sources', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS sources_ai AFTER INSERT ON sources BEGIN
    INSERT INTO sources_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS sources_au AFTER UPDATE ON sources BEGIN
    INSERT INTO sources_fts(sources_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text);
    INSERT INTO sources_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;

CREATE TABLE IF NOT EXISTS findings (
    id          INTEGER PRIMARY KEY,
    claim       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'assumption',
    confidence  REAL NOT NULL DEFAULT 0.5,       -- 0..1
    topic       TEXT NOT NULL DEFAULT 'general',
    source_ids  TEXT NOT NULL DEFAULT '[]',      -- JSON list of sources.id
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
    claim, topic, notes, content='findings', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(rowid, claim, topic, notes) VALUES (new.id, new.claim, new.topic, new.notes);
END;
CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, claim, topic, notes)
        VALUES ('delete', old.id, old.claim, old.topic, old.notes);
    INSERT INTO findings_fts(rowid, claim, topic, notes) VALUES (new.id, new.claim, new.topic, new.notes);
END;

-- Every change of a conclusion is kept, so Scholar can say *why* he changed his mind.
CREATE TABLE IF NOT EXISTS finding_history (
    id          INTEGER PRIMARY KEY,
    finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    status      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    changed_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY,
    question    TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'research', -- research | plan
    body        TEXT NOT NULL,                    -- markdown
    data        TEXT NOT NULL DEFAULT '{}',       -- structured JSON behind the markdown
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
"""

# How sure Scholar is allowed to sound. Assigned by code from the evidence, not by the model.
STATUSES = ("verified", "reported", "assumption", "disputed", "refuted")


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def _fts_query(text: str) -> str:
    """Turn free text into a forgiving FTS5 query (any word, prefix match)."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]
    return " OR ".join(f'"{w}"*' for w in words)


def _finding(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d["source_ids"] = json.loads(d["source_ids"] or "[]")
    return d


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _insert(self, sql: str, params: tuple) -> int:
        with self.conn:
            return self.conn.execute(sql, params).lastrowid

    def _search(self, table: str, query: str, limit: int, extra_sql: str = "",
                extra: list | None = None) -> list[dict]:
        q, extra = _fts_query(query), extra or []
        if q:
            sql = f"""SELECT t.* FROM {table}_fts f JOIN {table} t ON t.id = f.rowid
                      WHERE {table}_fts MATCH ? {extra_sql} ORDER BY bm25({table}_fts) LIMIT ?"""
            return _rows(self.conn.execute(sql, [q, *extra, limit]))
        sql = f"SELECT t.* FROM {table} t WHERE 1=1 {extra_sql} ORDER BY t.id DESC LIMIT ?"
        return _rows(self.conn.execute(sql, [*extra, limit]))

    # ------------------------------------------------------------------ memory
    def remember(self, content: str, topic: str = "general", kind: str = "note") -> int:
        return self._insert(
            "INSERT INTO memories(content, topic, kind) VALUES (?, ?, ?)", (content, topic, kind)
        )

    def forget(self, memory_id: int) -> bool:
        with self.conn:
            return self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0

    def recall(self, query: str = "", limit: int = 8) -> list[dict]:
        return self._search("memories", query, limit)

    # ----------------------------------------------------------------- sources
    def add_source(self, url: str, title: str = "", domain: str = "", kind: str = "web",
                   tier: int = 3, text: str = "", published: str = "") -> int:
        """Insert or refresh a source; returns its id (stable per URL)."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO sources(url, title, domain, kind, tier, text, published)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(url) DO UPDATE SET
                     title = COALESCE(NULLIF(excluded.title, ''), title),
                     text = CASE WHEN length(excluded.text) > length(COALESCE(text, ''))
                                 THEN excluded.text ELSE text END,
                     published = COALESCE(NULLIF(excluded.published, ''), published),
                     fetched_at = datetime('now','localtime')""",
                (url, title, domain, kind, tier, text, published),
            )
        return self.conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()[0]

    def get_source(self, source_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return dict(row) if row else None

    def sources(self, query: str = "", limit: int = 20) -> list[dict]:
        return self._search("sources", query, limit)

    # ---------------------------------------------------------------- findings
    def add_finding(self, claim: str, status: str = "assumption", confidence: float = 0.5,
                    topic: str = "general", source_ids: list[int] | None = None,
                    notes: str = "") -> int:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        fid = self._insert(
            """INSERT INTO findings(claim, status, confidence, topic, source_ids, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (claim, status, _clamp(confidence), topic, json.dumps(sorted(set(source_ids or []))), notes),
        )
        self._history(fid, status, _clamp(confidence), "first recorded")
        return fid

    def update_finding(self, finding_id: int, status: str | None = None,
                       confidence: float | None = None, add_source_ids: list[int] | None = None,
                       reason: str = "") -> dict | None:
        """Revise a conclusion when new evidence appears; the old state goes to history."""
        f = self.get_finding(finding_id)
        if f is None:
            return None
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        new_status = status or f["status"]
        new_conf = f["confidence"] if confidence is None else _clamp(confidence)
        ids = sorted(set(f["source_ids"]) | set(add_source_ids or []))
        with self.conn:
            self.conn.execute(
                """UPDATE findings SET status = ?, confidence = ?, source_ids = ?,
                     notes = TRIM(notes || ' ' || ?), updated_at = datetime('now','localtime')
                   WHERE id = ?""",
                (new_status, new_conf, json.dumps(ids), reason, finding_id),
            )
        self._history(finding_id, new_status, new_conf, reason)
        return self.get_finding(finding_id)

    def _history(self, finding_id: int, status: str, confidence: float, reason: str) -> None:
        self._insert(
            "INSERT INTO finding_history(finding_id, status, confidence, reason) VALUES (?, ?, ?, ?)",
            (finding_id, status, confidence, reason),
        )

    def get_finding(self, finding_id: int) -> dict | None:
        return _finding(self.conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone())

    def findings(self, query: str = "", limit: int = 20, status: str | None = None) -> list[dict]:
        extra_sql, extra = ("AND t.status = ?", [status]) if status else ("", [])
        rows = self._search("findings", query, limit, extra_sql, extra)
        for r in rows:
            r["source_ids"] = json.loads(r["source_ids"] or "[]")
        return rows

    def finding_history(self, finding_id: int) -> list[dict]:
        return _rows(self.conn.execute(
            "SELECT * FROM finding_history WHERE finding_id = ? ORDER BY id", (finding_id,)
        ))

    # ----------------------------------------------------------------- reports
    def add_report(self, question: str, body: str, mode: str = "research", data: dict | None = None) -> int:
        return self._insert(
            "INSERT INTO reports(question, mode, body, data) VALUES (?, ?, ?, ?)",
            (question, mode, body, json.dumps(data or {}, default=str)),
        )

    def get_report(self, report_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["data"] = json.loads(d["data"] or "{}")
        return d

    def reports(self, limit: int = 20) -> list[dict]:
        return _rows(self.conn.execute(
            "SELECT id, question, mode, created_at FROM reports ORDER BY id DESC LIMIT ?", (limit,)
        ))

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        one = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        by_status = dict(self.conn.execute(
            "SELECT status, COUNT(*) FROM findings GROUP BY status").fetchall())
        return {
            "memories": one("SELECT COUNT(*) FROM memories"),
            "sources": one("SELECT COUNT(*) FROM sources"),
            "findings": one("SELECT COUNT(*) FROM findings"),
            "findings_by_status": {s: by_status.get(s, 0) for s in STATUSES},
            "reports": one("SELECT COUNT(*) FROM reports"),
        }


def _clamp(x: float) -> float:
    return round(min(1.0, max(0.0, float(x))), 2)
