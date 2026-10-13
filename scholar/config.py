"""Runtime configuration, read from environment variables.

Everything has a sane default so `scholar chat` works out of the box on a
MacBook running Ollama. The same settings carry over to the Raspberry Pi
later — usually only the model name changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _default_home() -> Path:
    return Path(_env("SCHOLAR_HOME", str(Path.home() / ".scholar"))).expanduser()


@dataclass
class Config:
    home: Path = field(default_factory=_default_home)
    ollama_url: str = field(default_factory=lambda: _env("SCHOLAR_OLLAMA_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _env("SCHOLAR_MODEL", "qwen2.5:7b"))
    assistant_name: str = field(default_factory=lambda: _env("SCHOLAR_NAME", "Scholar"))
    user_name: str = field(default_factory=lambda: _env("SCHOLAR_USER", "boss"))
    # Search: a self-hosted SearXNG is the most private and reliable; DuckDuckGo is the fallback.
    searxng_url: str = field(default_factory=lambda: _env("SCHOLAR_SEARXNG_URL"))
    # Optional Tor SOCKS5 proxy (e.g. 127.0.0.1:9050) — enables reading .onion sources.
    tor_proxy: str = field(default_factory=lambda: _env("SCHOLAR_TOR_PROXY"))
    # HTTP API (how the rest of the family will reach Scholar)
    api_host: str = field(default_factory=lambda: _env("SCHOLAR_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(_env("SCHOLAR_API_PORT", "8766")))
    api_token: str = field(default_factory=lambda: _env("SCHOLAR_API_TOKEN"))

    @property
    def db_path(self) -> Path:
        return self.home / "scholar.db"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    def ensure_home(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
