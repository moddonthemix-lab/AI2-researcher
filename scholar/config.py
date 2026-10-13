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
    # OpenAlex (250M+ scholarly works) wants a free API key: https://openalex.org/settings/api
    openalex_key: str = field(default_factory=lambda: _env("SCHOLAR_OPENALEX_KEY"))
    # SEC EDGAR asks every automated client to identify itself with contact info.
    contact: str = field(default_factory=lambda: _env("SCHOLAR_CONTACT"))
    # Semantic memory: embeddings from Ollama, stored in Qdrant if configured, else in scholar.db.
    embed_model: str = field(default_factory=lambda: _env("SCHOLAR_EMBED_MODEL", "nomic-embed-text"))
    qdrant_url: str = field(default_factory=lambda: _env("SCHOLAR_QDRANT_URL"))
    qdrant_api_key: str = field(default_factory=lambda: _env("SCHOLAR_QDRANT_API_KEY"))
    # Python sandbox for quantitative analysis
    sandbox_timeout: int = field(default_factory=lambda: int(_env("SCHOLAR_SANDBOX_TIMEOUT", "60")))
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

    @property
    def documents_dir(self) -> Path:
        """Downloaded originals (PDFs, filings) and the tables extracted from them."""
        return self.home / "documents"

    @property
    def analysis_dir(self) -> Path:
        """One folder per sandboxed Python run: the code, its inputs and its outputs."""
        return self.home / "analysis"

    def ensure_home(self) -> None:
        for d in (self.reports_dir, self.documents_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
