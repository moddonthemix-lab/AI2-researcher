"""Minimal Ollama client (standard library only).

Ollama runs the model locally on the MacBook (later the Pi), so there is
no API key and no monthly bill.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str, model: str, timeout: float = 600):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama returned {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"Can't reach Ollama at {self.url} ({e.reason}). Is it running? Try `ollama serve`."
            ) from e

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> dict:
        """Return the assistant message dict (``content`` and maybe ``tool_calls``)."""
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.2}}  # a researcher should be steady, not creative
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"
        return self._post("/api/chat", payload)["message"]

    def available(self) -> bool:
        try:
            req = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model for n in names)
        except (urllib.error.URLError, OSError, ValueError):
            return False


def parse_json(text: str) -> dict:
    """Best-effort JSON from a model reply (tolerates code fences and chatter)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}
