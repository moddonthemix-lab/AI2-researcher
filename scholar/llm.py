"""Minimal Ollama client (standard library only).

Ollama runs the model locally on the MacBook (later the Pi), so there is
no API key and no monthly bill.

Three things here matter a lot on CPU-only machines (Intel Macs, Raspberry Pi):

- ``schema``: Ollama constrains the model's output to an exact JSON schema, so even a
  small model can't return malformed or wrongly-shaped answers.
- ``think``: reasoning models (Qwen3, DeepSeek-R1, gpt-oss) "think out loud" before
  answering. That can be thousands of extra tokens — minutes on a CPU — so it is off
  for routine reading and only used where judgement matters (``SCHOLAR_THINK``).
- ``stats``: prompt/answer token counts and timings for every call, so you can see
  where the time goes (``scholar bench`` uses them).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

# Families that support turning thinking off with think=false.
_THINK_BOOL = re.compile(r"^(qwen3|qwen3\.\d|deepseek-r1|magistral|cogito)", re.I)
# gpt-oss can't turn thinking off, only down.
_THINK_LEVEL = re.compile(r"^gpt-oss", re.I)


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str, model: str, timeout: float = 900, num_ctx: int = 8192,
                 think: bool = False, max_output: int = 2500):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Ollama's default context is only 2-4k tokens and it silently drops the start of longer
        # prompts — ask for a real window. Keep it fixed per model: changing it reloads the model.
        self.num_ctx = num_ctx
        self.think = think
        self.max_output = max_output
        self.last: dict = {}
        self.totals = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        self._think_supported = True

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
        except (TimeoutError, OSError) as e:
            raise LLMError(f"Ollama took longer than {self.timeout:.0f}s to answer ({e}). A smaller --depth "
                           "or a faster model helps; raise SCHOLAR_LLM_TIMEOUT if your machine is slow.") from e

    def _think_value(self, want: bool | None):
        want = self.think if want is None else want
        if not self._think_supported:
            return None
        if _THINK_LEVEL.match(self.model):
            return "medium" if want else "low"
        if _THINK_BOOL.match(self.model):
            return bool(want)
        return None  # model doesn't think; don't send the flag

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, schema: dict | None = None, think: bool | None = None) -> dict:
        """Return the assistant message dict (``content`` and maybe ``tool_calls``).

        ``schema`` (a JSON schema) constrains the answer exactly; ``json_mode`` only asks for some JSON.
        """
        options = {"temperature": 0.2, "num_ctx": self.num_ctx}  # a researcher should be steady, not creative
        structured = json_mode or schema is not None
        if structured:
            options["num_predict"] = self.max_output  # a stuck model can't ramble forever
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if tools:
            payload["tools"] = tools
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        t = self._think_value(think)
        if t is not None:
            payload["think"] = t
        start = time.monotonic()
        try:
            data = self._post("/api/chat", payload)
        except LLMError as e:
            if "think" in payload and "think" in str(e).lower():  # model/version without thinking control
                self._think_supported = False
                payload.pop("think")
                data = self._post("/api/chat", payload)
            else:
                raise
        self._record(data, time.monotonic() - start)
        return data["message"]

    def _record(self, data: dict, wall: float) -> None:
        ns = 1e9
        self.last = {
            "model": self.model,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "load_s": round(data.get("load_duration", 0) / ns, 2),
            "read_s": round(data.get("prompt_eval_duration", 0) / ns, 2),
            "write_s": round(data.get("eval_duration", 0) / ns, 2),
            "seconds": round(wall, 2),
        }
        self.last["read_tps"] = round(self.last["prompt_tokens"] / self.last["read_s"], 1) if self.last["read_s"] else None
        self.last["write_tps"] = round(self.last["output_tokens"] / self.last["write_s"], 1) if self.last["write_s"] else None
        self.totals["calls"] += 1
        self.totals["prompt_tokens"] += self.last["prompt_tokens"]
        self.totals["output_tokens"] += self.last["output_tokens"]
        self.totals["seconds"] = round(self.totals["seconds"] + wall, 1)

    def available(self) -> bool:
        try:
            req = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model or n == self.model + ":latest"
                       for n in names)
        except (urllib.error.URLError, OSError, ValueError):
            return False


def parse_json(text: str) -> dict:
    """Best-effort JSON from a model reply (tolerates code fences, chatter and stray think tags)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                out = json.loads(text[start:end + 1])
                return out if isinstance(out, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def estimate_tokens(text: str) -> int:
    """Rough token count (~3.6 characters per token for English prose)."""
    return int(len(text) / 3.6) + 1
