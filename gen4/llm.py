"""
Model layer — provider-agnostic LLM client with an honest offline fallback.

    provider = "auto"  -> anthropic if ANTHROPIC_API_KEY, else openai if
                          OPENAI_API_KEY, else offline
    provider = "offline" -> never calls the network

Every call returns (text, provider_used) through `.last_provider`, and the
services put that in their trace -- so an output produced by the offline
fallback is never mistaken for a live model's output.

The key design rule: the LLM is used for *language* (explanations,
classifying free text, extracting fields, judging open answers), never as
the sole authority on a hard policy rule. Every call site passes an
`offline` function that produces a safe, deterministic answer, which is
also what runs if the live call fails or returns unparseable JSON.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Callable


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 max_retries: int = 2, timeout: float = 30.0):
        requested = (provider or os.environ.get("GEN4_LLM_PROVIDER", "auto")).lower()
        if requested == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                requested = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                requested = "openai"
            else:
                requested = "offline"
        if requested == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            requested = "offline"
        if requested == "openai" and not os.environ.get("OPENAI_API_KEY"):
            requested = "offline"
        self.provider = requested
        self.model = model or os.environ.get("GEN4_LLM_MODEL") or {
            "anthropic": "claude-haiku-4-5",
            "openai": "gpt-4o-mini",
        }.get(requested, "offline")
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_provider = self.provider
        self.last_error: str | None = None

    @property
    def is_live(self) -> bool:
        return self.provider != "offline"

    # -- public API ----------------------------------------------------------
    def complete(self, prompt: str, offline: Callable[[], str], system: str = "",
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        if not self.is_live:
            self.last_provider = "offline"
            return offline()
        try:
            text = self._with_retries(lambda: self._call(prompt, system, max_tokens, temperature))
            self.last_provider = self.provider
            return text
        except Exception as e:  # network, auth, rate limit -- degrade, don't crash
            self.last_error = str(e)[:300]
            self.last_provider = "offline(fallback)"
            return offline()

    def complete_json(self, prompt: str, offline: Callable[[], dict], system: str = "",
                      max_tokens: int = 512, validate: Callable[[dict], bool] | None = None) -> dict:
        """Ask for JSON; if the live answer is missing, unparseable, or fails
        `validate`, use the offline answer instead (and record why)."""
        if not self.is_live:
            self.last_provider = "offline"
            return offline()
        sys_prompt = (system + "\n" if system else "") + "Respond with a single JSON object only."
        raw = self.complete(prompt, offline=lambda: "", system=sys_prompt, max_tokens=max_tokens)
        parsed = _extract_json(raw)
        if parsed is None or (validate and not validate(parsed)):
            self.last_error = self.last_error or "unparseable or invalid JSON from model"
            self.last_provider = "offline(fallback)"
            return offline()
        return parsed

    def describe_image(self, image_url: str, prompt: str, offline: Callable[[], str],
                       max_tokens: int = 200) -> str:
        """Vision call (Anthropic only). Falls back to `offline` otherwise."""
        if self.provider != "anthropic":
            self.last_provider = "offline"
            return offline()
        body = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": image_url}},
            {"type": "text", "text": prompt}]}]}
        try:
            data = self._with_retries(lambda: _post(
                "https://api.anthropic.com/v1/messages", body,
                {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                self.timeout))
            self.last_provider = self.provider
            return "".join(b.get("text", "") for b in data.get("content", [])).strip()
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)[:300]
            self.last_provider = "offline(fallback)"
            return offline()

    # -- transport -----------------------------------------------------------
    def _with_retries(self, fn):
        err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(0.5 * (2 ** attempt))
        raise err  # type: ignore[misc]

    def _call(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        if self.provider == "anthropic":
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]}
            if system:
                body["system"] = system
            data = _post("https://api.anthropic.com/v1/messages", body, {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01"}, self.timeout)
            return "".join(b.get("text", "") for b in data.get("content", [])).strip()
        if self.provider == "openai":
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            data = _post("https://api.openai.com/v1/chat/completions",
                         {"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                          "temperature": temperature},
                         {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, self.timeout)
            return data["choices"][0]["message"]["content"].strip()
        raise ValueError(f"unknown provider {self.provider}")


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _extract_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None
    return None
