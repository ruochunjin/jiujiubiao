"""LLM interface for JiuJiuBiao: protocol, stdlib OpenAI-compatible client, FakeLLM.

Configuration is environment-driven (no SDK dependency — plain HTTP):

- ``JIUJIUBIAO_API_KEY`` / ``OPENAI_API_KEY``: API key
- ``JIUJIUBIAO_BASE_URL`` / ``OPENAI_BASE_URL``: OpenAI-compatible endpoint
  (works with vLLM, Ollama, DeepSeek, ...); defaults to OpenAI
- ``JIUJIUBIAO_MODEL`` / ``OPENAI_MODEL``: model name
- ``JIUJIUBIAO_TIMEOUT`` / ``JIUJIUBIAO_RETRIES``: request tuning

Nothing is created at import time; pass any object implementing ``LLM``
(``chat`` + ``sample``) to the agents — including ``FakeLLM`` for tests.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _anthropic_credentials() -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model) for the Anthropic-messages backend.

    Environment first (``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_MODEL``); falls back to the Claude Code settings file's
    ``env`` block so a K3-style endpoint configured for Claude Code works
    in-process without exporting anything.  Never logged or printed.
    """

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL")
    if base_url and api_key and model:
        return base_url, api_key, model
    settings = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings, encoding="utf-8-sig") as handle:
            env = json.load(handle).get("env", {})
        base_url = base_url or env.get("ANTHROPIC_BASE_URL")
        api_key = api_key or env.get("ANTHROPIC_API_KEY")
        model = model or env.get("ANTHROPIC_MODEL")
    except (OSError, json.JSONDecodeError):
        pass
    if not (base_url and api_key and model):
        raise RuntimeError(
            "Anthropic backend needs ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY/ANTHROPIC_MODEL"
        )
    return base_url, api_key, model


class AnthropicClient:
    """Stdlib-only Anthropic messages client (e.g. a K3 coding endpoint).

    Implements the same ``LLM`` protocol as :class:`ChatClient`.  Extended
    thinking is explicitly disabled — the endpoint defaults to it, which
    eats the whole token budget on small requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        max_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        if base_url and api_key and model:
            resolved = (base_url, api_key, model)
        else:
            resolved = _anthropic_credentials()
        self.base_url, self.api_key, self.model = resolved
        self.url = self.base_url.rstrip("/") + "/v1/messages"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = float(os.environ.get("JIUJIUBIAO_TIMEOUT", "180"))
        self.retries = int(os.environ.get("JIUJIUBIAO_RETRIES", "6"))
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.verbose = verbose
        self.n_calls = 0
        self.n_tokens = 0

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "thinking": {"type": "disabled"},
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop_sequences"] = stop
        data = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                self.n_calls += 1
                usage = payload.get("usage") or {}
                self.n_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                return "".join(
                    block.get("text", "")
                    for block in payload.get("content", [])
                    if block.get("type") == "text"
                )
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                last = error
                # Rate-limit/quota statuses get real backoff with jitter;
                # a fixed 2-6s ladder dies under sustained load (TabFact-100).
                status = getattr(error, "code", None)
                if status in (403, 429, 500, 502, 503, 529):
                    time.sleep(min(90.0, 5.0 * (2 ** attempt)) + random.random())
                else:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} retries: {last}")

    def sample(self, prompt: str, n: int, **kwargs) -> list[str]:
        if n <= 1:
            return [self.chat(prompt, **kwargs)]
        return list(self.pool.map(lambda _: self._safe(prompt, **kwargs), range(n)))

    def _safe(self, prompt: str, **kwargs) -> str:
        try:
            return self.chat(prompt, **kwargs)
        except Exception as error:  # keep sibling samples alive
            if self.verbose:
                print("[jiujiubiao llm error]", error)
            return ""


def default_client(backend: str | None = None) -> LLM:
    """Build the environment-configured client.

    ``backend`` (or ``JIUJIUBIAO_BACKEND``) selects ``anthropic`` (K3-style
    messages endpoint) or the default OpenAI-compatible ``openai``.
    """

    name = (backend or os.environ.get("JIUJIUBIAO_BACKEND") or "openai").lower()
    if name == "anthropic":
        return AnthropicClient()
    return ChatClient()


class LLM(Protocol):
    """The two call shapes every JiuJiuBiao agent needs."""

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn completion of one prompt."""
        ...

    def sample(self, prompt: str, n: int, **kwargs) -> list[str]:
        """``n`` independent samples of one prompt (self-consistency)."""
        ...


class ChatClient:
    """Stdlib-only OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        max_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        url = (
            base_url
            or os.environ.get("JIUJIUBIAO_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        self.url = url
        self.api_key = (
            api_key or os.environ.get("JIUJIUBIAO_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        )
        self.model = (
            model or os.environ.get("JIUJIUBIAO_MODEL") or os.environ.get("OPENAI_MODEL") or ""
        )
        if not self.model:
            raise ValueError("Set JIUJIUBIAO_MODEL (or pass model=) to the endpoint's model name")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = float(os.environ.get("JIUJIUBIAO_TIMEOUT", "180"))
        self.retries = int(os.environ.get("JIUJIUBIAO_RETRIES", "6"))
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.verbose = verbose
        self.n_calls = 0
        self.n_tokens = 0

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if stop:
            body["stop"] = stop
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                self.n_calls += 1
                self.n_tokens += (payload.get("usage") or {}).get("total_tokens", 0)
                return payload["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                last = error
                # Rate-limit/quota statuses get real backoff with jitter;
                # a fixed 2-6s ladder dies under sustained load (TabFact-100).
                status = getattr(error, "code", None)
                if status in (403, 429, 500, 502, 503, 529):
                    time.sleep(min(90.0, 5.0 * (2 ** attempt)) + random.random())
                else:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} retries: {last}")

    def sample(self, prompt: str, n: int, **kwargs) -> list[str]:
        if n <= 1:
            return [self.chat(prompt, **kwargs)]
        return list(self.pool.map(lambda _: self._safe(prompt, **kwargs), range(n)))

    def _safe(self, prompt: str, **kwargs) -> str:
        try:
            return self.chat(prompt, **kwargs)
        except Exception as error:  # keep sibling samples alive
            if self.verbose:
                print("[jiujiubiao llm error]", error)
            return ""


class FakeLLM:
    """Scripted LLM for offline tests and examples.

    ``rules`` maps a regex pattern to a response (string) or a callable
    receiving the prompt.  Rules are tried in order; the first match wins.
    When a ``sample`` call requests ``n`` outputs, each rule may also be a
    list of strings cycled across samples.  Unmatched prompts get
    ``default``.  All calls are recorded in ``self.calls``.
    """

    def __init__(
        self,
        rules: list[tuple[str, str | list[str] | Callable[[str], str]]] | None = None,
        default: str = "Final Answer: unknown",
    ) -> None:
        self.rules = [(re.compile(pattern, re.S | re.I), resp) for pattern, resp in (rules or [])]
        self.default = default
        self.calls: list[str] = []
        self.n_calls = 0
        self.n_tokens = 0

    def _respond(self, prompt: str, sample_idx: int = 0) -> str:
        for pattern, response in self.rules:
            if pattern.search(prompt):
                if callable(response):
                    return response(prompt)
                if isinstance(response, list):
                    return response[sample_idx % len(response)]
                return response
        return self.default

    def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        self.n_calls += 1
        return self._respond(prompt)

    def sample(self, prompt: str, n: int, **kwargs) -> list[str]:
        self.calls.append(prompt)
        self.n_calls += 1
        return [self._respond(prompt, i) for i in range(n)]


__all__ = ["AnthropicClient", "ChatClient", "FakeLLM", "LLM", "default_client"]
