"""Provider-agnostic LLM client over OpenAI-compatible chat completion APIs.

Includes a disk cache (re-runs are free and deterministic) and client-side rate limiting
with retries, which matters on free tiers. Token usage is recorded on every response so
cost per query can be computed from the prices in `configs/models.yaml`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False

    def json(self) -> dict:
        """Parse the content as a JSON object, tolerating markdown fences around it."""
        text = self.content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"no JSON object in response: {self.content[:200]!r}")
        return json.loads(text[start : end + 1])


class MissingAPIKeyError(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """The provider will not serve more requests for a while (daily quota, or still rate
    limited after every retry). Long jobs catch it to stop cleanly and resume later; the
    response cache makes the resumed run skip everything already done."""


def _is_daily_quota(exc: Exception) -> bool:
    """Daily quotas do not clear by waiting a minute, so retrying them is pointless."""
    text = str(exc).lower()
    return any(s in text for s in ("per day", "tokens per day", "requests per day", "perday"))


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    base_url: str
    api_key_env: str | None
    rpm: int | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    # USD per million tokens. Free tiers cost nothing, so these are the provider's paid
    # prices, used to estimate what each pipeline would cost in production.
    price_in: float | None = None
    price_out: float | None = None
    json_mode: bool = True  # whether the provider supports response_format=json_object

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    providers = raw["providers"]
    configs = {}
    for name, spec in raw["models"].items():
        spec = dict(spec)
        provider_name = spec.pop("provider")
        provider = providers[provider_name]
        configs[name] = ModelConfig(
            name=name,
            provider=provider_name,
            base_url=provider["base_url"],
            api_key_env=provider.get("api_key_env"),
            **spec,
        )
    return configs


class LLM:
    """Chat client for any OpenAI-compatible endpoint (Gemini, Groq, OpenRouter, Ollama...)."""

    def __init__(
        self, cfg: ModelConfig, cache_dir: Path | None = Path(".cache/llm"), max_retries: int = 6
    ):
        from openai import OpenAI

        api_key = "not-needed"
        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise MissingAPIKeyError(
                    f"Model '{cfg.name}' needs the {cfg.api_key_env} environment variable "
                    "(add it to your .env file)."
                )
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self._client = OpenAI(base_url=cfg.base_url, api_key=api_key, max_retries=0, timeout=120)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def chat(
        self,
        messages: list[dict],
        *,
        json_output: bool = False,
        temperature: float | None = None,
        cache_salt: str = "",
    ) -> LLMResponse:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_output and self.cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        cache_path = self._cache_path(payload, cache_salt)
        if cache_path is not None and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return LLMResponse(**{**data, "cached": True})

        response = self._request_with_retries(payload)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(asdict(response), ensure_ascii=False), encoding="utf-8"
            )
        return response

    def ask_json(self, system: str, user: str, **kwargs) -> tuple[dict, LLMResponse]:
        """One system + user turn whose answer must be a JSON object."""
        response = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_output=True,
            **kwargs,
        )
        return response.json(), response

    def _cache_path(self, payload: dict, salt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"salt": salt, **payload}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / self.cfg.name / f"{digest}.json"

    def _throttle(self) -> None:
        if not self.cfg.rpm:
            return
        with self._lock:
            wait = self._last_request + 60.0 / self.cfg.rpm - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _request_with_retries(self, payload: dict) -> LLMResponse:
        import openai

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        for attempt in range(self.max_retries + 1):
            self._throttle()
            start = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**payload)
            except openai.BadRequestError as exc:
                # Providers that enforce JSON mode reject output that is not valid JSON
                # (Groq: "Failed to validate JSON"). That is a bad answer, not a bad request:
                # callers count it like any other unparseable output.
                if "json" in str(exc).lower():
                    raise ValueError(f"{self.cfg.name}: invalid JSON output") from exc
                raise
            except retryable as exc:
                if _is_daily_quota(exc):
                    raise QuotaExhausted(f"{self.cfg.name}: {exc}") from exc
                if attempt == self.max_retries:
                    raise QuotaExhausted(f"{self.cfg.name}: gave up after retries: {exc}") from exc
                time.sleep(min(60.0, 2.0 * 2**attempt))
                continue
            usage = completion.usage
            return LLMResponse(
                content=completion.choices[0].message.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.perf_counter() - start,
            )
        raise AssertionError("unreachable")
