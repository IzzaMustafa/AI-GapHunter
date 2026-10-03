"""Bound late-stage context for an 8,000 TPM Groq account."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import time
from typing import Any, Callable

import tiktoken
from crewai import LLM

CONTEXT_TOKENS = 2600
TASK_TOKENS = 3500
# Leave room for CrewAI's system/format instructions and the completion.
CONTEXT_NOTE = (
    "Inputs below may contain shortened passages or lists to fit the token budget. "
    "Treat omitted detail as a research limitation, never invent missing facts. "
    "Evaluate/refine every supplied opportunity. Use short sentences, lists of "
    "at most 2 brief items, and return complete JSON without extra commentary."
)
_IDENTITY_KEYS = {"name", "opportunity_name", "source"}
_COLLECTION_KEYS = {"opportunities", "evaluations", "attacks", "feasibility", "gaps"}


@lru_cache(maxsize=1)
def _encoding():
    # GPT-OSS uses the o200k token vocabulary. No LLM call is needed to count.
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact(value: Any, leaf_tokens: int, list_items: int, key: str = "") -> Any:
    if isinstance(value, str):
        if key in _IDENTITY_KEYS:
            return value
        limit = leaf_tokens
        tokens = _encoding().encode(value, disallowed_special=())
        if len(tokens) <= limit:
            return value
        # Decode on a UTF-8 boundary and mark that the passage was shortened.
        return _encoding().decode(tokens[:limit], errors="ignore") + " [shortened]"
    if isinstance(value, dict):
        return {k: _compact(v, leaf_tokens, list_items, k) for k, v in value.items()}
    if isinstance(value, list):
        # Preserve all candidate identities and their associated analyses.
        items = value if key in _COLLECTION_KEYS else value[:list_items]
        return [_compact(item, leaf_tokens, list_items, key) for item in items]
    return value


def compact_contexts(sections: dict[str, Any]) -> dict[str, str]:
    """Return valid JSON per section, retaining fields and candidate identities."""
    if count_tokens(_dump(sections)) <= CONTEXT_TOKENS:
        return {key: _dump(value) for key, value in sections.items()}
    # Compact the entire set together; a large upstream output cannot consume
    # the space reserved for the user's constraints or other stage results.
    for leaf_tokens in (256, 128, 64, 32, 16, 8):
        for list_items in (8, 4, 3, 2, 1):
            compacted = _compact(sections, leaf_tokens, list_items)
            if count_tokens(_dump(compacted)) <= CONTEXT_TOKENS:
                return {key: _dump(value) for key, value in compacted.items()}
    raise ValueError("Too many structured context fields to fit the Groq token budget.")


def validate_task(description: str) -> str:
    if count_tokens(description) > TASK_TOKENS:
        raise ValueError("Agent task exceeds the configured Groq token budget.")
    return description


def budget_messages(messages: Any, output_tokens: int) -> Any:
    """Check actual CrewAI messages, including format-repair history."""
    limit = 8000 - output_tokens - 800  # provider framing / estimation margin
    if isinstance(messages, str):
        if count_tokens(messages) > limit:
            raise ValueError("LLM input exceeds the Groq request budget.")
        return messages
    bounded = [dict(message) for message in messages]
    # These agents have no tools. An earlier invalid-format assistant reply
    # is redundant during repair; keep the task, system and repair instructions.
    while count_tokens(_dump(bounded)) > limit:
        previous = next((i for i, message in enumerate(bounded)
                         if message.get("role") == "assistant"), None)
        if previous is None:
            raise ValueError("LLM input exceeds the Groq request budget.")
        bounded.pop(previous)
    return bounded


class BudgetedLLM(LLM):
    """Validate the assembled request before each real LiteLLM call."""
    def call(self, messages, *args, **kwargs):
        output_tokens = int(self.max_tokens or self.max_completion_tokens or 2400)
        return super().call(budget_messages(messages, output_tokens), *args, **kwargs)


def run_with_rate_limit_retry(operation: Callable[[], Any]) -> Any:
    """Retry temporary quota exhaustion with a fresh Crew, never oversized input."""
    from litellm import RateLimitError

    for attempt in range(3):
        try:
            return operation()
        except RateLimitError as exc:
            if "request too large" in str(exc).lower() or attempt == 2:
                raise
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("retry-after", 61))
            except (TypeError, ValueError):
                delay = 61
            if not math.isfinite(delay):
                delay = 61
            delay = min(max(delay, 1), 120)
            # Short intervals also allow interruption while the app waits.
            while delay > 0:
                interval = min(delay, 30)
                time.sleep(interval)
                delay -= interval
