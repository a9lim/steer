"""Shared generation plumbing for OpenAI and Ollama route families.

Both protocols serialize generation on ``app.state.gen_lock`` and wrap each
request in a ``_gen_config_override`` context manager.  This module factors
out the pieces that were literally duplicated between ``server.py`` and
``ollama_api.py``:

* ``acquire_lock_with_timeout`` — 5-minute-bounded lock acquire used by
  both streaming paths (non-streaming paths use ``async with
  app.state.gen_lock`` directly since they don't need the timeout).
* ``run_blocking_generate`` — ``async with gen_lock`` + override +
  ``session.generate`` for non-streaming callers.

Wire-format emission (SSE chat deltas, SSE completions, NDJSON Ollama
chat, NDJSON Ollama generate) stays in each protocol module — those
diverge too much to share without ugliness.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from saklas.session import SaklasSession

LOCK_TIMEOUT_SECONDS = 300


def _get_override_cm():
    # Lazy import: server.py imports ollama_api which imports this module.
    from saklas.server import _gen_config_override
    return _gen_config_override


@asynccontextmanager
async def acquire_lock_with_timeout(app) -> AsyncIterator[bool]:
    """Acquire ``app.state.gen_lock`` with a ``LOCK_TIMEOUT_SECONDS`` bound.

    Yields ``True`` if acquired (releases on exit), ``False`` on timeout.
    Callers branch on the result to emit their protocol-specific 503.
    """
    try:
        async with asyncio.timeout(LOCK_TIMEOUT_SECONDS):
            await app.state.gen_lock.acquire()
    except (TimeoutError, asyncio.TimeoutError):
        yield False
        return
    try:
        yield True
    finally:
        app.state.gen_lock.release()


async def run_blocking_generate(
    app,
    session: SaklasSession,
    *,
    input: Any,
    raw: bool,
    gen_kwargs: dict[str, Any],
    temperature: Any = None,
    top_p: Any = None,
    max_tokens: Any = None,
    top_k: Any = None,
) -> Any:
    """Acquire gen_lock, apply per-request config override, call ``session.generate``.

    Propagates ``ConcurrentGenerationError`` for callers to map to their
    wire format.
    """
    _gen_config_override = _get_override_cm()
    async with app.state.gen_lock:
        with _gen_config_override(session, temperature, top_p, max_tokens, top_k=top_k):
            return session.generate(input, raw=raw, **gen_kwargs)
