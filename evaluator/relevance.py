"""
Relevance scoring module.

Uses GPT-4o as a zero-shot judge to evaluate how relevant
a context chunk is to a given financial query.

Returns structured JSON with score + reasoning.

Exception model (important for callers):
  - Infrastructure failures (network, auth, rate-limit, timeout) are RAISED,
    not swallowed, so callers can distinguish them from model judgment.
  - Model-level failures (bad JSON parse from a valid API response) are
    caught here and returned as score=0.0 with a clear reason string.
    These represent a model output problem, not an infra problem.
"""

import json
import logging
from pathlib import Path
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, APITimeoutError, APIStatusError

logger = logging.getLogger(__name__)

# Load system prompt at module level
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "relevance_judge.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""

# Infra-level exceptions that callers should handle explicitly
# (distinct from model output / judgment failures)
INFRA_EXCEPTIONS = (
    APIConnectionError,   # network unreachable
    AuthenticationError,  # bad API key
    RateLimitError,       # quota exhausted
    APITimeoutError,      # request timeout
    APIStatusError,       # 5xx server errors
)


def score_relevance(
    query: str,
    chunk_text: str,
    client: OpenAI,
    model: str = "gpt-4o",
    max_tokens: int = 200
) -> dict:
    """
    Uses GPT-4o as a zero-shot judge to score how relevant
    a context chunk is to a financial query.

    Args:
        query: The user's financial query.
        chunk_text: The context chunk text to evaluate.
        client: An initialized OpenAI client.
        model: Model to use for scoring.
        max_tokens: Max response tokens (keep low for speed).

    Returns:
        {"score": float 0-1, "reason": str}

    Raises:
        openai.APIConnectionError: Network failure — caller must handle.
        openai.AuthenticationError: Bad API key — caller must handle.
        openai.RateLimitError: Quota exhausted — caller must handle.
        openai.APITimeoutError: Request timeout — caller must handle.
        openai.APIStatusError: 5xx server error — caller must handle.

    Model-level failures (malformed JSON in an otherwise valid response)
    are caught internally and returned as score=0.0 so they don't surface
    as infra alarms in the UI.
    """
    if not chunk_text or not chunk_text.strip():
        return {"score": 0.0, "reason": "Empty chunk text"}

    if not query or not query.strip():
        return {"score": 0.0, "reason": "Empty query"}

    # ── Infra call — exceptions propagate to caller ──────────────────
    # Any INFRA_EXCEPTIONS raised here bubble up so app.py can classify
    # them as "scoring unavailable" (infra failure ≠ model judgment).
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\nChunk: {chunk_text}"
            }
        ]
    )

    # ── Model output parsing — failures are NOT infra failures ────────
    try:
        result = json.loads(response.choices[0].message.content)
        score  = float(result.get("score", 0.0))
        score  = max(0.0, min(1.0, score))          # Clamp to [0, 1]
        reason = str(result.get("reason", "No reason provided"))[:300]
        return {"score": round(score, 4), "reason": reason}

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Model returned a valid HTTP response but with unparseable content.
        # This is a model judgment failure, not an infra failure — return gracefully.
        logger.warning(f"Model returned invalid JSON (model failure, not infra): {e}")
        return {"score": 0.0, "reason": "Model returned unparseable response"}
