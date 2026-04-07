"""
Relevance scoring module.

Uses GPT-4o as a zero-shot judge to evaluate how relevant
a context chunk is to a given financial query.

Returns structured JSON with score + reasoning.
"""

import json
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

# Load system prompt at module level
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "relevance_judge.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""


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
        On error, returns {"score": 0.0, "reason": "Scoring unavailable"}
    """
    if not chunk_text or not chunk_text.strip():
        return {"score": 0.0, "reason": "Empty chunk text"}
    
    if not query or not query.strip():
        return {"score": 0.0, "reason": "Empty query"}

    try:
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
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate response structure
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        reason = str(result.get("reason", "No reason provided"))[:100]
        
        return {"score": round(score, 4), "reason": reason}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT response as JSON: {e}")
        return {"score": 0.0, "reason": "Response parsing failed"}
    except Exception as e:
        logger.error(f"Relevance scoring failed: {e}")
        return {"score": 0.0, "reason": "Scoring unavailable"}
