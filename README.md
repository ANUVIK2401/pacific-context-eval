# ⚡ Context Freshness Evaluator

**Score and rerank financial context chunks by relevance and freshness.**

Built for [Pacific](https://pacific.app) — an Enterprise Context Management System for finance. Financial context goes stale fast. Serving an outdated 10-K filing in an agent's context window is worse than no context at all. This tool quantifies that problem.

---

## What It Does

Given a financial query and N context chunks (with timestamps), the evaluator:

1. **Freshness Scoring** — Exponential decay: `e^(-λ·days)`. Zero latency, pure math.
2. **Relevance Scoring** — GPT-4o judges how relevant each chunk is to the query. Returns a score + reasoning.
3. **Composite Reranking** — Weighted blend of both signals. Tunable weights in the UI.
4. **Stale Chunk Detection** — Flags chunks below a freshness threshold as potentially dangerous.

```
Query + Chunks
     │
     ├──► Freshness Scorer    (pure math, zero latency)
     │         └── e^(-λ · days_old)
     │
     ├──► Relevance Judge     (GPT-4o, ~300ms per chunk)
     │         └── Structured JSON output
     │
     └──► Reranker            (weighted composite sort)
               └── Final ranked list with stale warnings
```

---

## Run Locally

```bash
# Clone
git clone https://github.com/ANUVIK2401/pacific-context-eval.git
cd pacific-context-eval

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (Optional) Set API key for GPT-4o relevance scoring
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
streamlit run app.py
```

> **No API key?** The app works in **demo mode** using keyword-matching for relevance. GPT-4o mode activates when you enter an API key in the sidebar.

---

## Live Demo

🔗 **[pacific-context-eval.streamlit.app](https://pacific-context-eval.streamlit.app)**

---

## Configuration

All parameters are tunable via the sidebar UI or in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `DECAY_LAMBDA` | 0.03 | Higher = faster staleness penalty. 0.03 ≈ 50% at 23 days. |
| `RELEVANCE_WEIGHT` | 0.6 | How much query relevance matters in final ranking. |
| `FRESHNESS_WEIGHT` | 0.4 | How much recency matters. Auto-balanced with relevance. |
| `GPT_MODEL` | gpt-4o | Model for relevance scoring. |
| `MAX_TOKENS_JUDGE` | 200 | Response budget per chunk (keep low for speed). |

### Lambda Tuning Guide for Finance

| Data Type | Recommended λ | Half-life |
|---|---|---|
| Price / earnings data | 0.05 – 0.10 | 7–14 days |
| Macro policy / rates | 0.02 – 0.04 | 17–35 days |
| Regulatory / structural | 0.01 – 0.02 | 35–70 days |

---

## Tech Stack

- **Streamlit** — UI framework
- **OpenAI GPT-4o** — Relevance judge (structured JSON output)
- **Python** — Pure math freshness scoring
- **Pandas** — Data presentation

---

## Project Structure

```
pacific-context-eval/
├── app.py                  # Streamlit entrypoint
├── evaluator/
│   ├── __init__.py
│   ├── freshness.py        # Exponential decay scoring
│   ├── relevance.py        # GPT-4o judge scoring
│   └── reranker.py         # Composite reranking
├── config.py               # Tunable parameters
├── prompts/
│   └── relevance_judge.txt # System prompt for GPT-4o
├── demo_data/
│   └── examples.json       # Pre-built demo scenarios
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml         # Dark theme config
└── README.md
```

---

## Why This Matters for Pacific

Pacific builds context management infrastructure for finance. This evaluator addresses a core challenge:

- **Context has an expiration date.** A Fed rate decision from 2022 is dangerous context for a 2025 query.
- **Relevance alone isn't enough.** A perfectly relevant but stale chunk can cause hallucinations.
- **Two-signal reranking** (relevance × freshness) is what production context APIs need.

This could extend to:
- Batch evaluation pipelines for automated context quality monitoring
- Permissions-aware scoring (access tier × freshness × relevance)
- TTFT optimization by pre-filtering stale chunks before LLM inference

---

## Author

**Anuvik Thota** — [GitHub](https://github.com/ANUVIK2401)
