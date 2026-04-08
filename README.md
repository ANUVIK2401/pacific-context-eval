# ⚡ Context Freshness Evaluator

**Score and rerank financial context chunks by relevance and time-decay freshness — before stale data reaches your LLM.**

Built for [Pacific](https://pacific.app) · April 8, 2026 · [Anuvik Thota](https://github.com/ANUVIK2401)

---

## The Problem

Standard RAG pipelines retrieve context by semantic similarity. They do not account for time.

A query like *"What is the current Fed interest rate?"* may retrieve a 2022 Federal Reserve speech — highly relevant by cosine similarity, but actively dangerous as financial context. The model will answer confidently based on outdated data.

This evaluator adds a **temporal gate** before context reaches the LLM.

---

## What It Does

Given a financial query and N context chunks (with timestamps):

```
User Query + N Context Chunks
        │
        ├──► Freshness Scorer     Zero latency, pure math
        │         e^(−λ × days_old)
        │
        ├──► Relevance Judge      GPT-4o · structured JSON output
        │         score ∈ [0,1] + natural language reason
        │
        └──► Composite Reranker   Weighted blend, configurable
                  score = (rel × w) + (fresh × (1−w))
                  → Ranked list + per-chunk action: Use / Review / Replace
```

**Stale chunk detection** flags any chunk below a freshness threshold as a hallucination risk.  
**GPT infra failures** are handled separately from model judgment — if the scoring API is unavailable, that chunk shows "Scoring Unavailable" and is excluded from action recommendations. Freshness scores are unaffected.

---

## Live Demo

🔗 **[pacific-context-eval.streamlit.app](https://pacific-context-eval.streamlit.app)**

### First-Run Path

1. Open the **Fed Rate Policy** walkthrough first. It shows the core failure mode: a stale-but-relevant macro chunk should lose to fresher policy guidance.
2. Review the result cards to see how **relevance**, **freshness**, **age**, and the final **Use / Review / Replace** recommendation line up.
3. Change **λ** or the **relevance weight** in the sidebar to see how aggressively the evaluator should demote stale context.

---

## Run Locally

```bash
git clone https://github.com/ANUVIK2401/pacific-context-eval.git
cd pacific-context-eval

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set API key for GPT-4o relevance scoring
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

streamlit run app.py
```

> **No API key?** The app works in **Demo Mode** using keyword-overlap relevance (no cost, no API). The sidebar shows your active mode. For Streamlit Cloud deployment, set the key in **Settings → Secrets**.

### Run Tests

```bash
pytest tests/ -v
```

```
tests/test_evaluator.py::TestFreshnessScore::test_today_is_perfect     PASSED
tests/test_evaluator.py::TestFreshnessScore::test_decay_is_monotonic    PASSED
...
43 passed in 0.96s
```

---

## API Key Setup

The application **never exposes the API key in the UI**. Set it one of two ways:

| Environment | How to set |
|---|---|
| Local dev | `OPENAI_API_KEY=sk-...` in `.env` file |
| Streamlit Cloud | Settings → Secrets: `OPENAI_API_KEY = "sk-..."` |

---

## Configuration

All parameters are tunable via the sidebar or in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `DECAY_LAMBDA` | 0.03 | Freshness decay speed. λ=0.03 → ~50% at 23 days. |
| `RELEVANCE_WEIGHT` | 0.6 | Relevance contribution to composite score. |
| `FRESHNESS_WEIGHT` | 0.4 | Freshness contribution. Auto-balanced: `1 - rel_weight`. |
| `GPT_MODEL` | gpt-4o | Model used for relevance judging. |
| `MAX_TOKENS_JUDGE` | 200 | Per-chunk token budget (keep low for latency). |

### Lambda Tuning Guide

| Data Type | Recommended λ | Half-life |
|---|---|---|
| Price / earnings data | 0.05 – 0.10 | 7–14 days |
| Macro / rates / policy | 0.02 – 0.04 | 17–35 days |
| Regulatory / structural | 0.01 – 0.02 | 35–70 days |

---

## Benchmark

The **📊 Benchmark** tab in the app runs 8 curated financial scenarios with fixed relevance labels (no API key required — labels are manually assigned, not live-scored). Each scenario demonstrates the key correctness property:

> *A highly-relevant but stale chunk is correctly demoted below a fresher chunk of comparable relevance.*

**8 scenarios · 24 chunks · 100% correct rankings** (verified at λ=0.03, rel_weight=0.6)

Scenarios include: Fed rate (2022 vs 2024), Apple EPS, NVIDIA revenue, oil price, JPMorgan credit quality, US CPI, S&P 500 outlook, Treasury yields.

---

## Demo Scenarios

The evaluator tab includes **8 walkthrough scenarios** with relative-date placeholders that resolve at load time, so the examples stay believable as time passes.

- **Recommended first run:** Fed Rate Policy
- Other walkthroughs: Apple earnings, NVIDIA AI revenue, crude oil outlook, JPMorgan credit, US CPI trend, S&P 500 outlook, Treasury yields
- Each scenario includes an explanation of what should rank first and what stale pattern the user should notice

This makes the demo usable for a first-time reviewer without requiring them to infer the intended behavior from the README or code.

---

## Project Structure

```
pacific-context-eval/
├── app.py                  # Streamlit entry point
├── evaluator/
│   ├── demo_loader.py      # Relative-date demo scenario normalization
│   ├── freshness.py        # Exponential decay scoring
│   ├── relevance.py        # GPT-4o relevance judge
│   └── reranker.py         # Composite scoring + stale detection
├── tests/
│   ├── test_demo_loader.py # Demo-data reliability tests
│   └── test_evaluator.py   # Core evaluator tests
├── config.py               # Tunable parameters
├── prompts/
│   └── relevance_judge.txt # GPT-4o system prompt
├── demo_data/
│   └── examples.json       # 8 walkthrough scenarios with explanatory metadata
├── ONE_PAGE_LETTER.md      # Submission letter
├── requirements.txt
└── .env.example
```

---

## Why This Matters for Pacific

- **Context has an expiration date.** A Fed rate decision from 2022 is not just unhelpful for a 2025 query — it is actively wrong.
- **Relevance alone is insufficient.** Production RAG systems need both axes: *is this relevant?* and *is this still true?*
- **Pre-LLM filtering reduces cost and latency** — filtering stale chunks before the prompt reduces token count and improves TTFT.

**Immediate extension paths:**
1. **Batch eval pipeline** — score an entire retrieval corpus against canonical queries; surface systematic staleness; trigger re-ingestion alerts
2. **Permissions-aware reranking** — add access control as a third axis: `score = (rel × w₁) + (fresh × w₂) + (access × w₃)`
3. **Pre-LLM context budgeting** — use the evaluator as a filter to fit the highest-signal chunks into a fixed token budget before LLM inference

---

## Author

**Anuvik Thota** · [GitHub](https://github.com/ANUVIK2401) · [pacific-context-eval](https://github.com/ANUVIK2401/pacific-context-eval)
