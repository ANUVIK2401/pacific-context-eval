**Anuvik Thota**
anuvik.thota.work@gmail.com · github.com/ANUVIK2401

Pacific Engineering Team
builders@pacific.app

April 8, 2026

---

I built a **Context Freshness Evaluator** for financial RAG systems. It scores and reranks context chunks using two signals: semantic relevance (via GPT-4o) and temporal freshness (exponential time decay). Each chunk gets a Use/Review/Replace recommendation based on its composite score.

**Why this matters.** I've seen firsthand how standard RAG breaks in time-sensitive domains. Cosine similarity doesn't care if data is from 2022 or 2024. A query asking "what's the current Fed rate?" could pull a perfectly relevant Jackson Hole speech from two years ago — and the model will answer confidently with outdated numbers. In finance, that's not just wrong, it's dangerous. I wanted to build something that treats time as a first-class signal, not an afterthought.

**How I built it.** I kept relevance and freshness as separate, interpretable scores. They decay at different rates (earnings data stales in days, regulations in months), so having independent tuning knobs matters. The composite formula is simple: `(relevance × w) + (freshness × (1−w))`. No black boxes. In production finance systems, you need to be able to explain why context was included or excluded.

I also spent time on error handling that actually matters. If the GPT-4o scoring API goes down, those chunks get flagged explicitly rather than silently scored as zero. Infrastructure failures look different than model failures in the UI, which means you can debug the right thing when something breaks.

**Where this could go at Pacific.** The architecture naturally extends to three things I'd want to build:

1. **Batch evaluation pipeline** — instead of scoring one query at a time, run an entire corpus against canonical queries. Surface patterns like "40% of Fed-related chunks are >60 days old" before they hit users. Basically, treat context quality as something you monitor, not just something you fix when it breaks.

2. **Permissions-aware reranking** — in enterprise deployments, you need a third axis. A chunk can be accurate, fresh, and relevant but still wrong for a given user if they don't have access. Making this a pre-LLM filter means you handle relevance, freshness, and authorization in one pass instead of three.

3. **TTFT optimization via context budgeting** — use composite scores to fit the best chunks into a fixed token budget. Drop the lowest-scoring ones before the prompt even gets built. Fewer tokens = faster first token and lower cost.

**Does it work?** The repo has 43 tests covering the decay math, reranking, error handling, demo-data normalization, and a benchmark suite with 8 financial scenarios (Fed policy, Apple earnings, etc.). The benchmark runs without any API calls and proves the core property: stale-but-relevant chunks get demoted correctly every time. There's also a live Streamlit demo you can try.

I'm interested in the problems Pacific is solving. The context management layer feels underexplored — everyone focuses on embeddings and retrieval but nobody's really nailing the quality control piece that sits between retrieval and generation. That gap is where production systems break, and I think the approach I built here shows how I'd think about those problems on your team.

— Anuvik

---

**Links**
Source: https://github.com/ANUVIK2401/pacific-context-eval
Live demo: https://pacific-context-eval.streamlit.app

**Setup**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Tests**
```bash
pytest tests/ -v  # 43 passed
```
