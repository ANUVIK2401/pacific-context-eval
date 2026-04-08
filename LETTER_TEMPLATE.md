# One-Page Letter — Context Freshness Evaluator
## Submit to: builders@pacific.app by Noon PT, Wednesday April 8

> ⚠️ **This is a personal writing template.** Rewrite every section in your own words.
> Pacific specifically asked for a letter NOT written using AI. Use this as an outline only.

---

**[Your Name]**
[your email] | [your GitHub]

Pacific Engineering Team
builders@pacific.app

April 8, 2025

---

**Subject: Take-Home Submission — Context Freshness Evaluator**

---

**What I built**

[2–3 sentences. e.g.: I built a Context Freshness Evaluator — a tool that scores and reranks financial
context chunks using two orthogonal signals: semantic relevance (judged by GPT-4o) and time-decay
freshness (using exponential decay e^(-λ·t)). The result is a composite reranker that surfaces the most
useful context for a user's query, filtering out both irrelevant and stale data before it reaches the LLM.]

**Why I built this for Pacific**

[2–3 sentences tying to Pacific's pillars. e.g.: Pacific's focus on context management, latency (TTFT),
and enterprise-grade evals maps directly to this problem. In financial AI specifically, a context chunk
from 18 months ago can actively harm answer quality — a user asking about current Fed policy should not
get the 2022 Jackson Hole speech as their top result. This evaluator makes that failure mode visible and
measurable.]

**Technical design decisions**

[3–4 bullets — write from your actual understanding]

- Chose exponential decay over step-function staleness because it's differentiable and tunable per domain
  (price data decays at λ=0.05–0.10; structural/regulatory data at λ=0.01–0.02).

- Separated relevance and freshness into independent signals (rather than a single LLM judge) so each
  axis is interpretable, tunable, and replaceable independently. The composite is transparent:
  `score = (rel × w) + (fresh × (1−w))`.

- Used GPT-4o as the relevance judge (not embeddings) because financial queries often require
  understanding of specific entities, time periods, and numerical claims — not just semantic similarity.

- Included a keyword-overlap fallback mode so the evaluator is usable at zero marginal cost for demos
  or high-throughput batch pipelines.

**Where this goes next**

[2–3 sentences on extension paths. These are the ones Pacific cares most about — be specific]

The most direct extension is a **batch eval pipeline**: instead of single-query evaluation, score an
entire retrieval corpus against a set of ground-truth queries, surface systematic staleness patterns,
and flag which documents most urgently need re-ingestion. A second extension is **permissions-aware
reranking**: in enterprise contexts, freshness and relevance alone are insufficient — a context chunk
may be accurate and current but inaccessible to a given user. Adding a permissions layer to the
composite score (demoting chunks a user cannot access) turns this into a full pre-LLM context filter.
Together, these two extensions directly address latency and cost: filtering stale or inaccessible context
before the LLM prompt reduces token count and improves time-to-first-token.

**Why I'd be a good fit**

[2–3 personal sentences. Write from scratch.]

---

*Source code: https://github.com/ANUVIK2401/pacific-context-eval*
*Live demo: [your Streamlit Cloud URL]*
*Run locally: `pip install -r requirements.txt && streamlit run app.py`*
*Tests: `pytest tests/ -v` — 17 tests, all passing*
