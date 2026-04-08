**Anuvik Thota**
anuvik.thota.work@gmail.com · github.com/ANUVIK2401 · linkedin.com/in/anuvik-thota

Pacific Engineering Team
builders@pacific.app

April 8, 2026

---

I built a **Context Freshness Evaluator** — a tool that scores and reranks financial context chunks along two orthogonal axes before they reach an LLM: semantic relevance (judged by GPT-4o) and temporal freshness (exponential decay: e^(−λ·t)). The output is a composite-ranked list with a per-chunk recommendation — Use, Review, or Replace — along with the AI's reasoning for each score.

**The problem.** Standard RAG pipelines retrieve context by cosine similarity. They do not account for time. A query about the current Federal Reserve rate may surface a 2022 Jackson Hole speech — highly relevant by embedding distance, but actively wrong as financial context. The model will answer confidently based on stale data. In finance, confident wrongness is worse than uncertainty. This tool makes temporal decay a first-class signal in the retrieval stack.

**Technical decisions.** I separated relevance and freshness into independent signals intentionally. They decay at different rates and for different reasons — earnings data goes stale in days; regulatory guidance in months. By keeping them separate, each axis is interpretable, tunable per domain (λ=0.05–0.10 for price data, λ=0.01–0.02 for regulatory), and replaceable independently. The composite formula — `(relevance × w) + (freshness × (1−w))` — is deliberately transparent. A black-box scorer would be hard to trust in a production finance context where the cost of a wrong answer is real. I also separated infrastructure failures from model judgment: if GPT-4o scoring is unavailable, those chunks are flagged explicitly and excluded from action recommendations rather than silently scored as zero.

**Why Pacific.** Pacific's focus on context management, TTFT, and agentic workflows maps directly to this problem. The three extension paths I would prioritize are: (1) a **batch evaluation pipeline** — instead of single-query evaluation, score an entire retrieval corpus against a set of canonical queries, surface systematic staleness patterns, and trigger re-ingestion alerts before failures reach users; (2) **permissions-aware reranking** — in enterprise deployments, a context chunk may be accurate, current, and relevant but inaccessible to the requesting user; adding access control as a third axis creates a single pre-LLM context filter that handles relevance, freshness, and authorization in one pass; (3) **pre-LLM context budgeting** — use the evaluator's composite scores to select the highest-signal chunks that fit within a fixed token budget, directly reducing prompt size, cost, and time-to-first-token. These are not hypothetical; they follow directly from the architecture I built and would extend it without rewrites.

**Proof it works.** The repository includes 31 pytest tests covering the core decay math, reranking logic, GPT infra failure handling, action label consistency across all three UI surfaces, and benchmark correctness. The application has a built-in Benchmark tab demonstrating the key correctness property on 8 curated financial scenarios: stale-but-relevant chunks are correctly demoted in every case without a live API call.

I am genuinely interested in the problems Pacific is working on. Context management for financial AI is an underexplored layer of the stack, and I believe the approach I've built reflects both the practical engineering and the product intuition I'd bring to the team.

— Anuvik Thota

---

*Source: https://github.com/ANUVIK2401/pacific-context-eval*
*Live demo: https://pacific-context-eval.streamlit.app*
*Run: `pip install -r requirements.txt && streamlit run app.py`*
*Test: `pytest tests/ -v` → 31 passed*
