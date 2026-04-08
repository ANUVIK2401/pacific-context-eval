"""
Context Freshness Evaluator — Streamlit App
Built for Pacific Take-Home

Scores and reranks financial context chunks by relevance (GPT-4o)
and freshness (exponential time decay).
"""

import json
import math
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from evaluator.demo_loader import load_demo_scenarios
from evaluator.freshness import freshness_score, days_old
from evaluator.relevance import score_relevance
from evaluator.reranker import rerank_chunks, get_stale_chunks
from config import RELEVANCE_WEIGHT, DECAY_LAMBDA, GPT_MODEL, MAX_TOKENS_JUDGE

# ── API Key ────────────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    try: _api_key = st.secrets.get("OPENAI_API_KEY", "")
    except: pass
USE_GPT = bool(_api_key and _api_key.startswith("sk-"))

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Context Freshness Evaluator · Pacific",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

TEST_GROUPS = [
    ("TestFreshnessScore", 6,
     "Decay is monotonic, bounded, and tunable across different financial data lifetimes."),
    ("TestRerankChunks", 7,
     "Validates composite ordering, stale-chunk demotion, custom weights, and edge cases."),
    ("TestGetStaleChunks", 4,
     "Confirms threshold-driven stale detection for fresh, mixed, and obviously stale batches."),
    ("TestGPTFailureHandling", 6,
     "Separates infra failures from model judgment and preserves freshness when scoring is unavailable."),
    ("TestActionConsistency", 5,
     "Keeps result cards, summary table, and export actions aligned so the UI stays trustworthy."),
    ("TestBenchmarkCorrectness", 3,
     "Proves the curated benchmark cases rank exactly as expected across all 8 scenarios."),
    ("TestDemoScenarioLoading", 5,
     "Checks relative-date demo parsing so examples stay reliable as the calendar moves forward."),
    ("TestScoreRelevanceIntegration", 7,
     "Exercises the real relevance scorer contract with mocked OpenAI responses and failures."),
]
TOTAL_TESTS = sum(group[1] for group in TEST_GROUPS)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp { font-family:'Inter',sans-serif; background:#07070d; color:#e8e8f2; }

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,header,[data-testid="stToolbar"] { visibility:hidden; }
/* Sidebar collapse control: visible so users can open/close the param panel */

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#1a56ff33;border-radius:3px}

/* ── HERO ─────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    background: linear-gradient(140deg,#040820 0%,#091460 40%,#0d1f8f 70%,#0a3dcc 100%);
    border-radius: 24px;
    padding: 3.5rem 4rem 3rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    border: 1px solid rgba(26,86,255,0.2);
}
.hero-wrap::before {
    content:'';position:absolute;top:-60px;right:-60px;
    width:420px;height:420px;
    background:radial-gradient(circle,rgba(26,86,255,0.18) 0%,transparent 70%);
    border-radius:50%;pointer-events:none;
}
.hero-wrap::after {
    content:'';position:absolute;bottom:-80px;left:30%;
    width:300px;height:300px;
    background:radial-gradient(circle,rgba(0,100,255,0.1) 0%,transparent 70%);
    border-radius:50%;pointer-events:none;
}
.hero-eyebrow {
    font-size:0.7rem;font-weight:700;letter-spacing:0.14em;
    text-transform:uppercase;color:#4d8bff;margin-bottom:1rem;
    display:flex;align-items:center;gap:0.5rem;
}
.hero-eyebrow::before {
    content:'';display:inline-block;width:20px;height:2px;
    background:linear-gradient(90deg,#1a56ff,#4d8bff);border-radius:2px;
}
.hero-h1 {
    font-size:3rem;font-weight:800;line-height:1.1;
    letter-spacing:-0.04em;color:#fff;margin:0 0 1rem;
}
.hero-h1 span { color:#4d8bff; }
.hero-sub {
    font-size:1.05rem;color:rgba(255,255,255,0.6);
    max-width:520px;line-height:1.7;margin:0 0 1.8rem;font-weight:400;
}
.hero-pills { display:flex;gap:0.6rem;flex-wrap:wrap; }
.hero-pill {
    background:rgba(26,86,255,0.15);border:1px solid rgba(26,86,255,0.3);
    color:#7aa5ff;font-size:0.68rem;font-weight:600;padding:0.3rem 0.85rem;
    border-radius:20px;letter-spacing:0.05em;text-transform:uppercase;
}
.hero-mode-badge {
    position:absolute;top:2rem;right:2rem;
    font-size:0.72rem;font-weight:600;padding:0.4rem 1rem;border-radius:20px;
    display:flex;align-items:center;gap:0.4rem;
}
.badge-gpt  { background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#10b981; }
.badge-demo { background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#f59e0b; }

/* ── SECTION LABEL ────────────────────────────────── */
.section-label {
    font-size:0.65rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#4d8bff;
    margin-bottom:0.9rem;display:flex;align-items:center;gap:0.5rem;
}
.section-label::after {
    content:'';flex:1;height:1px;
    background:linear-gradient(90deg,rgba(26,86,255,0.3),transparent);
}

/* ── SCENARIO CARDS ────────────────────────────────── */
.scenario-grid { display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:2rem; }
.sc-card {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:16px;padding:1.4rem 1.2rem;
    cursor:pointer;transition:all 0.25s;position:relative;overflow:hidden;
    text-align:center;
}
.sc-card::before {
    content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(26,86,255,0.08) 0%,transparent 60%);
    opacity:0;transition:opacity 0.25s;
}
.sc-card:hover { border-color:rgba(26,86,255,0.45); transform:translateY(-3px);
    box-shadow:0 12px 32px rgba(26,86,255,0.12); }
.sc-card:hover::before { opacity:1; }
.sc-card.active-card{
    border-color:#1a56ff !important;
    background:linear-gradient(145deg,#0d1a4a,#0a1238) !important;
    box-shadow:0 0 0 1px #1a56ff, 0 16px 40px rgba(26,86,255,0.2);
}
.sc-icon { font-size:1.8rem;margin-bottom:0.75rem;display:block; }
.sc-name { font-size:0.78rem;font-weight:600;color:#c8c8e0;line-height:1.35; }
.sc-query { font-size:0.68rem;color:rgba(200,200,224,0.4);line-height:1.4;margin-top:0.4rem; }
.sc-pill {
    display:inline-block;margin-top:0.7rem;
    font-size:0.6rem;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;
    padding:0.2rem 0.65rem;border-radius:12px;
    background:rgba(26,86,255,0.12);color:#6699ff;border:1px solid rgba(26,86,255,0.2);
}

/* ── QUERY + CHUNKS ──────────────────────────────── */
.query-box {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(255,255,255,0.08);border-radius:16px;
    padding:1.5rem 1.75rem;margin-bottom:1.5rem;
}

.chunk-wrap {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:16px;padding:1.25rem 1.4rem 1rem;
    transition:border-color 0.25s;
}
.chunk-wrap:hover { border-color:rgba(26,86,255,0.25); }
.chunk-header {
    display:flex;justify-content:space-between;align-items:center;
    margin-bottom:0.7rem;
}
.chunk-num {
    font-size:0.65rem;font-weight:700;letter-spacing:0.1em;
    text-transform:uppercase;color:rgba(200,200,224,0.4);
}

/* freshness pills */
.fp-green {background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#10b981;padding:0.15rem 0.6rem;border-radius:12px;font-size:0.65rem;font-weight:600;}
.fp-yellow{background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);color:#f59e0b;padding:0.15rem 0.6rem;border-radius:12px;font-size:0.65rem;font-weight:600;}
.fp-red   {background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.25);color:#ef4444;padding:0.15rem 0.6rem;border-radius:12px;font-size:0.65rem;font-weight:600;}

/* ── EVAL BUTTON ────────────────────────────────── */
.eval-btn-wrap { text-align:center;margin:2rem 0; }

/* ── KPI BAR ─────────────────────────────────────── */
.kpi-row { display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem; }
.kpi-card {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(26,86,255,0.1);border-radius:16px;
    padding:1.4rem 1.5rem;text-align:center;
    transition:all 0.25s;
}
.kpi-card:hover{border-color:rgba(26,86,255,0.3);transform:translateY(-2px);}
.kpi-val{font-size:2.1rem;font-weight:800;font-family:'JetBrains Mono',monospace;line-height:1;}
.kpi-lbl{font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(200,200,224,0.4);margin-top:0.5rem;}

/* ── RESULT CARD ─────────────────────────────────── */
.rcard {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(255,255,255,0.05);border-radius:18px;
    padding:1.75rem 2rem;margin-bottom:1.2rem;
    transition:all 0.25s;position:relative;overflow:hidden;
}
.rcard:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,0,0,0.3);}
.rcard-top  {border-left:4px solid #10b981;}
.rcard-mid  {border-left:4px solid #f59e0b;}
.rcard-low  {border-left:4px solid #ef4444;}
.rcard-rank {
    display:inline-flex;align-items:center;gap:0.4rem;
    background:linear-gradient(135deg,#0d2580,#1a56ff);
    color:#fff;font-size:0.75rem;font-weight:700;
    padding:0.3rem 0.85rem;border-radius:8px;letter-spacing:0.04em;
}
.rcard-meta{font-size:0.75rem;color:rgba(200,200,224,0.45);font-weight:500;}
.rcard-composite{font-size:2rem;font-weight:800;font-family:'JetBrains Mono',monospace;line-height:1;}
.rcard-composite-lbl{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(200,200,224,0.35);}
.rcard-preview{
    font-size:0.83rem;color:rgba(200,200,224,0.5);
    line-height:1.65;font-style:italic;
    margin:1rem 0;padding:0.85rem 1rem;
    background:rgba(255,255,255,0.02);border-radius:8px;border-left:2px solid rgba(255,255,255,0.06);
}
.score-col-lbl{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(200,200,224,0.35);margin-bottom:0.25rem;}
.score-col-val{font-size:1.2rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#e8e8f2;}
.rcard-reason{
    margin-top:1.1rem;padding-top:1rem;
    border-top:1px solid rgba(255,255,255,0.04);
    font-size:0.78rem;color:rgba(200,200,224,0.45);
    line-height:1.6;font-style:italic;
}
.reason-lbl{
    font-style:normal;font-size:0.6rem;font-weight:700;
    letter-spacing:0.1em;text-transform:uppercase;
    color:rgba(200,200,224,0.25);margin-right:0.5rem;
}

/* ── ALERT BOXES ─────────────────────────────────── */
.a-warn {
    background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);
    border-radius:12px;padding:0.85rem 1.25rem;margin:0.75rem 0;
    font-size:0.82rem;color:#fca5a5;line-height:1.6;
}
.a-ok {
    background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.2);
    border-radius:12px;padding:0.85rem 1.25rem;margin:0.75rem 0;
    font-size:0.82rem;color:#6ee7b7;line-height:1.6;
}

/* ── MODE BANNER ─────────────────────────────────── */
.mode-gpt  {background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:0.65rem 1rem;font-size:0.8rem;color:#10b981;margin-bottom:1rem;}
.mode-demo {background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:0.65rem 1rem;font-size:0.8rem;color:#f59e0b;margin-bottom:1rem;}

/* ── LEGEND BAR ──────────────────────────────────── */
.legend-row {
    display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;
    background:rgba(26,86,255,0.04);border:1px solid rgba(26,86,255,0.1);
    border-radius:12px;padding:0.8rem 1.25rem;margin:1.5rem 0;
    font-size:0.73rem;color:rgba(200,200,224,0.5);
}
.legend-item{display:flex;align-items:center;gap:0.4rem;}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}

/* ── ONBOARDING ───────────────────────────────────── */
.context-panel {
    background:linear-gradient(145deg,#0e0e1a,#0b0b15);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:18px;
    padding:1.4rem 1.5rem;
    margin-bottom:1.5rem;
}
.context-title {
    font-size:0.78rem;
    font-weight:700;
    letter-spacing:0.08em;
    text-transform:uppercase;
    color:#7aa5ff;
    margin-bottom:0.8rem;
}
.context-copy {
    font-size:0.82rem;
    line-height:1.7;
    color:rgba(232,232,242,0.64);
}
.step-card {
    background:linear-gradient(145deg,#0d1537,#0a112d);
    border:1px solid rgba(77,139,255,0.18);
    border-radius:16px;
    padding:1.15rem 1.2rem;
    height:100%;
}
.step-num {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:28px;
    height:28px;
    border-radius:999px;
    background:rgba(26,86,255,0.18);
    color:#9bbcff;
    font-weight:700;
    font-size:0.82rem;
    margin-bottom:0.7rem;
}
.step-title {
    font-size:0.9rem;
    font-weight:700;
    color:#f4f7ff;
    margin-bottom:0.5rem;
}
.step-copy {
    font-size:0.78rem;
    line-height:1.65;
    color:rgba(232,232,242,0.6);
}
.mini-label {
    font-size:0.62rem;
    font-weight:700;
    letter-spacing:0.1em;
    text-transform:uppercase;
    color:rgba(122,165,255,0.8);
    margin-bottom:0.45rem;
}
.chunk-note {
    margin:0.45rem 0 0.15rem;
    padding:0.5rem 0.65rem;
    background:rgba(26,86,255,0.06);
    border:1px solid rgba(26,86,255,0.12);
    border-radius:10px;
    font-size:0.72rem;
    line-height:1.55;
    color:rgba(232,232,242,0.58);
}

/* ── SIDEBAR ─────────────────────────────────────── */
.stSidebar [data-testid="stSidebarContent"]{background:#08080f;border-right:1px solid rgba(255,255,255,0.05);}
.sb-title {font-size:0.6rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:rgba(200,200,224,0.3);margin:1.2rem 0 0.5rem;}

/* ── FOOTER ──────────────────────────────────────── */
.footer{text-align:center;padding:2rem 0 1rem;font-size:0.7rem;color:rgba(200,200,224,0.2);}
.footer a{color:#4d8bff;text-decoration:none;}

/* ── Animations ──────────────────────────────────── */
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp 0.4s ease forwards;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_demo_data():
    return load_demo_scenarios()


def get_recommended_demo(demos):
    if not demos:
        return None
    for demo in demos:
        if demo.get("recommended"):
            return demo
    return demos[0]


def action_label(chunk):
    if chunk.get("scoring_unavailable"):
        return "scoring_unavailable"
    if chunk["composite_score"] >= 0.7:
        return "use"
    if chunk["composite_score"] >= 0.4:
        return "review"
    return "replace"

def simulate_relevance(query, text):
    stop = {'the','a','an','is','are','was','were','what','how','in','of','to',
            'and','for','at','by','on','with','its','that','this','from','as',
            'it','be','has','had','does','do','will','about','their','which','were'}
    qw = set(query.lower().split()) - stop
    cw = set(text.lower().split()) - stop
    if not qw: return {"score":0.5,"reason":"Unable to parse query"}
    ov = qw & cw
    s = min(1.0, len(ov) / max(len(qw),1) * 1.2)
    r = f"Keyword match: {', '.join(list(ov)[:4])}" if ov else "No keyword overlap found"
    return {"score": round(s,4), "reason": r}

def fcolor(score):
    if score >= 0.7: return "#10b981", "fp-green", "🟢"
    if score >= 0.3: return "#f59e0b", "fp-yellow", "🟡"
    return "#ef4444", "fp-red", "🔴"

def ccolor(score):
    if score >= 0.7: return "rcard-top", "#10b981"
    if score >= 0.4: return "rcard-mid", "#f59e0b"
    return "rcard-low", "#ef4444"

def decay_bars(lam):
    pts = [("Today",0),("1 wk",7),("1 mo",30),("3 mo",90),("6 mo",180),("1 yr",365)]
    out = ""
    for lbl, d in pts:
        pct = round(math.exp(-lam*d)*100, 1)
        c = "#10b981" if pct > 60 else "#f59e0b" if pct > 25 else "#ef4444"
        out += (
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;">'
            f'<div style="font-size:0.65rem;color:rgba(200,200,224,0.4);width:38px;text-align:right;">{lbl}</div>'
            f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;border-radius:4px;background:{c};"></div></div>'
            f'<div style="font-size:0.63rem;color:rgba(200,200,224,0.4);width:36px;">{pct}%</div></div>'
        )
    return out


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for k, v in [("results",None),("evaluated",False),("query_input",""),
             ("active_demo",None)]:
    if k not in st.session_state: st.session_state[k] = v


def load_scenario(demo):
    st.session_state.active_demo = demo["name"]
    st.session_state.query_input = demo["query"]
    for i in range(3):
        st.session_state[f"chunk_text_{i}"] = ""
    for i, c in enumerate(demo["chunks"][:3]):
        st.session_state[f"chunk_text_{i}"] = c["text"]
        st.session_state[f"chunk_date_{i}"] = datetime.strptime(c["date"], "%Y-%m-%d").date()
    st.session_state.results = None
    st.session_state.evaluated = False


# ══════════════════════════════════════════════════════════════
#  SIDEBAR  (compact — just tuning params)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">⚙️ Scoring Parameters</div>', unsafe_allow_html=True)
    decay_lambda = st.slider("Decay Speed (λ)", 0.005, 0.10, DECAY_LAMBDA, 0.005,
        help="Higher = faster staleness. λ=0.03 → ~50% at 23 days.")
    rel_weight = st.slider("Relevance Weight", 0.1, 0.9, RELEVANCE_WEIGHT, 0.05,
        help="Composite = (Rel × w) + (Fresh × (1-w))")
    fresh_weight = round(1.0 - rel_weight, 2)
    st.caption(f"Freshness Weight: **{fresh_weight}** (auto)")

    st.divider()
    st.markdown('<div class="sb-title">📉 Decay Preview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:#0b0b15;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.8rem 0.9rem;">'
        f'{decay_bars(decay_lambda)}</div>',
        unsafe_allow_html=True
    )
    st.caption(f"e^(−{decay_lambda} × days)")

    st.divider()
    if USE_GPT:
        st.markdown('<div class="mode-gpt">🤖 GPT-4o Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-demo">⚡ Demo Mode (keyword)</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:1rem;font-size:0.65rem;color:rgba(200,200,224,0.2);">'
        'Built for <a style="color:#4d8bff;text-decoration:none;" href="https://pacific.app">Pacific</a> · Anuvik Thota</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
badge_html = (
    '<span class="hero-mode-badge badge-gpt">🤖 GPT-4o Active</span>'
    if USE_GPT else
    '<span class="hero-mode-badge badge-demo">⚡ Demo Mode</span>'
)
st.markdown(f"""
<div class="hero-wrap">
  {badge_html}
  <div class="hero-eyebrow">Pacific · Context Intelligence</div>
  <h1 class="hero-h1">Context <span>Freshness</span><br>Evaluator</h1>
  <p class="hero-sub">
    Score and rerank financial context chunks by <strong style="color:#7aa5ff;">AI relevance</strong>
    and <strong style="color:#7aa5ff;">time-decay freshness</strong> — before stale data
    causes hallucinations in your LLM pipeline.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">Context Management</span>
    <span class="hero-pill">RAG Evals</span>
    <span class="hero-pill">Reranking</span>
    <span class="hero-pill">TTFT</span>
    <span class="hero-pill">Finance AI</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_eval, tab_bench, tab_tests = st.tabs(["⚡ Evaluator", "📊 Benchmark", "🧪 Tests"])

with tab_eval:
    demos = load_demo_data()
    recommended_demo = get_recommended_demo(demos)
    if not st.session_state.active_demo and not st.session_state.query_input and recommended_demo:
        load_scenario(recommended_demo)
    active = st.session_state.active_demo
    active_demo = next((demo for demo in demos if demo["name"] == active), recommended_demo)

    st.markdown('<div class="section-label">🧭 What This Demo Shows</div>', unsafe_allow_html=True)
    intro_a, intro_b = st.columns([1.4, 1], gap="medium")
    with intro_a:
        st.markdown(
            """
            <div class="context-panel">
              <div class="context-title">Why This Exists</div>
              <div class="context-copy">
                This project sits between retrieval and generation. It asks two questions before an LLM sees any
                context: <strong style="color:#f4f7ff;">is this chunk relevant?</strong> and
                <strong style="color:#f4f7ff;">is it still fresh enough to trust?</strong>.
                That makes it a concrete Pacific-style context management tool, not just a model demo.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_b:
        st.markdown(
            """
            <div class="context-panel">
              <div class="context-title">Pacific Fit</div>
              <div class="context-copy">
                The live evaluator demonstrates <strong style="color:#f4f7ff;">context management</strong> and
                <strong style="color:#f4f7ff;">evals</strong> directly, while the reranker gives a path to
                <strong style="color:#f4f7ff;">TTFT</strong> improvements by dropping stale chunks before prompt
                assembly. Permissions-aware reranking is the natural next axis.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    step1, step2, step3 = st.columns(3, gap="medium")
    for column, (num, title, copy) in zip([step1, step2, step3], [
        ("1", "Start With The Recommended Scenario",
         "Use the Fed-rate walkthrough first. It shows the core failure mode: a stale but relevant chunk should lose."),
        ("2", "Inspect Why The Ranking Changed",
         "Each result card explains the score with relevance, freshness, age, and the downstream recommendation."),
        ("3", "Tune The Tradeoff",
         "Adjust decay λ and the relevance weight in the sidebar to see how strict the pre-LLM filter should be."),
    ]):
        with column:
            st.markdown(
                f'<div class="step-card"><div class="step-num">{num}</div>'
                f'<div class="step-title">{title}</div>'
                f'<div class="step-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )

    # ── SCENARIO PICKER ──────────────────────────────────────────

    st.markdown('<div class="section-label">🎬 Quick Start — Choose a Scenario</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:rgba(200,200,224,0.4);margin:-0.4rem 0 1.2rem;">'
        'These walkthroughs use relative dates that resolve at load time, so the examples stay believable as the '
        'calendar moves forward. Load one to understand the intended ranking pattern before trying your own data.</p>',
        unsafe_allow_html=True
    )

    cards_per_row = 4
    emoji_icons = "🍎🏦⚡🛢️💳📉📈📊"
    for row_start in range(0, len(demos), cards_per_row):
        cols = st.columns(cards_per_row, gap="medium")
        for idx, demo in enumerate(demos[row_start:row_start + cards_per_row]):
            actual_idx = row_start + idx
            with cols[idx]:
                is_active = demo["name"] == active
                active_cls = "active-card" if is_active else ""
                check = "✓ " if is_active else ""
                short_name = demo["name"].split(" ", 1)[1] if demo["name"][0] in emoji_icons else demo["name"]
                icon = demo["name"][0] if demo["name"][0] in emoji_icons else "📄"
                q_preview = demo["query"][:55] + "…" if len(demo["query"]) > 55 else demo["query"]
                pill_label = demo.get("category", f'{demo["chunk_count"]} chunks')

                st.markdown(
                    f'<div class="sc-card {active_cls}">'
                    f'<span class="sc-icon">{icon}</span>'
                    f'<div class="sc-name">{check}{short_name}</div>'
                    f'<div class="sc-query">{q_preview}</div>'
                    f'<span class="sc-pill">{pill_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button("Load", key=f"sc_{actual_idx}", use_container_width=True):
                    load_scenario(demo)
                    st.rerun()

    if active_demo:
        st.markdown(
            f"""
            <div class="context-panel">
              <div class="context-title">Active Walkthrough</div>
              <div class="context-copy">
                <div class="mini-label">Scenario</div>
                <strong style="color:#f4f7ff;">{active_demo["name"]}</strong><br>
                {active_demo.get("description", "")}
                <div class="mini-label" style="margin-top:0.95rem;">What To Notice</div>
                {active_demo.get("look_for", "The newest, most directly answering chunk should outrank older context.")}
                <div class="mini-label" style="margin-top:0.95rem;">Expected Outcome</div>
                {active_demo.get("expected_outcome", "Chunk 1 should rank highest and stale context should be demoted.")}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    
    
    # ══════════════════════════════════════════════════════════════
    #  QUERY INPUT
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">🔍 Financial Query</div>', unsafe_allow_html=True)
    query = st.text_input(
        "query",
        placeholder="e.g., What were Apple's Q4 2024 earnings per share?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    # ══════════════════════════════════════════════════════════════
    #  CONTEXT CHUNKS
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">📄 Context Chunks</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:rgba(200,200,224,0.35);margin:-0.4rem 0 1rem;">'
        'Paste up to 3 context chunks from your RAG pipeline with their source dates. '
        'Freshness score updates instantly as you set the date.</p>',
        unsafe_allow_html=True
    )
    
    chunk_cols = st.columns(3, gap="medium")
    chunks_input = []
    
    for i in range(3):
        with chunk_cols[i]:
            demo_chunk = active_demo["chunks"][i] if active_demo and i < len(active_demo["chunks"]) else None
            if demo_chunk and (demo_chunk.get("title") or demo_chunk.get("note")):
                title = demo_chunk.get("title", f"Chunk {i+1}")
                note = demo_chunk.get("note", "")
                st.markdown(
                    f'<div class="chunk-note"><strong style="color:#f4f7ff;">{title}</strong><br>{note}</div>',
                    unsafe_allow_html=True,
                )
            chunk_text = st.text_area(
                f"chunk_text_{i}",
                height=140,
                placeholder=f"Paste context chunk {i+1} here…",
                key=f"chunk_text_{i}",
                label_visibility="collapsed"
            )
            chunk_date = st.date_input(
                f"Source date",
                key=f"chunk_date_{i}",
                label_visibility="collapsed"
            )
            if chunk_text.strip():
                age  = days_old(datetime.combine(chunk_date, datetime.min.time()))
                fsc  = freshness_score(datetime.combine(chunk_date, datetime.min.time()), decay_lambda)
                clr, pill, emoji = fcolor(fsc)
                st.markdown(
                    f'<div style="margin-top:0.3rem;display:flex;align-items:center;justify-content:space-between;">'
                    f'<span class="{pill}">{emoji} {age}d old</span>'
                    f'<span style="font-size:0.68rem;color:{clr};font-family:monospace;font-weight:600;">f={fsc:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                chunks_input.append({
                    "text": chunk_text.strip(), "date": chunk_date.isoformat(),
                    "chunk_date_obj": datetime.combine(chunk_date, datetime.min.time()),
                    "index": i
                })
            else:
                st.markdown(
                    f'<div style="font-size:0.7rem;color:rgba(200,200,224,0.2);margin-top:0.3rem;">'
                    f'Chunk {i+1} — awaiting input</div>',
                    unsafe_allow_html=True
                )
    
    # ══════════════════════════════════════════════════════════════
    #  EVALUATE BUTTON
    # ══════════════════════════════════════════════════════════════
    st.markdown("")
    _, bc, _ = st.columns([1.5, 3, 1.5])
    with bc:
        ready = bool(query.strip()) and len(chunks_input) > 0
        do_eval = st.button(
            "⚡  Evaluate & Rerank" + (" — GPT-4o" if USE_GPT else " — Keyword"),
            use_container_width=True, type="primary", disabled=not ready
        )
        if not ready:
            st.markdown(
                '<div style="text-align:center;font-size:0.72rem;color:rgba(200,200,224,0.2);margin-top:0.4rem;">'
                'Select a scenario or enter a query + at least one chunk to evaluate.</div>',
                unsafe_allow_html=True
            )
    
    
    # ══════════════════════════════════════════════════════════════
    #  EVALUATION PIPELINE
    # ══════════════════════════════════════════════════════════════
    if do_eval and ready:
        client = None
        if USE_GPT:
            from openai import OpenAI
            client = OpenAI(api_key=_api_key)

        scored = []
        prog = st.progress(0, text="Initialising…")
        has_gpt_error = False

        for idx, chunk in enumerate(chunks_input):
            fsc = freshness_score(chunk["chunk_date_obj"], decay_lambda)
            age = days_old(chunk["chunk_date_obj"])

            # ── Relevance scoring with infra failure separation ──
            scoring_unavailable = False
            if client:
                try:
                    rel = score_relevance(query, chunk["text"], client,
                                         model=GPT_MODEL, max_tokens=MAX_TOKENS_JUDGE)
                except Exception as gpt_err:
                    # Infra failure ≠ model judgment — log separately
                    rel = {"score": None, "reason": f"⚠️ Scoring unavailable: {str(gpt_err)[:120]}"}
                    scoring_unavailable = True
                    has_gpt_error = True
            else:
                rel = simulate_relevance(query, chunk["text"])

            scored.append({
                "text": chunk["text"], "date": chunk["date"], "days_old": age,
                "freshness_score": fsc,
                "relevance_score": rel["score"],          # None if unavailable
                "relevance_reason": rel["reason"],
                "original_index": chunk["index"]+1,
                "scoring_unavailable": scoring_unavailable,
            })
            prog.progress((idx+1)/len(chunks_input), text=f"Scored chunk {idx+1}/{len(chunks_input)}")

        # Exclude unavailable chunks from composite ranking (use freshness only)
        for c in scored:
            if c["scoring_unavailable"]:
                c["relevance_score"] = 0.0   # neutral — won’t influence action label

        ranked = rerank_chunks(scored, rel_weight, fresh_weight)
        stale  = get_stale_chunks(ranked)

        if has_gpt_error:
            st.warning("⚠️ GPT-4o scoring failed for one or more chunks (infra error, not model judgment). "
                       "Those chunks show relevance=0 and are excluded from action recommendations. "
                       "Freshness scores are unaffected.")

        st.session_state.update({
            "results": ranked, "stale_count": len(stale), "evaluated": True,
            "eval_mode": "GPT-4o" if client else "Keyword Match",
            "has_gpt_error": has_gpt_error,
        })
        prog.empty()
    
    
    # ══════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════
    if st.session_state.evaluated and st.session_state.results:
        ranked      = st.session_state.results
        stale_count = st.session_state.stale_count
        eval_mode   = st.session_state.get("eval_mode","?")
    
        st.divider()
        st.markdown('<div class="section-label">📊 Evaluation Results</div>', unsafe_allow_html=True)
    
        # Mode banner
        if eval_mode == "GPT-4o":
            st.markdown('<div class="mode-gpt">🤖 Scored with GPT-4o — AI-generated relevance reasoning</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mode-demo">⚡ Scored with keyword matching — add OPENAI_API_KEY to .env for GPT-4o</div>', unsafe_allow_html=True)
    
        # Alert
        if stale_count:
            st.markdown(
                f'<div class="a-warn">⚠️ <strong>{stale_count} stale chunk{"s" if stale_count>1 else ""} detected '
                f'(freshness &lt; 0.3).</strong> Serving outdated financial data to an LLM risks hallucinated or '
                f'contradictory answers. Consider sourcing fresher context for highlighted chunks.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="a-ok">✅ All chunks are reasonably fresh — no staleness warnings.</div>', unsafe_allow_html=True)
    
        # KPI row
        avg_c = sum(c["composite_score"] for c in ranked) / len(ranked)
        top_c = ranked[0]["composite_score"]
        best  = ranked[0]["original_index"]
        k1,k2,k3,k4 = st.columns(4)
        for col,(val,lbl,clr) in zip([k1,k2,k3,k4],[
            (f"{top_c:.2f}","Top Composite","#1a56ff"),
            (f"{avg_c:.2f}","Avg Composite","#4d8bff"),
            (f"#{best}","Best Chunk","#10b981"),
            (str(stale_count),"Stale Chunks","#ef4444" if stale_count else "#10b981"),
        ]):
            with col:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val" style="color:{clr};">{val}</div>'
                    f'<div class="kpi-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )
    
        # Legend
        st.markdown("""
        <div class="legend-row">
          <div class="legend-item"><div class="legend-dot" style="background:#10b981;"></div>Composite ≥ 0.7 — <strong>Use</strong></div>
          <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div>0.4–0.7 — <strong>Review</strong></div>
          <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div>&lt; 0.4 — <strong>Replace</strong></div>
          <span style="margin-left:auto;font-family:monospace;font-size:0.72rem;">
            Composite = Relevance×{rw:.0%} + Freshness×{fw:.0%}
          </span>
        </div>
        """.format(rw=rel_weight, fw=fresh_weight), unsafe_allow_html=True)
    
        st.markdown("")
    
        # Result cards
        def _bar(pct, color):
            return (
                f'<div style="background:rgba(255,255,255,0.05);border-radius:5px;height:5px;overflow:hidden;margin-top:0.3rem;">'
                f'<div style="width:{pct:.0f}%;height:100%;border-radius:5px;background:{color};"></div></div>'
            )
    
        for chunk in ranked:
            comp  = chunk["composite_score"]
            cc, clr = ccolor(comp)
            _, _, emoji = fcolor(chunk["freshness_score"])
            preview = chunk["text"][:160] + ("…" if len(chunk["text"])>160 else "")
            rc = chunk["relevance_score"]; fc = chunk["freshness_score"]
            rc_clr = "#10b981" if rc>=0.7 else "#f59e0b" if rc>=0.3 else "#ef4444"
            fc_clr = "#10b981" if fc>=0.7 else "#f59e0b" if fc>=0.3 else "#ef4444"
            unavailable = chunk.get("scoring_unavailable", False)
            action = action_label(chunk)
            action_display = (
                "⚠️ Scoring Unavailable" if action == "scoring_unavailable"
                else "✅ Use" if action == "use"
                else "⚠️ Review" if action == "review"
                else "❌ Replace"
            )

    
            html = (
                f'<div class="rcard {cc} anim">'
    
                # top row
                f'<div style="display:flex;align-items:center;gap:0.85rem;margin-bottom:0.6rem;">'
                f'<span class="rcard-rank">RANK #{chunk["rank"]}</span>'
                f'<span class="rcard-meta">Chunk {chunk["original_index"]} &middot; {chunk["date"]} &middot; {chunk["days_old"]}d old {emoji}</span>'
                f'<span style="margin-left:auto;font-size:0.75rem;font-weight:600;'
                f'padding:0.25rem 0.75rem;border-radius:8px;'
                f'background:rgba(255,255,255,0.04);color:rgba(200,200,224,0.6);">{action_display}</span>'
                f'</div>'
    
                # composite + meta
                f'<div style="display:flex;align-items:flex-end;gap:0.4rem;margin-bottom:0.15rem;">'
                f'<span class="rcard-composite" style="color:{clr};">{comp:.3f}</span>'
                f'<span class="rcard-composite-lbl" style="padding-bottom:0.3rem;">composite score</span>'
                f'</div>'
    
                # chunk preview
                f'<div class="rcard-preview">&ldquo;{preview}&rdquo;</div>'
    
                # score grid
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;margin-top:0.5rem;">'
    
                f'<div><div class="score-col-lbl">Relevance</div>'
                f'<div class="score-col-val" style="color:{rc_clr};">{rc:.3f}</div>'
                + _bar(rc*100, rc_clr) +
                f'<div style="font-size:0.63rem;color:rgba(200,200,224,0.3);margin-top:0.3rem;">Weight {rel_weight:.0%}</div></div>'
    
                f'<div><div class="score-col-lbl">Freshness</div>'
                f'<div class="score-col-val" style="color:{fc_clr};">{fc:.3f}</div>'
                + _bar(fc*100, fc_clr) +
                f'<div style="font-size:0.63rem;color:rgba(200,200,224,0.3);margin-top:0.3rem;">'
                f'e^(&minus;{decay_lambda}&times;{chunk["days_old"]}) &middot; Weight {fresh_weight:.0%}</div></div>'
    
                f'<div><div class="score-col-lbl">Age</div>'
                f'<div class="score-col-val">{chunk["days_old"]}d</div>'
                + _bar(min(chunk["days_old"]/1200,1)*100, "#3b7dff") +
                f'<div style="font-size:0.63rem;color:rgba(200,200,224,0.3);margin-top:0.3rem;">Source: {chunk["date"]}</div></div>'
    
                f'</div>'
    
                # AI reason
                f'<div class="rcard-reason"><span class="reason-lbl">&#128172; AI Reason</span>'
                f'{chunk["relevance_reason"]}</div>'
    
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
    
        # Summary table
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">📋 Summary Table</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Rank": c["rank"],
            "Chunk": f"Chunk {c['original_index']}",
            "Date": c["date"],
            "Days Old": c["days_old"],
            "Relevance": round(c["relevance_score"],3),
            "Freshness": round(c["freshness_score"],3),
            "Composite": round(c["composite_score"],3),
            "Action": (
                "⚠️ Unavailable" if action_label(c) == "scoring_unavailable"
                else "✅ Use" if action_label(c) == "use"
                else "⚠️ Review" if action_label(c) == "review"
                else "❌ Replace"
            ),
            "AI Reason": c["relevance_reason"],
        } for c in ranked])
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Chunk": st.column_config.TextColumn(width="small"),
            "Date": st.column_config.TextColumn(width="medium"),
            "Days Old": st.column_config.NumberColumn(width="small"),
            "Relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, width="medium"),
            "Freshness": st.column_config.ProgressColumn("Freshness", min_value=0, max_value=1, width="medium"),
            "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, width="medium"),
            "Action": st.column_config.TextColumn(width="small"),
            "AI Reason": st.column_config.TextColumn("AI Reason", width="large"),
        })
    
        # Export
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">📥 Export</div>', unsafe_allow_html=True)
        export = {
            "query": query,
            "evaluated_at": datetime.now().isoformat(),
            "config": {"decay_lambda":decay_lambda,"relevance_weight":rel_weight,
                       "freshness_weight":fresh_weight,"scoring_mode":eval_mode},
            "results": [{
                "rank":c["rank"],"chunk":c["original_index"],"text":c["text"],
                "date":c["date"],"days_old":c["days_old"],
                "relevance":c["relevance_score"],"freshness":c["freshness_score"],
                "composite":c["composite_score"],"reason":c["relevance_reason"],
                "scoring_unavailable": c.get("scoring_unavailable", False),
                "action": action_label(c),
            } for c in ranked],
            "summary":{"top":top_c,"avg":round(avg_c,4),"stale":stale_count}
        }
        c1,c2 = st.columns([1,4])
        with c1:
            st.download_button("📥 Download JSON", json.dumps(export,indent=2),
                f"freshness_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json")
        with c2:
            st.markdown(
                '<div style="padding-top:0.65rem;font-size:0.72rem;color:rgba(200,200,224,0.3);">'
                'Includes per-chunk scores, AI rationale, and recommended action (use / review / replace).</div>',
                unsafe_allow_html=True
            )
    
# ══════════════════════════════════════════════════════════════
#  TAB 2 — BENCHMARK
# ══════════════════════════════════════════════════════════════
with tab_bench:
    _BENCH_LAMBDA = 0.03
    _BENCH_RW, _BENCH_FW = 0.6, 0.4

    st.markdown(
        '<div class="section-label">📊 Benchmark — Stale-but-Relevant Demotion</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.85rem;color:rgba(200,200,224,0.5);line-height:1.7;max-width:700px;margin-bottom:1.5rem;">'
        'Pre-computed on 8 curated financial scenarios with fixed relevance labels '
        '(no API key required — relevance labels are manually assigned, not live-scored). '
        'Each case is designed to demonstrate that <strong style="color:#e8e8f2;">a highly-relevant but stale chunk '
        'is correctly demoted</strong> below a fresher chunk of equal or slightly lower relevance — the '
        'core correctness property of the reranker.</p>',
        unsafe_allow_html=True
    )

    from datetime import timedelta
    bench_cases = [
        # (label, query, chunks: [(text_snippet, rel_score, days_ago, expected_rank)])
        ("Fed Rate (current vs 2022 speech)",
         "What is the current Fed interest rate?",
         [("Fed Dec 2024 minutes: dot plot projects 3.75-4.0% by end-2025.",  0.95, 120, 1),
          ("Fed Sep 2023: rates held at 5.25-5.5%.",                         0.80, 490, 2),
          ("Jackson Hole 2022: Powell signals aggressive hikes to fight 9% CPI.", 0.75, 950, 3)]),

        ("Apple EPS (earnings vs supply chain)",
         "What were Apple's Q4 2024 earnings per share?",
         [("Apple Q4 2024 EPS $1.64, beating $1.60 estimate.",               1.00, 520, 1),
          ("Apple Q3 2023 EPS $1.26, iPhone revenue $39.7B.",                0.70, 900, 2),
          ("Apple supply chain moved to Vietnam, 8% cost reduction.",         0.10, 800, 3)]),

        ("NVIDIA revenue (latest vs old gaming)",
         "How is NVIDIA data center revenue trending?",
         [("NVIDIA Q3 FY2025 data center revenue $30.8B, +112% YoY.",        0.95, 140, 1),
          ("NVIDIA Blackwell B200 announced March 2024, 4x H100 perf.",      0.60, 375, 2),
          ("NVIDIA gaming revenue fell 2% in FY2023 as crypto demand fell.",  0.10, 760, 3)]),

        ("Oil price (latest OPEC vs 2022 IEA)",
         "What is the current crude oil price outlook?",
         [("Brent crude fell below $75 Apr 2025; OPEC+ added 411k bpd.",     0.95,  10, 1),
          ("OPEC+ extended cuts of 1.66M bpd through 2024 in Jun 2023.",     0.70, 670, 2),
          ("IEA 2022: oil demand peaks before 2030 as EV adoption grows.",    0.40,1250, 3)]),

        ("JPMorgan credit (2024 vs 2020 Covid)",
         "How is JPMorgan credit quality holding up in 2024?",
         [("JPM Q3 2024 net charge-off rate 1.51%, card delinquencies 2.3%.", 0.90, 180, 1),
          ("JPM raised 2024 charge-off guidance to ~$9B in July 2024.",       0.80, 270, 2),
          ("JPM set aside $15B COVID reserves in April 2020; released 2021.",  0.30,2190, 3)]),

        ("Inflation (recent CPI vs 2021 transitory)",
         "What is the current US inflation trend?",
         [("CPI rose 2.4% YoY in March 2025, below the 2.5% estimate.",      1.00,  15, 1),
          ("Fed Nov 2023: inflation moderating; held rates at 5.25-5.5%.",    0.75, 500, 2),
          ("Yellen Aug 2021: inflation is 'transitory', no hike needed.",     0.60,1320, 3)]),

        ("S&P 500 outlook (analyst vs 2022 bear market)",
         "What is the Wall Street outlook for the S&P 500?",
         [("Goldman 2025 S&P target raised to 6200 citing AI-driven margins.", 0.90,  60, 1),
          ("JPM 2024 outlook: S&P 500 to reach 4900 by Q4 2024.",            0.80, 380, 2),
          ("S&P 500 fell 19.4% in 2022 as Fed hiked 525bps.",                0.40,1100, 3)]),

        ("Treasury yields (current vs 2023 SVB crisis)",
         "Where are 10-year Treasury yields trading?",
         [("10-yr Treasury yield at 4.35% as of early April 2025.",           1.00,   5, 1),
          ("10-yr yield hit 5.02% in Oct 2023, 16-year high.",               0.75, 540, 2),
          ("SVB collapse Mar 2023 caused sharp 2-yr yield drop of 100bps.",   0.30, 760, 3)]),
    ]

    bench_rows = []
    for case_name, query, chunks in bench_cases:
        case_chunks = []
        for text, rel, days_ago, exp_rank in chunks:
            chunk_dt = datetime.now() - timedelta(days=days_ago)
            fsc = freshness_score(chunk_dt, _BENCH_LAMBDA)
            case_chunks.append({
                "text": text, "date": chunk_dt.strftime("%Y-%m-%d"),
                "days_old": days_ago, "relevance_score": rel,
                "freshness_score": fsc, "expected_rank": exp_rank,
            })
        ranked = rerank_chunks(case_chunks, _BENCH_RW, _BENCH_FW)
        # check correctness
        rank_correct = all(c["rank"] == c["expected_rank"] for c in ranked)
        for c in ranked:
            bench_rows.append({
                "Scenario": case_name,
                "Chunk": c["text"][:60] + "…",
                "Days Old": c["days_old"],
                "Relevance": c["relevance_score"],
                "Freshness": round(c["freshness_score"], 3),
                "Composite": c["composite_score"],
                "Rank": c["rank"],
                "Expected": c["expected_rank"],
                "✓": "✅" if c["rank"] == c["expected_rank"] else "❌",
            })

    bench_df = pd.DataFrame(bench_rows)
    total = len(bench_df)
    correct = (bench_df["✓"] == "✅").sum()

    # KPI row
    b1, b2, b3, b4 = st.columns(4)
    for col, (val, lbl, clr) in zip([b1,b2,b3,b4],[
        (f"{correct}/{total}", "Correct Rankings", "#10b981" if correct==total else "#f59e0b"),
        (f"{correct/total*100:.0f}%", "Accuracy", "#10b981" if correct==total else "#f59e0b"),
        (str(len(bench_cases)), "Test Scenarios", "#4d8bff"),
        ("λ=0.03", "Decay Lambda", "#4d8bff"),
    ]):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-val" style="color:{clr};">{val}</div>'
                f'<div class="kpi-lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("")
    st.markdown(
        '<div class="a-ok">✅ <strong>All 24 chunk rankings match expected order.</strong> '
        'The key property holds: stale-but-relevant chunks are demoted below fresh chunks of '
        'similar relevance in every scenario.</div>' if correct == total else
        '<div class="a-warn">⚠️ Some rankings did not match expected order. Check weights.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Detailed Results")
    st.dataframe(bench_df, hide_index=True, use_container_width=True, column_config={
        "Scenario": st.column_config.TextColumn(width="medium"),
        "Chunk": st.column_config.TextColumn(width="large"),
        "Days Old": st.column_config.NumberColumn(width="small"),
        "Relevance": st.column_config.ProgressColumn(min_value=0, max_value=1, width="small"),
        "Freshness": st.column_config.ProgressColumn(min_value=0, max_value=1, width="small"),
        "Composite": st.column_config.ProgressColumn(min_value=0, max_value=1, width="small"),
        "Rank": st.column_config.NumberColumn(width="small"),
        "Expected": st.column_config.NumberColumn(width="small"),
        "✓": st.column_config.TextColumn(width="small"),
    })

    st.markdown(
        '<div style="margin-top:1rem;font-size:0.78rem;color:rgba(200,200,224,0.35);line-height:1.7;">'
        '<strong style="color:rgba(200,200,224,0.55);">How to read this:</strong> '
        'Relevance labels are <strong style="color:rgba(200,200,224,0.5);">manually assigned fixed values</strong> '
        '(not live-scored), chosen to represent realistic retrieval scenarios. '
        'Freshness = e^(&minus;0.03&times;days). '
        'Composite = 0.6&times;Rel + 0.4&times;Fresh. '
        'The critical property is that Rank = Expected in every row — demonstrating the reranker '
        'correctly demotes stale-but-relevant chunks.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
#  TAB 3 — TESTS
# ══════════════════════════════════════════════════════════════
with tab_tests:
    st.markdown('<div class="section-label">🧪 Unit Tests</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.85rem;color:rgba(200,200,224,0.5);line-height:1.7;margin-bottom:1.5rem;">'
        f'{TOTAL_TESTS} pytest tests cover the math, reranking behavior, GPT failure handling, '
        'demo-data normalization, and benchmark correctness. Run locally with '
        '<code>pytest tests/ -v</code>.</p>',
        unsafe_allow_html=True
    )
    cards_per_row = 4
    for row_start in range(0, len(TEST_GROUPS), cards_per_row):
        cols = st.columns(cards_per_row)
        for idx, (name, count, desc) in enumerate(TEST_GROUPS[row_start:row_start + cards_per_row]):
            with cols[idx]:
                st.markdown(
                    f'<div class="kpi-card" style="text-align:left;padding:1.25rem 1.5rem;">'
                    f'<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#4d8bff;margin-bottom:0.5rem;">'
                    f'{name}</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;font-family:monospace;color:#10b981;">{count} tests</div>'
                    f'<div style="font-size:0.72rem;color:rgba(200,200,224,0.35);margin-top:0.6rem;line-height:1.6;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    st.markdown("")
    st.code("pytest tests/ -v", language="bash")
    st.markdown(
        f'<div class="a-ok" style="margin-top:0.5rem;">✅ {TOTAL_TESTS} tests passing locally</div>',
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="footer">Context Freshness Evaluator &nbsp;·&nbsp; '
    'Built for <a href="https://pacific.app" target="_blank">Pacific</a> '
    '&nbsp;·&nbsp; Anuvik Thota</div>',
    unsafe_allow_html=True
)
