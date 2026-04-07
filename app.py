"""
Context Freshness Evaluator — Streamlit App
Built for Pacific Take-Home

Scores and reranks financial context chunks by relevance (GPT-4o judge)
and freshness (exponential time decay). Helps identify stale context
that could cause LLM hallucinations in financial analysis.
"""

import json
import os
import math
import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

from evaluator.freshness import freshness_score, days_old
from evaluator.relevance import score_relevance
from evaluator.reranker import rerank_chunks, get_stale_chunks
from config import (
    RELEVANCE_WEIGHT, FRESHNESS_WEIGHT, DECAY_LAMBDA,
    GPT_MODEL, MAX_TOKENS_JUDGE, TOP_K
)

# ──────────────────────────────────────────────────────────────
# API Key — load silently from environment, no UI exposure
# ──────────────────────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    try:
        _api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

USE_GPT = bool(_api_key and _api_key.startswith("sk-"))

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context Freshness Evaluator · Pacific",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0a0a0f;
    }

    /* ── Header ────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0d1b6e 0%, #1a2ea0 35%, #1a56ff 70%, #005fff 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(26, 86, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero h1 {
        color: #fff;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.4rem;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .hero p {
        color: rgba(255,255,255,0.75);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
        max-width: 560px;
        line-height: 1.6;
    }
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: rgba(255,255,255,0.9);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Step Guide ─────────────────────────── */
    .steps-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .step-card {
        flex: 1;
        background: linear-gradient(145deg, #12121c, #0f0f18);
        border: 1px solid rgba(26, 86, 255, 0.12);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        transition: border-color 0.25s;
    }
    .step-card:hover { border-color: rgba(26, 86, 255, 0.35); }
    .step-num {
        background: linear-gradient(135deg, #1a56ff, #3b7dff);
        color: #fff;
        font-size: 0.85rem;
        font-weight: 700;
        width: 30px;
        height: 30px;
        min-width: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e8e8f0;
        margin-bottom: 0.2rem;
    }
    .step-desc {
        font-size: 0.75rem;
        color: rgba(240,240,240,0.45);
        line-height: 1.5;
    }

    /* ── Mode Banner ────────────────────────── */
    .mode-gpt {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #10b981;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .mode-demo {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #f59e0b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ── Chunk Cards ────────────────────────── */
    .chunk-card {
        background: linear-gradient(145deg, #12121c, #0f0f18);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.25s;
    }
    .chunk-card:hover { border-color: rgba(26, 86, 255, 0.25); }
    .chunk-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: rgba(240,240,240,0.4);
        margin-bottom: 0.4rem;
    }

    /* ── Freshness Badge (inline) ───────────── */
    .f-pill-fresh  { background:rgba(16,185,129,0.12); color:#10b981; padding:0.2rem 0.65rem; border-radius:20px; font-size:0.7rem; font-weight:600; display:inline-block; }
    .f-pill-mid    { background:rgba(245,158,11,0.12); color:#f59e0b; padding:0.2rem 0.65rem; border-radius:20px; font-size:0.7rem; font-weight:600; display:inline-block; }
    .f-pill-stale  { background:rgba(239,68,68,0.12);  color:#ef4444; padding:0.2rem 0.65rem; border-radius:20px; font-size:0.7rem; font-weight:600; display:inline-block; }

    /* ── Score Legend ───────────────────────── */
    .legend {
        background: linear-gradient(145deg, #12121c, #0f0f18);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0 0.5rem;
        font-size: 0.75rem;
        color: rgba(240,240,240,0.5);
        line-height: 1.8;
    }
    .legend strong { color: rgba(240,240,240,0.8); }

    /* ── Result Cards ───────────────────────── */
    .result-card {
        background: linear-gradient(145deg, #12121c, #0f0f18);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
        transition: border-color 0.25s, transform 0.2s;
    }
    .result-card:hover { border-color: rgba(26,86,255,0.25); transform: translateY(-1px); }
    .result-top  { border-left: 4px solid #10b981; }
    .result-mid  { border-left: 4px solid #f59e0b; }
    .result-low  { border-left: 4px solid #ef4444; }

    .rank-chip {
        background: linear-gradient(135deg, #1a56ff, #3b7dff);
        color: #fff;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 8px;
        letter-spacing: 0.03em;
    }
    .composite-score {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .score-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(240,240,240,0.4);
        margin-bottom: 0.2rem;
    }
    .score-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e8e8f0;
        font-family: 'JetBrains Mono', monospace;
    }
    .score-bar-wrap { background: rgba(255,255,255,0.05); border-radius:6px; height:5px; overflow:hidden; margin-top:0.3rem; }
    .score-bar-fill { height:100%; border-radius:6px; }
    .bar-green  { background: linear-gradient(90deg,#10b981,#34d399); }
    .bar-yellow { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
    .bar-red    { background: linear-gradient(90deg,#ef4444,#f87171); }
    .bar-blue   { background: linear-gradient(90deg,#1a56ff,#3b7dff); }

    .reason-strip {
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        font-size: 0.78rem;
        color: rgba(240,240,240,0.5);
        font-style: italic;
        line-height: 1.5;
    }
    .reason-strip span { color: rgba(240,240,240,0.35); font-style: normal; text-transform: uppercase; font-size:0.65rem; letter-spacing:0.06em; }

    /* ── Metric Cards ───────────────────────── */
    .kpi-card {
        background: linear-gradient(145deg, #12121c, #0f0f18);
        border: 1px solid rgba(26,86,255,0.12);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.25s;
    }
    .kpi-card:hover { border-color: rgba(26,86,255,0.3); transform: translateY(-2px); }
    .kpi-val { font-size: 2rem; font-weight: 800; color: #1a56ff; margin:0; line-height:1.1; font-family:'JetBrains Mono',monospace; }
    .kpi-lbl { font-size:0.7rem; color:rgba(240,240,240,0.4); text-transform:uppercase; letter-spacing:0.08em; margin-top:0.4rem; }

    /* ── Warning / Info boxes ───────────────── */
    .warn-box {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 12px;
        padding: 0.85rem 1.2rem;
        font-size: 0.82rem;
        color: #fca5a5;
        margin: 0.75rem 0;
    }
    .info-box {
        background: rgba(26,86,255,0.07);
        border: 1px solid rgba(26,86,255,0.18);
        border-radius: 12px;
        padding: 0.85rem 1.2rem;
        font-size: 0.82rem;
        color: rgba(180,200,255,0.85);
        margin: 0.75rem 0;
    }

    /* ── Decay Visualiser ───────────────────── */
    .decay-bar-wrap { display:flex; align-items:center; gap:0.5rem; margin:0.3rem 0; }
    .decay-bar-label { font-size:0.7rem; color:rgba(240,240,240,0.45); width:48px; text-align:right; }
    .decay-bar-bg { flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:7px; overflow:hidden; }
    .decay-bar-inner { height:100%; border-radius:4px; }

    /* ── Sidebar ────────────────────────────── */
    .sidebar-section-title {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(240,240,240,0.3);
        margin: 1rem 0 0.5rem;
    }

    /* ── Footer ─────────────────────────────── */
    .footer { text-align:center; padding:1.5rem 0 0.5rem; color:rgba(240,240,240,0.2); font-size:0.72rem; }
    .footer a { color:#1a56ff; text-decoration:none; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Animations */
    @keyframes fadeUp {
        from { opacity:0; transform:translateY(14px); }
        to   { opacity:1; transform:translateY(0); }
    }
    .anim { animation: fadeUp 0.4s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    p = Path(__file__).parent / "demo_data" / "examples.json"
    return json.load(open(p)) if p.exists() else []


def simulate_relevance(query: str, chunk_text: str) -> dict:
    """Keyword-overlap fallback when no API key is available."""
    stop = {'the','a','an','is','are','was','were','what','how','in','of','to',
            'and','for','at','by','on','with','its','that','this','from','as',
            'it','be','has','had','does','do','will','about','their','which'}
    qw = set(query.lower().split()) - stop
    cw = set(chunk_text.lower().split()) - stop
    if not qw:
        return {"score": 0.5, "reason": "Unable to parse query"}
    overlap = qw & cw
    score = min(1.0, len(overlap) / max(len(qw), 1) * 1.2)
    reason = f"Keyword overlap: {', '.join(list(overlap)[:4])}" if overlap else "No keyword overlap"
    return {"score": round(score, 4), "reason": reason}


def freshness_color(score: float):
    if score >= 0.7: return "bar-green", "f-pill-fresh", "🟢"
    if score >= 0.3: return "bar-yellow", "f-pill-mid", "🟡"
    return "bar-red", "f-pill-stale", "🔴"


def composite_color(score: float):
    if score >= 0.7: return "result-top", "bar-green", "#10b981"
    if score >= 0.4: return "result-mid", "bar-yellow", "#f59e0b"
    return "result-low", "bar-red", "#ef4444"


def decay_preview_html(lam: float) -> str:
    """Render a mini decay curve as HTML bars."""
    checkpoints = [("Today", 0), ("1 week", 7), ("1 month", 30),
                   ("3 months", 90), ("6 months", 180), ("1 year", 365)]
    rows = ""
    for label, d in checkpoints:
        pct = round(math.exp(-lam * d) * 100, 1)
        color = "#10b981" if pct > 60 else "#f59e0b" if pct > 25 else "#ef4444"
        rows += f"""
        <div class="decay-bar-wrap">
          <div class="decay-bar-label">{label}</div>
          <div class="decay-bar-bg">
            <div class="decay-bar-inner" style="width:{pct}%;background:{color};"></div>
          </div>
          <div style="font-size:0.68rem;color:rgba(240,240,240,0.5);width:38px;">{pct}%</div>
        </div>"""
    return rows


# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = None
if "evaluated"  not in st.session_state: st.session_state.evaluated  = False
if "query_input" not in st.session_state: st.session_state.query_input = ""


def _load_demo():
    sel = st.session_state.get("demo_sel", "")
    if not sel or sel.startswith("—"): return
    demos = load_demo_data()
    d = next((x for x in demos if x["name"] == sel), None)
    if not d: return
    st.session_state.query_input = d["query"]
    for i, c in enumerate(d["chunks"]):
        st.session_state[f"chunk_text_{i}"] = c["text"]
        st.session_state[f"chunk_date_{i}"] = datetime.strptime(c["date"], "%Y-%m-%d").date()
    st.session_state.results   = None
    st.session_state.evaluated = False


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Engine Status ──────────────────────────────────────────
    if USE_GPT:
        st.markdown("""
        <div class="mode-gpt">
          🤖 <strong>GPT-4o Active</strong> — Full AI relevance scoring enabled
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mode-demo">
          ⚡ <strong>Demo Mode</strong> — Add OPENAI_API_KEY to .env for GPT-4o scoring
        </div>""", unsafe_allow_html=True)

    # ── Demo Scenarios ─────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">📋 Demo Scenarios</div>', unsafe_allow_html=True)
    demos = load_demo_data()
    demo_names = ["— Choose a scenario —"] + [d["name"] for d in demos]
    st.selectbox("", demo_names, key="demo_sel", on_change=_load_demo,
                 label_visibility="collapsed")
    st.caption("Instantly loads a pre-built financial query + 3 context chunks.")

    st.divider()

    # ── Scoring Config ─────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">🎛 Scoring Parameters</div>', unsafe_allow_html=True)

    decay_lambda = st.slider(
        "Freshness Decay Speed (λ)",
        min_value=0.005, max_value=0.10,
        value=DECAY_LAMBDA, step=0.005,
        help="Higher λ = faster freshness decay. λ=0.03 → ~50% freshness at 23 days."
    )

    rel_weight = st.slider(
        "Relevance Weight",
        min_value=0.1, max_value=0.9,
        value=RELEVANCE_WEIGHT, step=0.05,
        help="Composite = (Relevance × this) + (Freshness × 1-this)"
    )
    fresh_weight = round(1.0 - rel_weight, 2)
    st.markdown(
        f'<div style="font-size:0.75rem;color:rgba(240,240,240,0.4);margin-top:-0.5rem;">'
        f'Freshness Weight: <strong style="color:rgba(240,240,240,0.7);">{fresh_weight}</strong>'
        f' (auto-balanced)</div>', unsafe_allow_html=True
    )

    st.divider()

    # ── Decay Visualiser ───────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">📉 Freshness Decay Preview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:linear-gradient(145deg,#12121c,#0f0f18);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:0.9rem 1rem;">'
        f'{decay_preview_html(decay_lambda)}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.caption(f"Formula: score = e^(−{decay_lambda} × days_old)")

    st.divider()
    st.markdown(
        '<div class="footer">Built for <a href="https://pacific.app" target="_blank">Pacific</a>'
        ' · Context Freshness Evaluator v1.0</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ Context Freshness Evaluator</h1>
  <p>Score and rerank financial context chunks by <strong>relevance</strong> and
  <strong>time freshness</strong> — before stale data causes hallucinations in your AI pipeline.</p>
  <div class="hero-badges">
    <span class="hero-badge">Context Management</span>
    <span class="hero-badge">Evals</span>
    <span class="hero-badge">RAG Reranking</span>
    <span class="hero-badge">TTFT Optimization</span>
    <span class="hero-badge">Finance AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── How It Works ───────────────────────────────────────────────
st.markdown("""
<div class="steps-row">
  <div class="step-card">
    <div class="step-num">1</div>
    <div>
      <div class="step-title">Pick a Scenario or Enter Your Own</div>
      <div class="step-desc">Choose a demo from the sidebar or type any financial query and paste context chunks below.</div>
    </div>
  </div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div>
      <div class="step-title">Set Chunk Dates</div>
      <div class="step-desc">Each chunk needs its source date — this drives the freshness score. Older = more decay.</div>
    </div>
  </div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div>
      <div class="step-title">Evaluate & Rerank</div>
      <div class="step-desc">We score relevance with GPT-4o and freshness with exponential decay, then composite-rerank.</div>
    </div>
  </div>
  <div class="step-card">
    <div class="step-num">4</div>
    <div>
      <div class="step-title">Act on Results</div>
      <div class="step-desc">Replace stale chunks (🔴), keep fresh high-relevance ones (🟢), tune weights for your use case.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Score Legend ───────────────────────────────────────────────
st.markdown("""
<div class="legend">
  <strong>Understanding the scores:</strong>&nbsp;&nbsp;
  <strong style="color:#10b981;">🟢 Composite ≥ 0.7</strong> — High relevance + fresh → use this chunk &nbsp;|&nbsp;
  <strong style="color:#f59e0b;">🟡 0.4 – 0.7</strong> — Moderate — review before using &nbsp;|&nbsp;
  <strong style="color:#ef4444;">🔴 &lt; 0.4</strong> — Stale or irrelevant — avoid or replace &nbsp;|&nbsp;
  <strong>Freshness = e^(−λ·days)</strong>&nbsp;·&nbsp;
  <strong>Composite = (Relevance × {rw}) + (Freshness × {fw})</strong>
</div>
""".format(rw=rel_weight, fw=fresh_weight), unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
#  QUERY + CHUNKS INPUT
# ══════════════════════════════════════════════════════════════
st.markdown("### 🔍 Financial Query")
query = st.text_input(
    "query",
    placeholder="e.g., What were Apple's Q4 2024 earnings per share?",
    key="query_input",
    label_visibility="collapsed"
)

st.markdown("### 📄 Context Chunks")
st.caption(
    "Paste up to **3 context chunks** from your RAG pipeline with their source dates. "
    "The freshness badge updates instantly as you pick dates."
)

chunks_input = []
cols = st.columns(3, gap="medium")

for i in range(3):
    with cols[i]:
        # Default date: today
        default_date = st.session_state.get(f"chunk_date_{i}", date.today())

        st.markdown(
            f'<div class="chunk-label">Chunk {i+1}</div>',
            unsafe_allow_html=True
        )
        chunk_text = st.text_area(
            f"chunk_text_{i}",
            height=145,
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
            age   = days_old(datetime.combine(chunk_date, datetime.min.time()))
            fsc   = freshness_score(datetime.combine(chunk_date, datetime.min.time()), decay_lambda)
            _, pill_cls, emoji = freshness_color(fsc)
            st.markdown(
                f'<div style="margin-top:0.3rem;">'
                f'<span class="{pill_cls}">{emoji} {age}d old · Freshness {fsc:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            chunks_input.append({
                "text": chunk_text.strip(),
                "date": chunk_date.isoformat(),
                "chunk_date_obj": datetime.combine(chunk_date, datetime.min.time()),
                "index": i
            })
        else:
            st.markdown(
                '<div style="margin-top:0.3rem;font-size:0.72rem;color:rgba(240,240,240,0.25);">Awaiting input…</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════
#  EVALUATE BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    ready = bool(query.strip()) and len(chunks_input) > 0
    evaluate = st.button(
        "⚡  Evaluate & Rerank",
        width="stretch",
        type="primary",
        disabled=not ready,
    )
    if not ready:
        st.markdown(
            '<div style="text-align:center;font-size:0.72rem;color:rgba(240,240,240,0.25);margin-top:0.3rem;">'
            'Add a query and at least one chunk to proceed.</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════
#  EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════
if evaluate and ready:
    client = None
    if USE_GPT:
        from openai import OpenAI
        client = OpenAI(api_key=_api_key)

    scored = []
    prog = st.progress(0, text="Scoring chunks…")

    for idx, chunk in enumerate(chunks_input):
        fsc  = freshness_score(chunk["chunk_date_obj"], decay_lambda)
        age  = days_old(chunk["chunk_date_obj"])
        if client:
            rel = score_relevance(query, chunk["text"], client,
                                  model=GPT_MODEL, max_tokens=MAX_TOKENS_JUDGE)
        else:
            rel = simulate_relevance(query, chunk["text"])

        scored.append({
            "text": chunk["text"],
            "date": chunk["date"],
            "days_old": age,
            "freshness_score": fsc,
            "relevance_score": rel["score"],
            "relevance_reason": rel["reason"],
            "original_index": chunk["index"] + 1,
        })
        prog.progress((idx + 1) / len(chunks_input),
                      text=f"Scored chunk {idx+1}/{len(chunks_input)}")

    ranked = rerank_chunks(scored, rel_weight, fresh_weight)
    stale  = get_stale_chunks(ranked)

    st.session_state.results    = ranked
    st.session_state.stale_count = len(stale)
    st.session_state.evaluated  = True
    st.session_state.eval_mode  = "GPT-4o" if client else "Keyword Match"
    prog.empty()

# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if st.session_state.evaluated and st.session_state.results:
    ranked     = st.session_state.results
    stale_count = st.session_state.stale_count
    eval_mode  = st.session_state.get("eval_mode", "Unknown")

    st.divider()
    st.markdown("## 📊 Evaluation Results")

    # ── Mode badge ──────────────────────────────────────────────
    if eval_mode == "GPT-4o":
        st.markdown(
            '<div class="mode-gpt">🤖 <strong>Scored with GPT-4o</strong> — '
            'Relevance explanations are AI-generated</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mode-demo">⚡ <strong>Scored with keyword matching (demo mode)</strong> — '
            'Add your OpenAI key to .env for AI relevance scoring</div>',
            unsafe_allow_html=True
        )

    # ── Stale warning ───────────────────────────────────────────
    if stale_count > 0:
        st.markdown(
            f'<div class="warn-box">⚠️ <strong>{stale_count} stale chunk'
            f'{"s" if stale_count > 1 else ""} detected</strong> (freshness &lt; 0.3). '
            f'Serving stale financial data to an LLM can cause outdated or hallucinated answers. '
            f'Consider replacing these chunks with more recent sources.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-box">✅ <strong>All chunks are reasonably fresh.</strong> '
            'No staleness warnings for the current decay settings.</div>',
            unsafe_allow_html=True
        )

    # ── KPI row ─────────────────────────────────────────────────
    avg_composite = sum(c["composite_score"] for c in ranked) / len(ranked)
    top_score     = ranked[0]["composite_score"]
    top_rank_chunk = ranked[0]["original_index"]

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (f"{top_score:.2f}", "Top Composite Score", "#1a56ff"),
        (f"{avg_composite:.2f}", "Average Composite", "#1a56ff"),
        (f"#{top_rank_chunk}", "Best Chunk", "#10b981"),
        (str(stale_count), "Stale Chunks (&lt; 0.3)", "#ef4444" if stale_count else "#10b981"),
    ]
    for col, (val, label, color) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'  <div class="kpi-val" style="color:{color};">{val}</div>'
                f'  <div class="kpi-lbl">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("")

    # ── Detailed result cards ───────────────────────────────────
    def _bar(pct: float, color: str) -> str:
        """Inline-styled progress bar."""
        return (
            f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;'
            f'height:6px;overflow:hidden;margin-top:0.3rem;">'
            f'<div style="width:{pct:.0f}%;height:100%;border-radius:6px;background:{color};"></div>'
            f'</div>'
        )

    for chunk in ranked:
        composite = chunk["composite_score"]
        card_cls, _, clr = composite_color(composite)
        _, _, emoji      = freshness_color(chunk["freshness_score"])
        preview = chunk["text"][:140] + ("…" if len(chunk["text"]) > 140 else "")

        rel_pct   = chunk["relevance_score"] * 100
        fre_pct   = chunk["freshness_score"] * 100
        age_pct   = min(chunk["days_old"] / 1200, 1) * 100
        rel_color = "#10b981" if chunk["relevance_score"] >= 0.7 else "#f59e0b" if chunk["relevance_score"] >= 0.3 else "#ef4444"
        fre_color = "#10b981" if chunk["freshness_score"] >= 0.7 else "#f59e0b" if chunk["freshness_score"] >= 0.3 else "#ef4444"

        card_html = (
            f'<div class="result-card {card_cls} anim">'
            f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.9rem;">'
            f'<span class="rank-chip">RANK #{chunk["rank"]}</span>'
            f'<span style="font-size:0.78rem;font-weight:600;color:rgba(240,240,240,0.6);">'
            f'Chunk {chunk["original_index"]} &nbsp;&middot;&nbsp; {chunk["date"]} &nbsp;&middot;&nbsp; {chunk["days_old"]}d old {emoji}'
            f'</span>'
            f'<span style="margin-left:auto;font-family:monospace;font-size:1.75rem;font-weight:800;color:{clr};">'
            f'{composite:.3f}</span>'
            f'<span style="font-size:0.65rem;color:rgba(240,240,240,0.4);text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-left:0.25rem;">composite</span>'
            f'</div>'
            f'<div style="font-size:0.83rem;color:rgba(240,240,240,0.5);line-height:1.6;'
            f'font-style:italic;margin-bottom:1.1rem;">&ldquo;{preview}&rdquo;</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;">'

            f'<div>'
            f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:rgba(240,240,240,0.4);margin-bottom:0.2rem;">Relevance</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#e8e8f0;font-family:monospace;">'
            f'{chunk["relevance_score"]:.3f}</div>'
            + _bar(rel_pct, rel_color) +
            f'<div style="font-size:0.68rem;color:rgba(240,240,240,0.3);margin-top:0.3rem;">'
            f'Weight {rel_weight:.0%} of composite</div></div>'

            f'<div>'
            f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:rgba(240,240,240,0.4);margin-bottom:0.2rem;">Freshness</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#e8e8f0;font-family:monospace;">'
            f'{chunk["freshness_score"]:.3f}</div>'
            + _bar(fre_pct, fre_color) +
            f'<div style="font-size:0.68rem;color:rgba(240,240,240,0.3);margin-top:0.3rem;">'
            f'e^(&minus;{decay_lambda}&times;{chunk["days_old"]}) &middot; Weight {fresh_weight:.0%}</div></div>'

            f'<div>'
            f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:rgba(240,240,240,0.4);margin-bottom:0.2rem;">Days Old</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#e8e8f0;font-family:monospace;">'
            f'{chunk["days_old"]}</div>'
            + _bar(age_pct, "#3b7dff") +
            f'<div style="font-size:0.68rem;color:rgba(240,240,240,0.3);margin-top:0.3rem;">'
            f'Source: {chunk["date"]}</div></div>'

            f'</div>'
            f'<div style="margin-top:1rem;padding-top:0.85rem;border-top:1px solid rgba(255,255,255,0.05);'
            f'font-size:0.78rem;color:rgba(240,240,240,0.5);font-style:italic;line-height:1.5;">'
            f'<span style="color:rgba(240,240,240,0.3);font-style:normal;text-transform:uppercase;'
            f'font-size:0.65rem;letter-spacing:0.06em;">&#128172; AI Reason &nbsp;</span>'
            f'{chunk["relevance_reason"]}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


    # ── Summary Table ──────────────────────────────────────────
    st.markdown("### 📋 Summary Table")
    df = pd.DataFrame([{
        "Rank":       c["rank"],
        "Chunk":      f"Chunk {c['original_index']}",
        "Date":       c["date"],
        "Days Old":   c["days_old"],
        "Relevance":  round(c["relevance_score"], 3),
        "Freshness":  round(c["freshness_score"], 3),
        "Composite":  round(c["composite_score"], 3),
        "Status":     "🟢 Use" if c["composite_score"] >= 0.7 else
                      "🟡 Review" if c["composite_score"] >= 0.4 else "🔴 Replace",
        "AI Reason":  c["relevance_reason"],
    } for c in ranked])

    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Rank":     st.column_config.NumberColumn(width="small"),
            "Chunk":    st.column_config.TextColumn(width="small"),
            "Date":     st.column_config.TextColumn(width="medium"),
            "Days Old": st.column_config.NumberColumn(width="small"),
            "Relevance":st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, width="medium"),
            "Freshness":st.column_config.ProgressColumn("Freshness", min_value=0, max_value=1, width="medium"),
            "Composite":st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, width="medium"),
            "Status":   st.column_config.TextColumn(width="small"),
            "AI Reason":st.column_config.TextColumn("AI Reason", width="large"),
        }
    )

    # ── Export ─────────────────────────────────────────────────
    st.markdown("### 📥 Export Results")
    export = {
        "query": query,
        "evaluated_at": datetime.now().isoformat(),
        "config": {
            "decay_lambda": decay_lambda,
            "relevance_weight": rel_weight,
            "freshness_weight": fresh_weight,
            "scoring_mode": eval_mode,
        },
        "results": [{
            "rank": c["rank"],
            "chunk_index": c["original_index"],
            "text": c["text"],
            "date": c["date"],
            "days_old": c["days_old"],
            "relevance_score": c["relevance_score"],
            "freshness_score": c["freshness_score"],
            "composite_score": c["composite_score"],
            "ai_reason": c["relevance_reason"],
            "action": "use" if c["composite_score"] >= 0.7 else
                      "review" if c["composite_score"] >= 0.4 else "replace",
        } for c in ranked],
        "summary": {
            "top_composite": top_score,
            "avg_composite": round(avg_composite, 4),
            "stale_chunks": stale_count,
        }
    }
    c1, c2 = st.columns([1, 3])
    with c1:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(export, indent=2),
            file_name=f"freshness_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with c2:
        st.markdown(
            '<div style="padding-top:0.6rem;font-size:0.75rem;color:rgba(240,240,240,0.35);">'
            'JSON includes per-chunk scores, AI reasons, and recommended action (use / review / replace).'
            '</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div class="footer">Context Freshness Evaluator · Built for '
    '<a href="https://pacific.app" target="_blank">Pacific</a> · Anuvik Thota</div>',
    unsafe_allow_html=True
)
