"""
Context Freshness Evaluator — Streamlit App
Built for Pacific Take-Home

Scores and reranks financial context chunks by relevance (GPT-4o)
and freshness (exponential time decay).
"""

import json, os, math
import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from evaluator.freshness import freshness_score, days_old
from evaluator.relevance import score_relevance
from evaluator.reranker import rerank_chunks, get_stale_chunks
from config import RELEVANCE_WEIGHT, FRESHNESS_WEIGHT, DECAY_LAMBDA, GPT_MODEL, MAX_TOKENS_JUDGE

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
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp { font-family:'Inter',sans-serif; background:#07070d; color:#e8e8f2; }

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,header,[data-testid="stToolbar"] { visibility:hidden; }
[data-testid="collapsedControl"] { display:none; }

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
    p = Path(__file__).parent / "demo_data" / "examples.json"
    return json.load(open(p)) if p.exists() else []

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

def bar(pct, color):
    return (
        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;height:5px;overflow:hidden;margin-top:0.3rem;">'
        f'<div style="width:{pct:.0f}%;height:100%;border-radius:6px;background:{color};"></div></div>'
    )

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
    st.session_state.active_demo   = demo["name"]
    st.session_state.query_input   = demo["query"]
    for i, c in enumerate(demo["chunks"]):
        st.session_state[f"chunk_text_{i}"] = c["text"]
        st.session_state[f"chunk_date_{i}"] = datetime.strptime(c["date"],"%Y-%m-%d").date()
    st.session_state.results   = None
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
#  SCENARIO PICKER  (center stage)
# ══════════════════════════════════════════════════════════════
demos = load_demo_data()
active = st.session_state.active_demo

st.markdown('<div class="section-label">🎬 Quick Start — Choose a Scenario</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.82rem;color:rgba(200,200,224,0.4);margin:-0.4rem 0 1.2rem;">'
    'Click any card to instantly load a demo query and 3 context chunks — or scroll down to enter your own.</p>',
    unsafe_allow_html=True
)

# Render 5 scenario cards via columns
cols = st.columns(5, gap="medium")
for idx, demo in enumerate(demos):
    with cols[idx]:
        is_active = (demo["name"] == active)
        active_cls = "active-card" if is_active else ""
        check = "✓ " if is_active else ""
        # strip emoji from name for pill label
        short_name = demo["name"].split(" ",1)[1] if demo["name"][0] in "🍎🏦⚡🛢️💳" else demo["name"]
        icon = demo["name"][0] if demo["name"][0] in "🍎🏦⚡🛢️💳" else "📄"
        # Truncate query for preview
        q_preview = demo["query"][:55] + "…" if len(demo["query"]) > 55 else demo["query"]

        st.markdown(
            f'<div class="sc-card {active_cls}">'
            f'<span class="sc-icon">{icon}</span>'
            f'<div class="sc-name">{check}{short_name}</div>'
            f'<div class="sc-query">{q_preview}</div>'
            f'<span class="sc-pill">{len(demo["chunks"])} chunks</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("Load", key=f"sc_{idx}", use_container_width=True):
            load_scenario(demo)
            st.rerun()

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
        width="stretch", type="primary", disabled=not ready
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
    for idx, chunk in enumerate(chunks_input):
        fsc = freshness_score(chunk["chunk_date_obj"], decay_lambda)
        age = days_old(chunk["chunk_date_obj"])
        rel = (score_relevance(query, chunk["text"], client,
                               model=GPT_MODEL, max_tokens=MAX_TOKENS_JUDGE)
               if client else simulate_relevance(query, chunk["text"]))
        scored.append({
            "text": chunk["text"], "date": chunk["date"], "days_old": age,
            "freshness_score": fsc, "relevance_score": rel["score"],
            "relevance_reason": rel["reason"], "original_index": chunk["index"]+1,
        })
        prog.progress((idx+1)/len(chunks_input), text=f"Scored chunk {idx+1}/{len(chunks_input)}")

    ranked = rerank_chunks(scored, rel_weight, fresh_weight)
    stale  = get_stale_chunks(ranked)
    st.session_state.update({
        "results": ranked, "stale_count": len(stale), "evaluated": True,
        "eval_mode": "GPT-4o" if client else "Keyword Match"
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
        action = "✅ Use" if comp>=0.7 else "⚠️ Review" if comp>=0.4 else "❌ Replace"

        html = (
            f'<div class="rcard {cc} anim">'

            # top row
            f'<div style="display:flex;align-items:center;gap:0.85rem;margin-bottom:0.6rem;">'
            f'<span class="rcard-rank">RANK #{chunk["rank"]}</span>'
            f'<span class="rcard-meta">Chunk {chunk["original_index"]} &middot; {chunk["date"]} &middot; {chunk["days_old"]}d old {emoji}</span>'
            f'<span style="margin-left:auto;font-size:0.75rem;font-weight:600;'
            f'padding:0.25rem 0.75rem;border-radius:8px;'
            f'background:rgba(255,255,255,0.04);color:rgba(200,200,224,0.6);">{action}</span>'
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
        "Action": "✅ Use" if c["composite_score"]>=0.7 else "⚠️ Review" if c["composite_score"]>=0.4 else "❌ Replace",
        "AI Reason": c["relevance_reason"],
    } for c in ranked])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
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
            "action":"use" if c["composite_score"]>=0.7 else "review" if c["composite_score"]>=0.4 else "replace",
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

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="footer">Context Freshness Evaluator &nbsp;·&nbsp; '
    'Built for <a href="https://pacific.app" target="_blank">Pacific</a> '
    '&nbsp;·&nbsp; Anuvik Thota</div>',
    unsafe_allow_html=True
)
