"""
Context Freshness Evaluator — Streamlit App
Built for Pacific Take-Home

Scores and reranks financial context chunks by relevance (GPT-4o judge)
and freshness (exponential time decay). Helps identify stale context
that could cause LLM hallucinations in financial analysis.
"""

import json
import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path

from evaluator.freshness import freshness_score, days_old
from evaluator.relevance import score_relevance
from evaluator.reranker import rerank_chunks, get_stale_chunks
from config import (
    RELEVANCE_WEIGHT, FRESHNESS_WEIGHT, DECAY_LAMBDA,
    GPT_MODEL, MAX_TOKENS_JUDGE, TOP_K
)

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context Freshness Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a56ff 0%, #0a2a8f 50%, #0f0f13 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(26, 86, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(26, 86, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a24 0%, #16161d 100%);
        border: 1px solid rgba(26, 86, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(26, 86, 255, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 86, 255, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a56ff;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(240, 240, 240, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .badge-fresh {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-stale {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(145deg, #1a1a24 0%, #16161d 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        border-color: rgba(26, 86, 255, 0.3);
    }
    .result-card-top {
        border-left: 3px solid #10b981;
    }
    .result-card-mid {
        border-left: 3px solid #f59e0b;
    }
    .result-card-low {
        border-left: 3px solid #ef4444;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #1a56ff, #3b7dff);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.75rem;
    }
    
    /* Score bar */
    .score-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .score-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    .score-bar-green { background: linear-gradient(90deg, #10b981, #34d399); }
    .score-bar-yellow { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .score-bar-red { background: linear-gradient(90deg, #ef4444, #f87171); }
    
    /* Sidebar styling */
    .sidebar-section {
        background: rgba(26, 86, 255, 0.05);
        border: 1px solid rgba(26, 86, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Info box */
    .info-box {
        background: rgba(26, 86, 255, 0.08);
        border: 1px solid rgba(26, 86, 255, 0.15);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: rgba(240, 240, 240, 0.7);
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chunk input areas */
    .chunk-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(240, 240, 240, 0.8);
        margin-bottom: 0.5rem;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }
    
    /* Architecture diagram */
    .arch-box {
        background: linear-gradient(145deg, #1a1a24 0%, #16161d 100%);
        border: 1px solid rgba(26, 86, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Inter', monospace;
        font-size: 0.85rem;
        color: rgba(240, 240, 240, 0.7);
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Load Demo Data
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    demo_path = Path(__file__).parent / "demo_data" / "examples.json"
    if demo_path.exists():
        with open(demo_path) as f:
            return json.load(f)
    return []

# ──────────────────────────────────────────────────────────────
# Simulated relevance (for demo mode without API key)
# ──────────────────────────────────────────────────────────────
def simulate_relevance(query: str, chunk_text: str) -> dict:
    """
    Simple keyword overlap scoring for demo mode.
    Not as good as GPT-4o, but works without an API key.
    """
    query_words = set(query.lower().split())
    chunk_words = set(chunk_text.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how',
                  'in', 'of', 'to', 'and', 'for', 'at', 'by', 'on', 'with',
                  'its', 'that', 'this', 'from', 'as', 'it', 'be', 'has', 'had'}
    query_words -= stop_words
    chunk_words -= stop_words
    
    if not query_words:
        return {"score": 0.5, "reason": "Unable to parse query keywords"}
    
    overlap = query_words & chunk_words
    score = min(1.0, len(overlap) / max(len(query_words), 1) * 1.2)
    
    if overlap:
        reason = f"Keyword match: {', '.join(list(overlap)[:3])}"
    else:
        reason = "No keyword overlap detected"
    
    return {"score": round(score, 4), "reason": reason}

# ──────────────────────────────────────────────────────────────
# Initialize Session State
# ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def load_demo_scenario():
    """Callback to populate widget keys when demo scenario changes."""
    selected = st.session_state.get("demo_selector", "— Select —")
    if selected == "— Select —":
        return
    demo_data = load_demo_data()
    demo = next((d for d in demo_data if d["name"] == selected), None)
    if not demo:
        return
    st.session_state.query_input = demo["query"]
    for i, chunk in enumerate(demo["chunks"]):
        st.session_state[f"chunk_text_{i}"] = chunk["text"]
        st.session_state[f"chunk_date_{i}"] = datetime.strptime(chunk["date"], "%Y-%m-%d").date()
    st.session_state.results = None
    st.session_state.evaluated = False

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required for GPT-4o relevance scoring. Leave empty for demo mode (keyword matching)."
    )
    
    if api_key:
        st.success("🔑 API key set — GPT-4o mode")
        use_gpt = True
    else:
        st.info("💡 Demo mode — using keyword matching for relevance")
        use_gpt = False
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scoring Parameters
    st.markdown("### 🎛️ Scoring Parameters")
    
    decay_lambda = st.slider(
        "Decay Lambda (λ)",
        min_value=0.01,
        max_value=0.10,
        value=DECAY_LAMBDA,
        step=0.005,
        help="Higher = faster staleness penalty. 0.03 = ~50% at 23 days."
    )
    
    rel_weight = st.slider(
        "Relevance Weight",
        min_value=0.1,
        max_value=0.9,
        value=RELEVANCE_WEIGHT,
        step=0.05,
        help="How much query relevance matters in final ranking."
    )
    fresh_weight = round(1.0 - rel_weight, 2)
    
    st.caption(f"Freshness Weight: **{fresh_weight}** (auto-balanced)")
    
    st.markdown("---")
    
    # Demo Data Loader
    st.markdown("### 📋 Demo Scenarios")
    demo_data = load_demo_data()
    
    if demo_data:
        demo_names = ["— Select —"] + [d["name"] for d in demo_data]
        st.selectbox(
            "Load a demo scenario",
            demo_names,
            key="demo_selector",
            on_change=load_demo_scenario
        )
    
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color: rgba(240,240,240,0.3); font-size: 0.75rem;">'
        'Built for Pacific · Context Freshness Evaluator v1.0'
        '</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────
# Main Content — Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚡ Context Freshness Evaluator</h1>
    <p>Score and rerank financial context chunks by relevance and freshness. 
    Identify stale data before it causes hallucinations.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Architecture Overview (collapsible)
# ──────────────────────────────────────────────────────────────
with st.expander("🏗️ How it works — Architecture", expanded=False):
    col_arch1, col_arch2, col_arch3 = st.columns(3)
    
    with col_arch1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📐</div>
            <div style="font-weight: 600; color: #f0f0f0; margin-bottom: 0.5rem;">Freshness Scorer</div>
            <div style="font-size: 0.8rem; color: rgba(240,240,240,0.5);">
                Pure math · Zero latency<br/>
                e<sup>-λ·days</sup> exponential decay
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-weight: 600; color: #f0f0f0; margin-bottom: 0.5rem;">Relevance Judge</div>
            <div style="font-size: 0.8rem; color: rgba(240,240,240,0.5);">
                GPT-4o · ~300ms/chunk<br/>
                Structured JSON scoring
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚖️</div>
            <div style="font-weight: 600; color: #f0f0f0; margin-bottom: 0.5rem;">Composite Reranker</div>
            <div style="font-size: 0.8rem; color: rgba(240,240,240,0.5);">
                Weighted blend · Tunable<br/>
                {rel_weight:.0%} relevance + {fresh_weight:.0%} freshness
            </div>
        </div>
        """.format(rel_weight=rel_weight, fresh_weight=fresh_weight), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Query Input
# ──────────────────────────────────────────────────────────────
st.markdown("### 🔍 Financial Query")
query = st.text_input(
    "Enter your query",
    placeholder="e.g., What were Apple's Q4 2024 earnings per share?",
    key="query_input",
    label_visibility="collapsed"
)

# ──────────────────────────────────────────────────────────────
# Chunk Inputs
# ──────────────────────────────────────────────────────────────
st.markdown("### 📄 Context Chunks")
st.caption("Enter up to 3 context chunks with their source dates. The evaluator will score and rank them.")

chunks_input = []
cols = st.columns(3)

for i in range(3):
    with cols[i]:
        st.markdown(f'<div class="chunk-header">Chunk {i + 1}</div>', unsafe_allow_html=True)
        
        chunk_text = st.text_area(
            f"Chunk {i + 1} text",
            height=150,
            placeholder=f"Paste context chunk {i + 1} here...",
            key=f"chunk_text_{i}",
            label_visibility="collapsed"
        )
        
        chunk_date = st.date_input(
            f"Chunk {i + 1} date",
            key=f"chunk_date_{i}",
            label_visibility="collapsed"
        )
        
        if chunk_text.strip():
            age = days_old(datetime.combine(chunk_date, datetime.min.time()))
            f_score = freshness_score(datetime.combine(chunk_date, datetime.min.time()), decay_lambda)
            
            if f_score >= 0.7:
                badge_class = "badge-fresh"
                badge_text = f"🟢 {age}d old · {f_score:.2f}"
            elif f_score >= 0.3:
                badge_class = "badge-moderate"
                badge_text = f"🟡 {age}d old · {f_score:.2f}"
            else:
                badge_class = "badge-stale"
                badge_text = f"🔴 {age}d old · {f_score:.2f}"
            
            st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
            
            chunks_input.append({
                "text": chunk_text.strip(),
                "date": chunk_date.isoformat(),
                "chunk_date_obj": datetime.combine(chunk_date, datetime.min.time()),
                "index": i
            })

# ──────────────────────────────────────────────────────────────
# Evaluate Button
# ──────────────────────────────────────────────────────────────
st.markdown("")  # spacer

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    evaluate_clicked = st.button(
        "⚡ Evaluate & Rerank",
        width="stretch",
        type="primary",
        disabled=not query.strip() or len(chunks_input) == 0
    )

# ──────────────────────────────────────────────────────────────
# Evaluation Pipeline
# ──────────────────────────────────────────────────────────────
if evaluate_clicked and query.strip() and chunks_input:
    with st.spinner("🔄 Scoring chunks..."):
        # Initialize OpenAI client if key provided
        client = None
        if use_gpt and api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        
        scored_chunks = []
        
        progress_bar = st.progress(0)
        
        for idx, chunk in enumerate(chunks_input):
            # Freshness score (instant)
            f_score = freshness_score(chunk["chunk_date_obj"], decay_lambda)
            age = days_old(chunk["chunk_date_obj"])
            
            # Relevance score (GPT-4o or simulated)
            if client:
                rel_result = score_relevance(
                    query, chunk["text"], client,
                    model=GPT_MODEL, max_tokens=MAX_TOKENS_JUDGE
                )
            else:
                rel_result = simulate_relevance(query, chunk["text"])
            
            scored_chunks.append({
                "text": chunk["text"],
                "date": chunk["date"],
                "days_old": age,
                "freshness_score": f_score,
                "relevance_score": rel_result["score"],
                "relevance_reason": rel_result["reason"],
                "original_index": chunk["index"] + 1
            })
            
            progress_bar.progress((idx + 1) / len(chunks_input))
        
        # Rerank
        ranked = rerank_chunks(scored_chunks, rel_weight, fresh_weight)
        stale = get_stale_chunks(ranked)
        
        st.session_state.results = ranked
        st.session_state.stale_count = len(stale)
        st.session_state.evaluated = True
        progress_bar.empty()

# ──────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────
if st.session_state.evaluated and st.session_state.results:
    ranked = st.session_state.results
    
    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")
    
    # Metric Cards
    avg_composite = sum(c["composite_score"] for c in ranked) / len(ranked)
    top_score = ranked[0]["composite_score"]
    stale_count = st.session_state.stale_count
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{top_score:.2f}</p>
            <p class="metric-label">Top Composite Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_composite:.2f}</p>
            <p class="metric-label">Avg Composite Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with mc3:
        color = "#ef4444" if stale_count > 0 else "#10b981"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: {color};">{stale_count}</p>
            <p class="metric-label">Stale Chunks (freshness &lt; 0.3)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with mc4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(ranked)}</p>
            <p class="metric-label">Chunks Evaluated</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")  # spacer
    
    # Stale warning
    if stale_count > 0:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>{stale_count} stale chunk{'s' if stale_count > 1 else ''} detected.</strong>
            Chunks with freshness below 0.3 may cause outdated or incorrect analysis.
            Consider removing or replacing with current data.
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Result Cards
    for chunk in ranked:
        composite = chunk["composite_score"]
        
        if composite >= 0.7:
            card_class = "result-card-top"
            score_bar_class = "score-bar-green"
            status_emoji = "🟢"
        elif composite >= 0.4:
            card_class = "result-card-mid"
            score_bar_class = "score-bar-yellow"
            status_emoji = "🟡"
        else:
            card_class = "result-card-low"
            score_bar_class = "score-bar-red"
            status_emoji = "🔴"
        
        preview = chunk["text"][:120] + ("..." if len(chunk["text"]) > 120 else "")
        
        st.markdown(f"""
        <div class="result-card {card_class} animate-in">
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <span class="rank-badge">#{chunk['rank']}</span>
                <span style="font-weight: 600; color: #f0f0f0; flex: 1;">
                    Chunk {chunk['original_index']} {status_emoji}
                </span>
                <span style="font-size: 1.4rem; font-weight: 700; color: #1a56ff;">
                    {composite:.3f}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: rgba(240,240,240,0.6); margin-bottom: 1rem;
                        line-height: 1.5; font-style: italic;">
                "{preview}"
            </div>
            <div style="display: flex; gap: 2rem; margin-bottom: 0.5rem;">
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; color: rgba(240,240,240,0.4); text-transform: uppercase;
                                letter-spacing: 0.05em; margin-bottom: 0.25rem;">Relevance</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f0f0f0;">
                        {chunk['relevance_score']:.3f}
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar {score_bar_class}" style="width: {chunk['relevance_score']*100:.0f}%;"></div>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; color: rgba(240,240,240,0.4); text-transform: uppercase;
                                letter-spacing: 0.05em; margin-bottom: 0.25rem;">Freshness</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f0f0f0;">
                        {chunk['freshness_score']:.3f}
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar {score_bar_class}" style="width: {chunk['freshness_score']*100:.0f}%;"></div>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; color: rgba(240,240,240,0.4); text-transform: uppercase;
                                letter-spacing: 0.05em; margin-bottom: 0.25rem;">Days Old</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f0f0f0;">
                        {chunk['days_old']}
                    </div>
                </div>
            </div>
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.06);">
                <span style="font-size: 0.75rem; color: rgba(240,240,240,0.4);">💬 Reason:</span>
                <span style="font-size: 0.8rem; color: rgba(240,240,240,0.6); margin-left: 0.5rem;">
                    {chunk['relevance_reason']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary Table
    st.markdown("### 📋 Summary Table")
    
    df = pd.DataFrame([{
        "Rank": c["rank"],
        "Chunk": f"Chunk {c['original_index']}",
        "Relevance": f"{c['relevance_score']:.3f}",
        "Freshness": f"{c['freshness_score']:.3f}",
        "Composite": f"{c['composite_score']:.3f}",
        "Days Old": c["days_old"],
        "Date": c["date"]
    } for c in ranked])
    
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Chunk": st.column_config.TextColumn("Chunk", width="small"),
            "Relevance": st.column_config.TextColumn("Relevance", width="small"),
            "Freshness": st.column_config.TextColumn("Freshness", width="small"),
            "Composite": st.column_config.TextColumn("Composite", width="small"),
            "Days Old": st.column_config.NumberColumn("Days Old", width="small"),
            "Date": st.column_config.TextColumn("Date", width="medium"),
        }
    )
    
    # Export JSON
    st.markdown("### 📥 Export")
    export_data = {
        "query": query,
        "config": {
            "decay_lambda": decay_lambda,
            "relevance_weight": rel_weight,
            "freshness_weight": fresh_weight,
            "mode": "gpt-4o" if use_gpt else "keyword-matching"
        },
        "results": [{
            "rank": c["rank"],
            "chunk_index": c["original_index"],
            "text": c["text"],
            "date": c["date"],
            "relevance_score": c["relevance_score"],
            "freshness_score": c["freshness_score"],
            "composite_score": c["composite_score"],
            "days_old": c["days_old"],
            "reason": c["relevance_reason"]
        } for c in ranked],
        "metadata": {
            "stale_chunks": st.session_state.stale_count,
            "evaluated_at": datetime.now().isoformat()
        }
    }
    
    st.download_button(
        label="📥 Download Results (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 0.8rem; color: rgba(240, 240, 240, 0.3);">
        Context Freshness Evaluator · Built for 
        <a href="https://pacific.app" target="_blank" style="color: #1a56ff; text-decoration: none;">Pacific</a>
        · Anuvik Thota
    </div>
</div>
""", unsafe_allow_html=True)
