"""
VibeFinder 2.0 — Streamlit UI

Run from the project root:
    streamlit run src/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.recommender import load_songs
from src.retriever import build_index
from src.agent import run_agent, AgentResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VibeFinder 2.0",
    page_icon="🎵",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .vibe-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .vibe-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    .song-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.2s;
    }
    .song-card:hover { border-color: #a78bfa; }

    .song-rank   { font-size: 0.75rem; color: #6b7280; font-weight: 600; letter-spacing: 0.05em; }
    .song-title  { font-size: 1.15rem; font-weight: 700; color: #f3f4f6; margin: 0.1rem 0; }
    .song-artist { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; }

    .tag {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .tag-genre  { background: #312e81; color: #a5b4fc; }
    .tag-mood   { background: #1e3a5f; color: #7dd3fc; }
    .tag-energy { background: #14532d; color: #6ee7b7; }

    .song-why {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 0.6rem;
        font-style: italic;
    }

    .genre-lock-banner {
        background: #451a03;
        border: 1px solid #92400e;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #fcd34d;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .step-row {
        font-size: 0.82rem;
        color: #d1d5db;
        padding: 0.25rem 0;
        border-bottom: 1px solid #1f2937;
    }
    .step-name { color: #a78bfa; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load catalog + build index once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading song catalog...")
def load_catalog():
    songs = load_songs("data/songs.csv")
    build_index(songs)
    return songs

songs = load_catalog()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="vibe-title">🎵 VibeFinder 2.0</p>', unsafe_allow_html=True)
st.markdown('<p class="vibe-subtitle">AI-Powered Music Discovery — describe what you want in plain English</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
with st.form("query_form"):
    query = st.text_input(
        label="What are you in the mood for?",
        placeholder='e.g. "something chill for studying late at night"',
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Find Music", use_container_width=True, type="primary")

# Example chips
st.markdown("**Try:**")
examples = [
    "chill lofi for studying",
    "hype me up at the gym",
    "sad rainy day vibes",
    "happy party music",
    "dark and angry metal",
    "peaceful acoustic for sleep",
]
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
        query = ex
        submitted = True

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
if submitted and query:
    with st.spinner("Finding your music..."):
        try:
            result: AgentResult = run_agent(query, songs, k=5)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # ---- Profile pill row ----
    p = result.profile
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genre",    p["favorite_genre"])
    c2.metric("Mood",     p["favorite_mood"])
    c3.metric("Energy",   f"{p['target_energy']:.0%}")
    c4.metric("Acoustic", "Yes" if p["likes_acoustic"] else "No")

    # ---- Genre-lock warning ----
    if result.genre_lock_detected and result.genre_lock_corrected:
        st.markdown(
            '<div class="genre-lock-banner">⚠️ <strong>Genre-lock detected & corrected</strong> — '
            'the top song was winning mainly on genre bonus despite a large energy mismatch. '
            'Re-ranked to prioritise energy and mood fit.</div>',
            unsafe_allow_html=True,
        )

    # ---- Recommendations ----
    st.subheader("Recommendations")
    for i, rec in enumerate(result.recommendations, 1):
        s = rec.song
        energy_pct = int(float(s["energy"]) * 100)
        conf_pct   = int(rec.confidence * 100)
        why_text   = rec.explanation.split("—", 1)[-1].strip() if "—" in rec.explanation else rec.explanation

        st.markdown(f"""
        <div class="song-card">
            <div class="song-rank">#{i}</div>
            <div class="song-title">{s['title']}</div>
            <div class="song-artist">{s['artist']}</div>
            <span class="tag tag-genre">{s['genre']}</span>
            <span class="tag tag-mood">{s['mood']}</span>
            <span class="tag tag-energy">energy {energy_pct}%</span>
            <div class="song-why">{why_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.progress(rec.confidence, text=f"Confidence: {conf_pct}%")

    # ---- Agent reasoning (collapsible) ----
    with st.expander("Agent reasoning steps", expanded=False):
        for step in result.steps:
            st.markdown(
                f'<div class="step-row">'
                f'<span class="step-name">[{step.name}]</span> {step.detail}'
                f'</div>',
                unsafe_allow_html=True,
            )

elif submitted and not query:
    st.warning("Please type something first.")
