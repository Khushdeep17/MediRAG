import streamlit as st
import time
import re
import sys
import os

# ── Fix import path ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generate import generate_answer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRAG",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b1220;
    --bg-gradient: radial-gradient(circle at 20% 20%, #0f1b32 0%, #0b1220 60%);
    --surface: #111c2e;
    --surface-light: #16263f;
    --border: #1f2f4a;
    --accent: #00d4ff;
    --accent-soft: #0099cc;
    --text: #e6f1ff;
    --muted: #a8bfd8;
    --success: #00e676;
    --danger: #ff4d6d;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', sans-serif;
    --navy: #3b6fd4;
    --navy-heading: #00b4e6;   /* medical navy cyan */
}

.navy-heading { color: var(--navy) !important; }

/* ── Global ── */
html, body, [class*="css"] {
    background: var(--bg-gradient) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

/* Force normal body text light, NOT headings */
p, span, li, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"] p {
    color: var(--text) !important;
}
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b32 0%, #0b1220 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Text Input ── */
[data-testid="stTextInput"],
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div {
    background: var(--surface-light) !important;
}
[data-testid="stTextInput"] input {
    background: var(--surface-light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    padding: 0.8rem 1rem !important;
    font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.2) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4a6080 !important; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s ease;
}

/* Primary button (type="primary") */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #00d4ff, #0099cc) !important;
    color: #001825 !important;
    border: none !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0,212,255,0.4) !important;
}
[data-testid="stButton"] button[kind="primary"]:disabled {
    background: #0d3a4d !important;
    color: #2a6a80 !important;
}

/* Ghost buttons */
[data-testid="stButton"] button[kind="secondary"] {
    background: var(--surface-light) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* ── Answer block ── */
.answer-block {
    background: linear-gradient(180deg, #0f1b32 0%, #111c2e 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.6rem;
    font-size: 15px;
    line-height: 1.75;
    color: var(--text) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
}
.answer-block * { color: var(--text) !important; }

/* ── Info card ── */
.info-card {
    background: var(--surface-light);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.info-card-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #a8bfd8 !important;
    margin-bottom: 8px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
}
.info-row:last-child { border-bottom: none; }
.info-key   { color: #a8bfd8 !important; }
.info-value { color: var(--text) !important; font-family: var(--mono); }

/* ── Section label ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--navy-heading) !important;
    margin: 1.8rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Chips ── */
.metric-row { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    padding: 6px 12px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 11px;
    border: 1px solid;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.chip-green { background: rgba(0,230,118,0.08); color: #00e676 !important; border-color: rgba(0,230,118,0.25); }
.chip-red   { background: rgba(255,77,109,0.08); color: #ff4d6d !important; border-color: rgba(255,77,109,0.25); }
.chip-blue  { background: rgba(0,212,255,0.08);  color: #00d4ff !important; border-color: rgba(0,212,255,0.25); }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface-light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: var(--text) !important; }
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] * {
    color: #a8bfd8 !important;
}

/* ── Warning / misc ── */
hr { border-color: var(--border) !important; }
/* Spinner text */
[data-testid="stSpinner"] {
    color: #00b4e6 !important;
    font-weight: 600;
}
[data-testid="stSpinner"] * {
    color: #00b4e6 !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "generating" not in st.session_state:
    st.session_state.generating = False


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 1.2rem">
        <div style="font-size:19px;font-weight:600;color:#e6f1ff">⚕ MediRAG</div>
        <div style="font-size:11px;color:#a8bfd8;margin-top:3px">Medical QA System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#a8bfd8;margin-bottom:8px">System Stack</div>', unsafe_allow_html=True)
    for label, value, color in [
        ("Corpus",    "Merck Manual",      "#00d4ff"),
        ("Chunks",    "4,239 passages",    "#00d4ff"),
        ("Dense",     "BGE-large (FAISS)", "#00e676"),
        ("Sparse",    "BM25",              "#00e676"),
        ("Fusion",    "RRF α=0.7",         "#00e676"),
        ("Generator", "Qwen3-32B (Groq)",  "#f59e0b"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1f2f4a">
            <span style="font-size:12px;color:#a8bfd8">{label}</span>
            <span style="font-size:12px;color:{color};font-family:'JetBrains Mono',monospace">{value}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#a8bfd8;margin-bottom:8px">Evaluation (offline)</div>', unsafe_allow_html=True)
    for label, value in [
        ("Recall@10",   "1.000"),
        ("MRR",         "0.883"),
        ("NDCG@10",     "0.918"),
        ("Grounded",    "~91%"),
        ("Faith (LLM)", "4.1 / 5"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1f2f4a">
            <span style="font-size:12px;color:#a8bfd8">{label}</span>
            <span style="font-size:12px;color:#e6f1ff;font-family:'JetBrains Mono',monospace">{value}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.4rem;font-size:11px;color:#4a6080">
        Built by <span style="color:#a8bfd8">Khushdeep Singh</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN — HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:24px;
            font-weight:700;
            color:#00b4e6;
            letter-spacing:-0.02em;
            margin-bottom:6px;">
    Medical Retrieval-Augmented QA
</div>

<div style="font-size:14px;
            font-weight:500;
            color:#4fa3c7;
            margin-bottom:1.6rem;
            line-height:1.6;">
    Hybrid Dense + BM25 retrieval over the Merck Manual
    &nbsp;·&nbsp; Citations map to retrieved source chunks
    &nbsp;·&nbsp; Powered by Qwen3-32B on Groq
</div>
""", unsafe_allow_html=True)

# ── Helper for example buttons ─────────────────────────────────────────────────
def set_example(text):
    st.session_state.query_input = text

# ── Input ──────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Medical Question",
    placeholder="e.g. What causes iron deficiency anemia?",
    label_visibility="collapsed",
    key="query_input",
)

# ── Button row ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    "How is asthma treated?",
    "What are the symptoms of Parkinson disease?",
    "How does portal hypertension develop?",
]

col_btn, col_ex1, col_ex2, col_ex3, _ = st.columns([1.6, 2.0, 2.4, 2.6, 3])

with col_btn:
    generate_btn = st.button(
        "⚕ Generate",
        type="primary",                          # ← cyan gradient button
        use_container_width=True,
        disabled=st.session_state.generating,
    )

with col_ex1:
    st.button("↗ Asthma",     key="ex1", use_container_width=True, on_click=set_example, args=(EXAMPLES[0],))
with col_ex2:
    st.button("↗ Parkinson",  key="ex2", use_container_width=True, on_click=set_example, args=(EXAMPLES[1],))
with col_ex3:
    st.button("↗ Portal HTN", key="ex3", use_container_width=True, on_click=set_example, args=(EXAMPLES[2],))

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# GENERATION + DISPLAY
# ══════════════════════════════════════════════════════════════════
if generate_btn and query.strip():

    st.session_state.generating = True

    with st.spinner("Retrieving context and generating answer…"):
        t0 = time.time()
        answer, retrieved_chunks = generate_answer(query, verbose=False)
        elapsed = time.time() - t0

    st.session_state.generating = False

    # Strip empty template sections
    answer_clean = answer

# Remove ### style markdown headers
    answer_clean = re.sub(r'#{1,6}\s*', '', answer_clean)

# Remove empty acute section
    answer_clean = re.sub(
    r'(?:Acute Management|Acute Treatment)\s*\n+Not covered[^\n]*\n*',
    '',
    answer_clean,
    flags=re.IGNORECASE,
)

    # ── Generated Answer ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Generated Answer</div>', unsafe_allow_html=True)

    col_ans, col_meta = st.columns([3, 1])

    with col_ans:
        st.markdown(f'<div class="answer-block">{answer_clean}</div>', unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">Response Info</div>
            <div class="info-row">
                <span class="info-key">Latency</span>
                <span class="info-value">{elapsed:.2f}s</span>
            </div>
            <div class="info-row">
                <span class="info-key">Sources</span>
                <span class="info-value">{len(retrieved_chunks)}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Words</span>
                <span class="info-value">~{len(answer_clean.split())}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Retrieved Sources ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Retrieved Sources</div>', unsafe_allow_html=True)

    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        chap_num   = chunk.get("chapter_number", "?")
        chap_title = chunk.get("chapter_title", "Unknown")
        content    = chunk.get("content", "")[:800]
        with st.expander(f"[{i}]  Ch. {chap_num} — {chap_title}"):
            st.markdown(f'<div style="font-size:13px;line-height:1.7;color:#a8bfd8">{content}</div>',
                        unsafe_allow_html=True)

    # ── Grounding Signals ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Grounding Signals</div>', unsafe_allow_html=True)

    retrieved_chapter_numbers = [c.get("chapter_number") for c in retrieved_chunks[:5]]
    cited_numbers = sorted(set(int(n) for n in re.findall(r'\[(\d+)\]', answer)))
    cited_actual  = [
        retrieved_chunks[i - 1].get("chapter_number")
        for i in cited_numbers if 0 < i <= len(retrieved_chunks)
    ]

    primary_chapter = retrieved_chapter_numbers[0] if retrieved_chapter_numbers else "—"
    primary_title   = retrieved_chunks[0].get("chapter_title", "") if retrieved_chunks else ""
    unique_chapters = len(set(retrieved_chapter_numbers))

    citation_consistent = bool(cited_actual) and set(cited_actual).issubset(set(retrieved_chapter_numbers))
    has_citations       = len(cited_numbers) > 0
    multi_chapter       = unique_chapters > 1

    st.markdown(f"""
    <div class="info-card" style="margin-bottom:0.8rem">
        <div class="info-card-label">Primary Source</div>
        <div class="info-row">
            <span class="info-key">Chapter</span>
            <span class="info-value">{primary_chapter}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Title</span>
            <span class="info-value" style="color:#00d4ff !important;text-align:right;max-width:72%">{primary_title}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Unique chapters</span>
            <span class="info-value">{unique_chapters}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Citations</span>
            <span class="info-value">{cited_numbers if cited_numbers else "none"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def chip(label, ok, true_text, false_text):
        cls  = "chip-green" if ok else "chip-red"
        icon = "✓" if ok else "✗"
        return f'<div class="chip {cls}">{icon}&nbsp;{label} — {true_text if ok else false_text}</div>'

    st.markdown(f"""
    <div class="metric-row">
        {chip("Citations Found",     has_citations,       f"{len(cited_numbers)} in-text", "none detected")}
        {chip("Citation Consistent", citation_consistent, "cited ⊆ retrieved",             "mismatch")}
        {chip("Multi-Chapter",       multi_chapter,       f"{unique_chapters} chapters",    "single chapter")}
        <div class="chip chip-blue">⏱&nbsp;{elapsed:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.9rem;font-size:11px;color:#4a6080;font-style:italic">
        Grounding signals are structural checks. Full faithfulness scores are in offline LLM-as-Judge evaluation.
    </div>
    """, unsafe_allow_html=True)

elif generate_btn and not query.strip():
    st.warning("Please enter a question first.")

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
        <div style="font-size:36px;opacity:0.2;margin-bottom:0.8rem">⚕</div>
        <div style="font-size:13px;font-family:'JetBrains Mono',monospace;color:#2a4060">
            Enter a question and click Generate
        </div>
    </div>
    """, unsafe_allow_html=True)