"""
IntelliCredit — AI Credit Appraisal Platform
Streamlit frontend integrating all 5 IntelliCredit modules.

Security hardening:
  - Uploaded files are written only to a per-session temp dir inside ./storage/uploads/
  - File extensions are allowlisted (.pdf, .csv, .json)
  - No shell commands are executed with user input
  - CAM markdown is rendered with st.markdown (no html=True) to prevent XSS
  - User-supplied company name is stripped and length-capped before use
  - External-signal JSON is parsed safely with json.loads and schema-validated
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# ── Path setup so imports from ai-brain work without installing the package ──
_ROOT = Path(__file__).parent
_AI_BRAIN = _ROOT / "ai-brain"
if str(_AI_BRAIN) not in sys.path:
    sys.path.insert(0, str(_AI_BRAIN))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Credix — AI Credit Appraisal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Security constants ────────────────────────────────────────────────────────
_ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json"}
_MAX_COMPANY_NAME_LEN = 200
_UPLOAD_DIR = _ROOT / "storage" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Custom CSS — premium dark glassmorphism theme (v3) ───────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    *, *::before, *::after { box-sizing: border-box; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.07) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 10%, rgba(168,85,247,0.06) 0%, transparent 55%),
                    linear-gradient(135deg, #06090f 0%, #0b1025 50%, #080e1c 100%);
        min-height: 100vh;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1e 0%, #0d1535 100%);
        border-right: 1px solid rgba(99,102,241,0.16);
    }
    section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

    /* ── Header ── */
    .intellicredit-header {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.12) 0%,
            rgba(168,85,247,0.08) 50%,
            rgba(56,189,248,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.22);
        border-radius: 20px; padding: 30px 38px; margin-bottom: 28px;
        backdrop-filter: blur(12px); position: relative; overflow: hidden;
    }
    .intellicredit-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8);
    }
    .intellicredit-header h1 {
        font-size: 2.5rem; font-weight: 900;
        background: linear-gradient(90deg, #818cf8, #a78bfa, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0; letter-spacing: -0.03em;
    }
    .intellicredit-header p { color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.6; }
    .ic-badge-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
    .ic-badge {
        display: inline-flex; align-items: center;
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.18);
        color: #818cf8; font-size: 0.72rem; font-weight: 600;
        padding: 3px 11px; border-radius: 999px; letter-spacing: 0.04em;
    }

    /* ── Module cards ── */
    .module-card {
        background: rgba(10,16,40,0.75);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 16px; padding: 20px 26px; margin-bottom: 20px;
        backdrop-filter: blur(10px); transition: border-color 0.25s;
    }
    .module-card:hover { border-color: rgba(99,102,241,0.42); }
    .module-card h3 { font-size: 1rem; font-weight: 700; color: #818cf8; margin: 0 0 4px 0; }
    .module-card p  { color: #475569; font-size: 0.83rem; margin: 0; line-height: 1.5; }

    /* ── Gauge cards ── */
    .gauge-card {
        background: rgba(10,16,40,0.82);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 18px; padding: 26px 18px;
        text-align: center; position: relative; overflow: hidden;
        transition: border-color 0.3s, transform 0.2s;
    }
    .gauge-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        background: var(--g-color, #6366f1); opacity: 0.55;
    }
    .gauge-card:hover { border-color: rgba(99,102,241,0.42); transform: translateY(-2px); }
    .gauge-label {
        color: #475569; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
    }
    .gauge-sub { color: #1e293b; font-size: 0.7rem; margin-top: 3px; }
    .gauge-badge {
        display: inline-block; font-size: 0.76rem; font-weight: 700;
        padding: 4px 16px; border-radius: 999px; margin-top: 10px;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .donut-wrap { display: flex; justify-content: center; margin: 4px 0; }

    /* ── Badge colours ── */
    .b-low  { background:rgba(34,197,94,0.12); color:#4ade80; border:1px solid rgba(34,197,94,0.25); }
    .b-med  { background:rgba(234,179,8,0.12); color:#fbbf24; border:1px solid rgba(234,179,8,0.25); }
    .b-high { background:rgba(239,68,68,0.12); color:#f87171; border:1px solid rgba(239,68,68,0.25); }
    .b-crit { background:rgba(239,68,68,0.2);  color:#ef4444; border:1px solid rgba(239,68,68,0.4); }
    .b-unk  { background:rgba(100,116,139,0.1);color:#94a3b8; border:1px solid rgba(100,116,139,0.18); }

    /* ── Step badge (sidebar) ── */
    .step-badge {
        display: inline-block;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: #fff; font-size: 0.68rem; font-weight: 700;
        padding: 2px 10px; border-radius: 999px; margin-right: 8px; letter-spacing: 0.04em;
    }

    /* ── Metric tiles ── */
    .metric-tile {
        background: rgba(10,16,40,0.7); border: 1px solid rgba(99,102,241,0.18);
        border-radius: 13px; padding: 16px 18px; text-align: center;
    }
    .metric-tile .mt-label { color: #475569; font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px; }
    .metric-tile .mt-value { color: #e2e8f0; font-size: 1.5rem; font-weight: 800; line-height: 1; }

    /* ── Score bars ── */
    .sbar-wrap { margin-bottom: 14px; }
    .sbar-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
    .sbar-label  { color: #94a3b8; font-size: 0.83rem; }
    .sbar-value  { font-weight: 700; font-size: 0.83rem; }
    .sbar-track  { background: rgba(20,30,50,0.95); border-radius: 999px; height: 8px; overflow: hidden; }
    .sbar-fill   { height: 100%; border-radius: 999px; transition: width 0.7s ease; }

    /* ── Chip styles ── */
    .flag-chip {
        display: inline-block; background: rgba(239,68,68,0.1); color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.22); border-radius: 8px;
        padding: 5px 12px; font-size: 0.8rem; margin: 3px 3px 0 0; cursor: default;
    }
    .ok-chip {
        display: inline-block; background: rgba(34,197,94,0.1); color: #86efac;
        border: 1px solid rgba(34,197,94,0.22); border-radius: 8px;
        padding: 5px 12px; font-size: 0.8rem; margin: 3px 3px 0 0;
    }
    .info-chip {
        display: inline-block; background: rgba(56,189,248,0.08); color: #7dd3fc;
        border: 1px solid rgba(56,189,248,0.18); border-radius: 8px;
        padding: 5px 12px; font-size: 0.8rem; margin: 3px 3px 0 0;
    }

    /* ── Info / warn panels ── */
    .info-panel {
        background: rgba(56,189,248,0.05); border: 1px solid rgba(56,189,248,0.17);
        border-radius: 12px; padding: 13px 17px; color: #7dd3fc;
        font-size: 0.83rem; line-height: 1.6; margin-bottom: 14px;
    }
    .warn-panel {
        background: rgba(234,179,8,0.05); border: 1px solid rgba(234,179,8,0.18);
        border-radius: 12px; padding: 13px 17px; color: #fbbf24;
        font-size: 0.83rem; line-height: 1.6; margin-bottom: 14px;
    }

    /* ── Row styles ── */
    .risk-row {
        background: rgba(239,68,68,0.06); border-left: 3px solid #f87171;
        border-radius: 0 10px 10px 0; padding: 9px 14px; color: #fca5a5;
        font-size: 0.83rem; margin-bottom: 7px; line-height: 1.5;
    }
    .insight-row {
        background: rgba(56,189,248,0.05); border-left: 3px solid #38bdf8;
        border-radius: 0 10px 10px 0; padding: 9px 14px; color: #7dd3fc;
        font-size: 0.83rem; margin-bottom: 7px; line-height: 1.5;
    }
    .signal-card {
        background: rgba(10,16,40,0.65); border-radius: 12px; padding: 13px 16px;
        margin-bottom: 10px; border-left: 4px solid var(--sc, #6366f1);
    }
    .rec-row {
        display: flex; align-items: flex-start; gap: 12px;
        background: rgba(10,16,40,0.65); border: 1px solid rgba(99,102,241,0.1);
        border-radius: 10px; padding: 11px 15px; margin-bottom: 8px;
    }
    .rec-field {
        min-width: 130px; font-weight: 700; font-size: 0.8rem;
        color: #818cf8; text-transform: capitalize; flex-shrink: 0;
    }
    .rec-status {
        font-size: 0.72rem; font-weight: 700; padding: 3px 10px;
        border-radius: 999px; flex-shrink: 0;
    }
    .rec-reason { color: #475569; font-size: 0.8rem; line-height: 1.4; }

    /* ── Dividers ── */
    .section-divider { border: none; border-top: 1px solid rgba(99,102,241,0.12); margin: 26px 0; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #fff; border: none; border-radius: 12px;
        padding: 11px 30px; font-weight: 700; font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
        box-shadow: 0 4px 20px rgba(99,102,241,0.22);
    }
    .stButton > button:hover {
        opacity: 0.9; transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99,102,241,0.38);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10,16,40,0.6); border-radius: 12px; padding: 4px; gap: 3px;
        border: 1px solid rgba(99,102,241,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px; color: #334155; font-weight: 500;
        font-size: 0.84rem; padding: 8px 15px; transition: color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #fff !important; font-weight: 700 !important;
    }

    /* ── File uploader ── */
    .stFileUploader > div {
        background: rgba(10,16,40,0.6) !important;
        border: 2px dashed rgba(99,102,241,0.3) !important; border-radius: 14px !important;
        transition: border-color 0.2s;
    }
    .stFileUploader > div:hover { border-color: rgba(99,102,241,0.6) !important; }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(10,16,40,0.8) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        color: #e2e8f0 !important; border-radius: 9px !important; font-size: 0.9rem !important;
    }

    /* ── Sidebar logo ── */
    .sb-logo {
        padding: 22px 16px 10px; text-align: center;
        border-bottom: 1px solid rgba(99,102,241,0.1); margin-bottom: 4px;
    }
    .sb-logo-name {
        font-size: 1.15rem; font-weight: 900;
        background: linear-gradient(90deg, #818cf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sb-logo-tag { color: #1e293b; font-size: 0.67rem; margin-top: 1px; }

    /* ── Finance value cards ── */
    .fin-card {
        background: rgba(10,16,40,0.7); border: 1px solid rgba(99,102,241,0.18);
        border-radius: 14px; padding: 18px 18px; text-align: center; margin-bottom: 12px;
        transition: border-color 0.25s, transform 0.2s;
    }
    .fin-card:hover { border-color: rgba(99,102,241,0.4); transform: translateY(-1px); }
    .fin-label { color: #475569; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 8px; }
    .fin-value { font-size: 1.35rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
    .fin-conf  { font-size: 0.68rem; font-weight: 700; padding: 2px 8px;
                 border-radius: 999px; display: inline-block; }

    /* ── CAM wrapper ── */
    .cam-wrapper {
        background: rgba(10,16,40,0.6); border: 1px solid rgba(99,102,241,0.17);
        border-radius: 16px; padding: 28px 32px;
        font-size: 0.9rem; line-height: 1.7; color: #cbd5e1;
    }

    /* ── Footer ── */
    .ic-footer {
        border-top: 1px solid rgba(99,102,241,0.07); padding: 14px 0; margin-top: 36px;
        text-align: center; color: #1e293b; font-size: 0.7rem; letter-spacing: 0.05em;
    }

    /* ── Misc overrides ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stExpander { border: 1px solid rgba(99,102,241,0.12) !important; border-radius: 12px !important; }
    div[data-testid="stDecoration"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_company_name(raw: str) -> str:
    return raw.strip()[:_MAX_COMPANY_NAME_LEN]


def _save_uploaded_files(uploaded_files) -> List[Path]:
    session_dir = _UPLOAD_DIR / st.session_state.get("session_id", "default")
    session_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            st.warning(f"⚠️ Skipped **{uf.name}**: unsupported extension `{suffix}`.")
            continue
        safe_name = f"{uuid.uuid4().hex}_{Path(uf.name).name}"
        dest = session_dir / safe_name
        dest.write_bytes(uf.read())
        saved.append(dest)
    return saved


def _parse_external_signals_json(raw: str):
    if not raw.strip():
        return [], None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(parsed, list):
        return None, "External signals must be a JSON array ([...])."
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            return None, f"Item {i} is not a JSON object."
        required = {"type", "severity", "confidence"}
        missing = required - item.keys()
        if missing:
            return None, f"Item {i} is missing required fields: {', '.join(sorted(missing))}."
        if not isinstance(item.get("confidence"), (int, float)):
            return None, f"Item {i}: 'confidence' must be a number between 0 and 1."
        if not (0.0 <= float(item["confidence"]) <= 1.0):
            return None, f"Item {i}: 'confidence' must be between 0.0 and 1.0."
    return parsed, None


def _risk_color(score: float) -> str:
    if score >= 75:
        return "#f87171"
    if score >= 45:
        return "#fbbf24"
    return "#4ade80"


def _badge_cls(category: str) -> str:
    """Map a risk category string to a CSS badge class."""
    cat = (category or "").lower()
    if "critical" in cat:
        return "b-crit"
    if "high" in cat:
        return "b-high"
    if "medium" in cat or "moderate" in cat:
        return "b-med"
    if "low" in cat:
        return "b-low"
    return "b-unk"


def _risk_badge_class(category: str) -> str:
    """Legacy alias kept for any remaining callers."""
    return _badge_cls(category)


def _fmt_currency(val: Any) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
        if abs(v) >= 1_000_000:
            return f"₹ {v/1_000_000:,.2f}M"
        if abs(v) >= 1_000:
            return f"₹ {v:,.0f}"
        return f"₹ {v:,.2f}"
    except (TypeError, ValueError):
        return str(val)


def _conf_label(conf_info) -> str:
    """Safely extract confidence label from either a str or a dict."""
    if isinstance(conf_info, str) and conf_info:
        return conf_info.capitalize()
    if isinstance(conf_info, dict):
        level = conf_info.get("level", "")
        return level.capitalize() if level else "—"
    return "—"


def _donut_svg(score: float, color: str, size: int = 112) -> str:
    """Return an SVG donut gauge for the given score (0–100)."""
    radius = 38
    cx = cy = size // 2
    circumference = 2 * 3.14159 * radius
    pct = min(max(score / 100.0, 0), 1)
    dash = pct * circumference
    gap = circumference - dash
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" '
        f'stroke="rgba(20,30,50,0.95)" stroke-width="9"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" '
        f'stroke="{color}" stroke-width="9" '
        f'stroke-dasharray="{dash:.1f} {gap:.1f}" '
        f'stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy+6}" text-anchor="middle" '
        f'font-family="Inter,sans-serif" font-weight="900" font-size="17" '
        f'fill="{color}">{score:.0f}</text>'
        f'</svg>'
    )


def _score_bar(label: str, score: float, max_score: float = 100.0):
    pct = min(max(score / max_score, 0), 1) * 100
    color = _risk_color(score)
    st.markdown(
        f'<div class="sbar-wrap">'
        f'<div class="sbar-header"><span class="sbar-label">{label}</span>'
        f'<span class="sbar-value" style="color:{color};">{score:.1f}</span></div>'
        f'<div class="sbar-track">'
        f'<div class="sbar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:16px 0 8px;">
              <div style="font-size:2.5rem;">🏦</div>
              <div style="font-size:1.15rem;font-weight:800;background:linear-gradient(90deg,#818cf8,#a78bfa);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                IntelliCredit
              </div>
              <div style="color:#475569;font-size:0.75rem;">AI Credit Appraisal Platform</div>
            </div>
            <hr style="border-color:rgba(99,102,241,0.15);margin:12px 0 20px;">
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🏢 Company Details")
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g. ABC Manufacturing Ltd.",
            help="Legal name of the borrower entity.",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Module Settings")
        include_cam = st.toggle("Generate CAM Report", value=True)
        perform_web_research = st.toggle("AI Web Research Agent", value=False)

        st.markdown("---")
        st.markdown("### 📑 Pipeline Steps")
        for badge, label in [
            ("1", "Multi-Source Ingestion"),
            ("2", "Cross-Checking Engine"),
            ("3", "AI Research Agent"),
            ("4", "Primary Inspection"),
            ("5", "Explainable AI + CAM"),
        ]:
            st.markdown(
                f'<div style="margin:5px 0;"><span class="step-badge">{badge}</span>'
                f'<span style="color:#94a3b8;font-size:0.82rem;">{label}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="color:#334155;font-size:0.72rem;text-align:center;">'
            "Draft outputs only · Not a lending decision</div>",
            unsafe_allow_html=True,
        )

    return {
        "company_name": _sanitize_company_name(company_name) if company_name else "",
        "include_cam": include_cam,
        "perform_web_research": perform_web_research,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1 — Ingestion
# ═══════════════════════════════════════════════════════════════════════════════

def render_ingestion_tab() -> List[Path]:
    st.markdown(
        """
        <div class="module-card">
          <h3>📂 Module 1 — Multi-Source Data Ingestion Engine</h3>
          <p>Upload company financial documents. The AI runs OCR, extracts financial fields via
             Vectorless RAG, and creates vector embeddings for narrative text retrieval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-tile"><div class="mt-label">Vectorless RAG</div>'
                    '<div class="mt-value" style="color:#818cf8;font-size:1rem;">Financial Numbers</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-tile"><div class="mt-label">Vector RAG</div>'
                    '<div class="mt-value" style="color:#a78bfa;font-size:1rem;">Narrative Text</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-tile"><div class="mt-label">Supported Formats</div>'
                    '<div class="mt-value" style="color:#38bdf8;font-size:1rem;">PDF · CSV · JSON</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload financial documents (PDF, CSV, JSON)",
        type=["pdf", "csv", "json"],
        accept_multiple_files=True,
        help="Bank statements, GST filings, annual reports, P&L statements.",
        key="doc_uploader",
    )

    if uploaded:
        st.markdown(
            f'<div style="color:#4ade80;font-size:0.9rem;margin-bottom:8px;">✅ {len(uploaded)} file(s) selected</div>',
            unsafe_allow_html=True,
        )
        for uf in uploaded:
            st.markdown(
                f'<div style="color:#94a3b8;font-size:0.82rem;padding:2px 0;">📄 {uf.name} '
                f'<span style="color:#475569;">({uf.size:,} bytes)</span></div>',
                unsafe_allow_html=True,
            )
        return _save_uploaded_files(uploaded)

    st.info("ℹ️ Upload one or more documents: Bank Statements, GST Filings, Annual Reports.")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Module 3 — Research Agent
# ═══════════════════════════════════════════════════════════════════════════════

def render_research_tab() -> List[Dict]:
    st.markdown(
        """
        <div class="module-card">
          <h3>🌐 Module 3 — AI Research Agent</h3>
          <p>The AI Research Agent scans publicly available information about the company, promoters,
             and industry. Supply pre-researched external signals or enable live web research.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    signal_source = st.radio(
        "External Signal Source",
        ["None", "Manual JSON Input", "Live Web Research (AI Research Agent)"],
        horizontal=True,
    )

    if signal_source == "None":
        st.info("ℹ️ No external signals. The external risk score will be a neutral placeholder.")
        return []

    if signal_source == "Live Web Research (AI Research Agent)":
        st.success("🔍 Live web research enabled via Google News RSS. The AI Research Agent will run during analysis.")
        st.markdown(
            """
            <div class="module-card" style="border-color:rgba(56,189,248,0.25);">
              <p style="color:#7dd3fc;font-size:0.85rem;">
                <strong>Research topics:</strong> Legal proceedings · Regulatory actions ·
                Promoter public records · Business news · Industry &amp; sector risks
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["perform_web_research_override"] = True
        return []

    # Manual JSON
    st.session_state.pop("perform_web_research_override", None)
    st.markdown(
        '<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:8px;">'
        'Paste a JSON array of signals. Required per item: <code>type</code>, '
        '<code>severity</code> (low/medium/high/critical), <code>confidence</code> (0–1).</div>',
        unsafe_allow_html=True,
    )

    example = json.dumps([
        {"type": "legal", "severity": "high", "confidence": 0.85, "source": "json",
         "evidence": "Promoter involved in two ongoing civil disputes filed in 2023."},
        {"type": "sector", "severity": "medium", "confidence": 0.70, "source": "json",
         "evidence": "Industry facing regulatory cost increases effective Q2 2025."}
    ], indent=2)

    raw_json = st.text_area(
        "External Signals JSON", value="", height=180, placeholder=example,
    )

    if raw_json.strip():
        signals, err = _parse_external_signals_json(raw_json)
        if err:
            st.error(f"❌ {err}")
            return []
        signals = signals or []
        st.success(f"✅ {len(signals)} valid external signal(s) parsed.")
        return signals

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Module 4 — Inspection
# ═══════════════════════════════════════════════════════════════════════════════

def render_inspection_tab():
    st.markdown(
        """
        <div class="module-card">
          <h3>🔍 Module 4 — Primary Insight Integration</h3>
          <p>Credit officers enter field inspection observations. The system converts qualitative inputs
             into structured risk indicators integrated into the scoring engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_inspection = st.toggle("Include Site Inspection Data", value=False)
    if not has_inspection:
        st.info("ℹ️ Toggle on to enter primary inspection findings from a factory / business site visit.")
        return None

    col1, col2 = st.columns(2)
    with col1:
        capacity_pct = st.slider("Capacity Utilization (%)", 0, 100, 75, 1)
        machinery = st.text_input("Machinery Condition", placeholder="e.g. Well-maintained CNC machines, 2 offline")
    with col2:
        workforce = st.number_input("Observed Workforce", min_value=0, step=1, value=0)
        expected_workforce = st.number_input("Expected Workforce", min_value=0, step=1, value=0)

    st.markdown("<br>**📋 Inspection Findings**", unsafe_allow_html=True)
    st.caption("Add findings observed during the site visit.")

    num_findings = st.number_input("Number of findings", min_value=0, max_value=10, value=0, step=1)
    findings = []

    for i in range(num_findings):
        with st.expander(f"Finding #{i+1}", expanded=(i == 0)):
            fc1, fc2 = st.columns(2)
            with fc1:
                category = st.selectbox("Category",
                    ["capacity", "machinery", "workforce", "inventory", "safety", "compliance", "other"],
                    key=f"cat_{i}")
                severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=1, key=f"sev_{i}")
            with fc2:
                observation = st.text_area("Observation *", placeholder="Describe what was observed...",
                    key=f"obs_{i}", height=80)
                evidence = st.text_input("Supporting Evidence", placeholder="Photo ref, document number...",
                    key=f"evi_{i}")
            if observation.strip():
                findings.append({
                    "category": category, "severity": severity,
                    "observation": observation.strip(),
                    "evidence": evidence.strip() if evidence.strip() else None,
                })

    payload: Dict = {"findings": findings, "capacity_utilization": capacity_pct / 100.0}
    if machinery.strip():
        payload["machinery"] = machinery.strip()
    if workforce > 0:
        payload["workforce"] = workforce
    if expected_workforce > 0:
        payload["expected_workforce"] = expected_workforce

    st.markdown(
        f'<div style="color:#4ade80;font-size:0.85rem;">✅ Inspection ready — {len(findings)} finding(s)</div>',
        unsafe_allow_html=True,
    )
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# Results rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_risk_overview(result: Dict):
    ic = result.get("intellicredit_risk", {}) or {}
    ai_score = float(result.get("final_score", 0) or 0)
    ic_score = float(ic.get("final_score", 0) or 0)
    ic_category = ic.get("risk_category", "Unknown") or "Unknown"
    ic_status = ic.get("status", "unknown") or "unknown"
    conf = float(result.get("confidence", 0) or 0)
    conf_level = (result.get("confidence_level", "unknown") or "unknown").lower()

    ai_color   = _risk_color(ai_score)
    ic_color   = _risk_color(ic_score)
    conf_color = "#4ade80" if conf_level == "high" else "#fbbf24" if conf_level == "medium" else "#f87171"

    ai_donut   = _donut_svg(ai_score, ai_color)
    ic_donut   = _donut_svg(ic_score if ic_status == "success" else 0, ic_color)
    conf_donut = _donut_svg(conf * 100, conf_color)

    col1, col2, col3 = st.columns(3)
    with col1:
        ai_badge_txt = "High Risk" if ai_score >= 75 else "Medium Risk" if ai_score >= 45 else "Low Risk"
        ai_badge_cls = "b-high" if ai_score >= 75 else "b-med" if ai_score >= 45 else "b-low"
        st.markdown(
            f'<div class="gauge-card" style="border-bottom:3px solid {ai_color};">'
            f'<div class="gauge-label">AI Brain Risk Score</div>'
            f'<div class="donut-wrap">{ai_donut}</div>'
            f'<div class="gauge-sub">/ 100 · Higher = Higher Risk</div>'
            f'<span class="gauge-badge {ai_badge_cls}">{ai_badge_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        ic_badge_cls = _badge_cls(ic_category)
        st.markdown(
            f'<div class="gauge-card" style="border-bottom:3px solid {ic_color};">'
            f'<div class="gauge-label">IntelliCredit Risk Score</div>'
            f'<div class="donut-wrap">{ic_donut}</div>'
            f'<div class="gauge-sub">/ 100 · Higher = Higher Risk</div>'
            f'<span class="gauge-badge {ic_badge_cls}">{ic_category}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        conf_label_txt = conf_level.upper() if conf_level != "unknown" else "N/A"
        st.markdown(
            f'<div class="gauge-card" style="border-bottom:3px solid {conf_color};">'
            f'<div class="gauge-label">Data Confidence</div>'
            f'<div class="donut-wrap">{conf_donut}</div>'
            f'<div class="gauge-sub">Extraction reliability</div>'
            f'<span class="gauge-badge" style="color:{conf_color};border-color:{conf_color}44;'
            f'background:{conf_color}18;">{conf_label_txt} · {conf:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


    st.markdown(
        '<div style="margin-top:12px;background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.12);'
        'border-radius:10px;padding:10px 16px;color:#334155;font-size:0.76rem;text-align:center;">'
        '⚠️ AI Brain score and IntelliCredit score use <strong>different scales</strong> and must not be '
        'directly compared. Neither constitutes a lending decision.</div>',
        unsafe_allow_html=True,
    )


def render_financials_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>💰 Module 1 — Extracted Financial Data</h3>
          <p>Values extracted by the Vectorless RAG engine. Always cross-verify against the original documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="warn-panel"><strong>⚠️ Extraction note:</strong> '
        'If your PDF has a Ratio Analysis page (with % values) <em>before</em> the income statement, '
        'those percentages may be extracted instead of absolute ₹ values. '
        'Negative revenue and small OCF values are penalised automatically — but always verify.</div>',
        unsafe_allow_html=True,
    )

    financials = result.get("financials", {})
    field_conf = result.get("field_confidence", {})

    if not financials:
        st.info("ℹ️ No financial fields were extracted from the uploaded documents.")
        return

    FIELD_META = {
        "revenue":             ("📈 Revenue",            "#818cf8"),
        "net_profit":          ("📉 Net Profit",          "#4ade80"),
        "total_debt":          ("💳 Total Debt",          "#f87171"),
        "bank_deposits":       ("🏦 Bank Deposits",       "#38bdf8"),
        "gst_revenue":         ("🧾 GST Revenue",         "#a78bfa"),
        "receivables":         ("📦 Receivables",         "#fbbf24"),
        "operating_cash_flow": ("🔄 Operating Cash Flow", "#34d399"),
    }

    rows = []
    for field, value in financials.items():
        if value is None or isinstance(value, (list, dict)):
            continue
        label, color = FIELD_META.get(field, (field.replace("_", " ").title(), "#94a3b8"))
        conf_info  = field_conf.get(field, {})
        conf_str   = _conf_label(conf_info)
        conf_color = {"High": "#4ade80", "Medium": "#fbbf24", "Low": "#f87171"}.get(conf_str, "#475569")
        rows.append((label, value, conf_str, conf_color, color))

    if rows:
        for i in range(0, len(rows), 3):
            chunk = rows[i:i + 3]
            cols  = st.columns(len(chunk))
            for col, (label, value, conf_str, conf_color, val_color) in zip(cols, chunk):
                with col:
                    st.markdown(
                        f'<div class="fin-card">'
                        f'<div class="fin-label">{label}</div>'
                        f'<div class="fin-value" style="color:{val_color};">{_fmt_currency(value)}</div>'
                        f'<span class="fin-conf" style="color:{conf_color};background:{conf_color}18;'
                        f'border:1px solid {conf_color}30;">{conf_str} confidence</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("ℹ️ No financial values could be extracted.")

    with st.expander("📐 Financial Ratios", expanded=False):
        ratios = result.get("ratios", {})
        if ratios:
            import pandas as pd
            def _fmt_ratio(v) -> str:
                if not isinstance(v, float):
                    return str(v)
                if v == 0.0:
                    return "0.0000"
                if abs(v) < 0.01:
                    return f"{v:.4g}"   # e.g. 0.0026 instead of 0.0000
                return f"{v:.4f}"
            ratio_rows = [
                {"Ratio": k.replace("_", " ").title(), "Value": _fmt_ratio(v)}
                for k, v in ratios.items() if v is not None
            ]

            if ratio_rows:
                st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No computable ratios — insufficient data or comparability flags not set.")
        else:
            st.info("No ratio data available.")

    with st.expander("🔬 Field Extraction Evidence", expanded=False):
        ic       = result.get("intellicredit_risk", {}) or {}
        evidence = ic.get("evidence", {})
        if evidence:
            import pandas as pd
            ev_rows = []
            for field, rec in evidence.items():
                if not isinstance(rec, dict):
                    continue
                ev_rows.append({
                    "Field":            field.replace("_", " ").title(),
                    "Value":            str(rec.get("value", "—")),
                    "Page":             str(rec.get("page", "—")),
                    "Source":           str(rec.get("source") or "—"),
                    "Evidence Status":  str(rec.get("evidence_status") or "—"),
                })
            if ev_rows:
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No field evidence records.")
        else:
            st.info("Evidence not available — run with metadata enabled.")


def render_cross_check_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>⚖️ Module 2 — Structured Cross-Checking Engine</h3>
          <p>Automated comparison of financial data across uploaded sources. Discrepancies trigger risk alerts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    flags = result.get("risk_flags", [])
    if flags:
        st.markdown("**🚩 Risk Flags**")
        flag_chips = []
        for f in flags:
            if isinstance(f, dict):
                ftype  = f.get("type", "flag").replace("_", " ").title()
                sev    = f.get("severity", "")
                metric = f.get("metric", "")
                value  = f.get("value", "")
                label  = ftype
                if metric:
                    label += f" · {metric.replace('_',' ')}"
                if value != "":
                    label += f" = {value}"
                sev_color = {"high": "#f87171", "medium": "#facc15", "low": "#4ade80"}.get(sev, "#f87171")
                flag_chips.append(
                    f'<span style="display:inline-block;background:rgba(239,68,68,0.12);'
                    f'color:{sev_color};border:1px solid {sev_color}44;border-radius:8px;'
                    f'padding:5px 12px;font-size:0.82rem;margin:4px 4px 0 0;">'
                    f'⚠ {label}</span>'
                )
            else:
                flag_chips.append(f'<span class="flag-chip">⚠ {f}</span>')
        st.markdown(f'<div style="margin-bottom:16px;">{" ".join(flag_chips)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="info-chip">✅ No risk flags detected</span><br><br>', unsafe_allow_html=True)

    # IntelliCredit breakdown
    ic = result.get("intellicredit_risk", {})
    if ic and ic.get("status") == "success":
        breakdown = ic.get("breakdown", {})
        st.markdown("#### 📊 IntelliCredit Risk Breakdown")
        _score_bar("💹 Financial Risk", breakdown.get("financial", 0))
        _score_bar("🔍 Fraud-Risk Indicators", breakdown.get("fraud", 0))
        _score_bar("🌐 External Risk", breakdown.get("external", 0))
        _score_bar("🏭 Operational Risk", breakdown.get("operational", 0))

        reasons = ic.get("reasons", [])
        if reasons:
            st.markdown("#### 🧠 Risk Reasons")
            for r in reasons:
                st.markdown(
                    f'<div style="background:rgba(239,68,68,0.07);border-left:3px solid #f87171;'
                    f'padding:8px 14px;border-radius:0 8px 8px 0;color:#fca5a5;font-size:0.85rem;margin-bottom:6px;">'
                    f'⚠ {r}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("🔢 Score Audit"):
            st.json(ic.get("score_audit", {}))

        ic_warnings = ic.get("warnings", [])
        if ic_warnings:
            with st.expander("⚠️ IntelliCredit Warnings"):
                for w in ic_warnings:
                    st.markdown(f"- {w}")
    elif ic:
        st.warning(f"IntelliCredit: {ic.get('reason', 'Assessment unavailable')}")

    # Reconciliation
    rec = (result.get("_metadata") or {}).get("reconciliation") or {}
    st.markdown("#### 🔄 Cross-Source Reconciliation")

    # Always explain how reconciliation works
    st.markdown(
        '''
        <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                    border-radius:10px;padding:12px 16px;margin-bottom:14px;color:#7dd3fc;font-size:0.85rem;">
          <strong>ℹ️ How reconciliation works:</strong> The engine compares the same financial field
          extracted from <em>different</em> source documents (e.g., bank statement vs GST filing vs annual report).
          Upload <strong>2 or more documents</strong> to enable full cross-source comparison.
          With a single document the engine compares values across different pages/sections of that document.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if not rec:
        st.info("Reconciliation data not available — run analysis with at least one document.")
    else:
        rec_status = rec.get("status", "not_run")
        comparisons = rec.get("comparisons", [])
        status_counts = rec.get("status_counts", {})
        obs_count = rec.get("observation_count", 0)
        cand_count = rec.get("candidate_count", 0)

        # Status row
        status_color = {
            "completed": "#4ade80", "insufficient_evidence": "#facc15", "not_run": "#94a3b8"
        }.get(rec_status, "#94a3b8")
        st.markdown(
            f'<div style="color:{status_color};font-weight:600;margin-bottom:12px;">'
            f'Reconciliation Status: {rec_status.replace("_"," ").title()} '
            f'&nbsp;|&nbsp; {obs_count} observations from {cand_count} candidate extractions</div>',
            unsafe_allow_html=True,
        )

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, key, label, color in [
            (sc1, "consistent", "Consistent", "#4ade80"),
            (sc2, "discrepancy", "Discrepancies", "#f87171"),
            (sc3, "incomparable", "Incomparable", "#facc15"),
            (sc4, "insufficient_evidence", "Needs Evidence", "#94a3b8"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-tile"><div class="mt-label">{label}</div>'
                    f'<div class="mt-value" style="color:{color};">{status_counts.get(key, 0)}</div></div>',
                    unsafe_allow_html=True,
                )

        if comparisons:
            import pandas as pd
            st.markdown("<br>", unsafe_allow_html=True)
            cols_to_show = [c for c in ["field", "source_a", "source_b", "status", "reason"] if c in comparisons[0]]
            df = pd.DataFrame(comparisons)[cols_to_show]
            df.columns = [c.replace("_", " ").title() for c in cols_to_show]
            st.dataframe(df, use_container_width=True, hide_index=True)

            if rec_status == "insufficient_evidence":
                st.info(
                    "💡 All comparisons have insufficient evidence because reporting period metadata "
                    "could not be extracted from the document. Upload documents that include explicit "
                    "period labels (e.g., FY2025, Q1 2026) near financial figures, or upload "
                    "multiple documents from different sources."
                )
        else:
            st.info("No cross-field comparison pairs found in the uploaded documents.")


def render_research_results_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>🌐 Module 3 — AI Research Agent Findings</h3>
          <p>Signals gathered about the company, promoters, and industry from public sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Always show activation instructions
    st.markdown(
        '''
        <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                    border-radius:10px;padding:12px 16px;margin-bottom:14px;color:#7dd3fc;font-size:0.85rem;">
          <strong>ℹ️ How to activate:</strong>
          <ul style="margin:6px 0 0 0;padding-left:18px;">
            <li><strong>Live Web Research</strong> — toggle <em>"AI Web Research Agent"</em> in the sidebar
                OR select <em>"Live Web Research"</em> in the Research Agent input tab before running.</li>
            <li><strong>Manual Signals</strong> — paste a JSON array of external signals in the
                Research Agent input tab (e.g., legal disputes, regulatory news).</li>
          </ul>
          Research topics: Legal proceedings · Regulatory actions · Promoter records · Business news · Sector risks
        </div>
        ''',
        unsafe_allow_html=True,
    )

    research = result.get("research", {})
    if not research:
        st.warning("⚠️ No web research was performed in this run. See activation instructions above.")
        return

    status = research.get("status", "unknown")
    status_colors = {"success": "#4ade80", "no_results": "#facc15",
                     "no_usable_evidence": "#facc15", "failed": "#f87171", "not_configured": "#94a3b8"}
    color = status_colors.get(status, "#94a3b8")
    st.markdown(
        f'<div style="color:{color};font-weight:600;margin-bottom:12px;">'
        f'Research Status: {status.replace("_"," ").title()}</div>',
        unsafe_allow_html=True,
    )

    findings = research.get("findings", [])
    if findings:
        for f in findings:
            sev_colors = {"low": "#4ade80", "medium": "#facc15", "high": "#f87171", "critical": "#ef4444"}
            sev = f.get("severity", "medium")
            sc = sev_colors.get(sev, "#94a3b8")
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.7);border:1px solid {sc}33;border-left:4px solid {sc};'
                f'border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span style="color:{sc};font-weight:700;font-size:0.82rem;text-transform:uppercase;">'
                f'{f.get("type","signal")} · {sev}</span>'
                f'<span style="color:#475569;font-size:0.75rem;">confidence: {f.get("confidence",0):.0%}</span>'
                f'</div><div style="color:#cbd5e1;font-size:0.87rem;">{f.get("evidence","")}</div></div>',
                unsafe_allow_html=True,
            )
        sources = research.get("sources", [])
        if sources:
            with st.expander("🔗 Research Sources"):
                for s in sources:
                    url = s.get("url", "")
                    title = s.get("title") or "Source"
                    if url.startswith(("http://", "https://")):
                        st.markdown(f"- [{title}]({url}) — {s.get('published_at','')}")
    else:
        st.info("No validated research findings returned.")

    warnings = research.get("warnings", [])
    if warnings:
        with st.expander("⚠️ Research Warnings"):
            for w in warnings:
                st.markdown(f"- {w}")


def render_inspection_results_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>🔍 Module 4 — Primary Inspection Summary</h3>
          <p>Validated field inspection findings converted into structured risk indicators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Always show how to activate
    st.markdown(
        '''
        <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                    border-radius:10px;padding:12px 16px;margin-bottom:14px;color:#7dd3fc;font-size:0.85rem;">
          <strong>ℹ️ How to activate:</strong> Go to the
          <strong>"🔍 4. Primary Inspection"</strong> input tab above, toggle
          <em>"Include Site Inspection Data"</em>, fill in capacity utilization,
          machinery condition, workforce numbers, and any findings from the field visit,
          then re-run the analysis.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    inspection = result.get("inspection", {})
    status = inspection.get("status", "not_provided")

    if status == "not_provided":
        st.warning("⚠️ No primary inspection data was provided for this analysis run. See instructions above.")
        return

    op = inspection.get("operational_data", {})
    findings = inspection.get("findings", [])

    if op:
        cols = st.columns(max(len(op), 1))
        for idx, (k, v) in enumerate(op.items()):
            with cols[idx]:
                label = k.replace("_", " ").title()
                display = f"{v:.0%}" if isinstance(v, float) and "utilization" in k else str(v)
                st.markdown(
                    f'<div class="metric-tile"><div class="mt-label">{label}</div>'
                    f'<div class="mt-value" style="font-size:1.4rem;">{display}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    if findings:
        st.markdown("**📋 Inspection Findings**")
        sev_colors = {"low": "#4ade80", "medium": "#facc15", "high": "#f87171", "critical": "#ef4444"}
        for f in findings:
            sev = f.get("severity", "medium")
            sc = sev_colors.get(sev, "#94a3b8")
            evidence_html = (
                f"<div style='color:#64748b;font-size:0.78rem;margin-top:4px;'>Evidence: {f.get('evidence')}</div>"
                if f.get("evidence") else ""
            )
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.6);border:1px solid {sc}33;border-left:4px solid {sc};'
                f'border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:8px;">'
                f'<span style="color:{sc};font-weight:700;font-size:0.78rem;text-transform:uppercase;">'
                f'{f.get("category","other")} · {sev}</span>'
                f'<div style="color:#cbd5e1;font-size:0.87rem;margin-top:4px;">{f.get("observation","")}</div>'
                f'{evidence_html}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No inspection findings recorded.")

    inspect_warnings = inspection.get("warnings", [])
    if inspect_warnings:
        with st.expander("⚠️ Inspection Warnings"):
            for w in inspect_warnings:
                st.markdown(f"- {w}")


def render_explainability_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>🧠 Module 5 — Explainable AI Decision System</h3>
          <p>Key factors that drove the risk score, presented in a structured explanation for credit officer review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    key_drivers = result.get("key_risk_drivers", [])
    insights = result.get("insights", [])
    risk_contributions = result.get("risk_contributions", {})
    warnings = result.get("warnings", [])

    if key_drivers:
        st.markdown("#### 🎯 Key Risk Drivers")
        for driver in key_drivers[:8]:
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.07);border-left:3px solid #f87171;'
                f'padding:8px 14px;border-radius:0 8px 8px 0;color:#fca5a5;font-size:0.85rem;margin-bottom:6px;">'
                f'⚠ {driver}</div>',
                unsafe_allow_html=True,
            )

    if risk_contributions:
        import pandas as pd
        st.markdown("#### 📊 Risk Factor Contributions")
        contrib_data = {
            k.replace("_", " ").title(): v
            for k, v in risk_contributions.items()
            if isinstance(v, (int, float))
        }
        if contrib_data:
            df = pd.DataFrame.from_dict(contrib_data, orient="index", columns=["Contribution"])
            df = df.sort_values("Contribution", ascending=False)
            st.bar_chart(df)

    if insights:
        st.markdown("#### 💡 AI-Generated Insights")
        for ins in insights[:8]:
            st.markdown(
                f'<div style="background:rgba(56,189,248,0.06);border-left:3px solid #38bdf8;'
                f'padding:8px 14px;border-radius:0 8px 8px 0;color:#7dd3fc;font-size:0.85rem;margin-bottom:6px;">'
                f'{ins}</div>',
                unsafe_allow_html=True,
            )

    clean_warnings = [w for w in (warnings or []) if w and isinstance(w, str)]
    if clean_warnings:
        with st.expander("⚠️ Analysis Warnings"):
            for w in clean_warnings:
                st.markdown(f"- {w}")


def render_cam_tab(result: Dict):
    st.markdown(
        """
        <div class="module-card">
          <h3>📄 Module 5 — Automated Credit Appraisal Memo (CAM)</h3>
          <p>AI-generated structured memo for credit committee review. Draft only — not a lending decision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cam = result.get("credit_appraisal_memo", {})
    if not cam:
        st.info("ℹ️ CAM not generated. Enable 'Generate CAM Report' in the sidebar.")
        return

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f'<div class="metric-tile"><div class="mt-label">Status</div>'
            f'<div class="mt-value" style="font-size:0.95rem;color:#818cf8;">{cam.get("status","—").title()}</div></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="metric-tile"><div class="mt-label">Decision</div>'
            f'<div class="mt-value" style="font-size:0.85rem;color:#facc15;">'
            f'{cam.get("decision","—").replace("_"," ").title()}</div></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f'<div class="metric-tile"><div class="mt-label">Discrepancies</div>'
            f'<div class="mt-value" style="color:#f87171;">{cam.get("discrepancy_count",0)}</div></div>',
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            f'<div class="metric-tile"><div class="mt-label">Inspection Findings</div>'
            f'<div class="mt-value" style="color:#facc15;">{cam.get("inspection_finding_count",0)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    markdown_content = cam.get("markdown", "")
    if markdown_content:
        st.markdown('<div class="cam-wrapper">', unsafe_allow_html=True)
        st.markdown(markdown_content)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        dl1, dl2, _ = st.columns([1, 1, 3])
        company_slug = (cam.get('company', 'company') or 'company').replace(' ', '_')
        date_slug    = (cam.get('generated_at', '') or '').split('T')[0]
        with dl1:
            st.download_button(
                label="⬇️ Download Markdown",
                data=markdown_content,
                file_name=f"CAM_{company_slug}_{date_slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="⬇️ Download TXT",
                data=markdown_content,
                file_name=f"CAM_{company_slug}_{date_slug}.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.warning("CAM content is empty.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="intellicredit-header">
          <h1>🏦 Credix</h1>
          <p>AI-Powered Credit Risk Appraisal Platform &nbsp;·&nbsp; Powered by IntelliCredit Engine &nbsp;·&nbsp;
             Multi-Source Ingestion &nbsp;·&nbsp; Deterministic RAG &nbsp;·&nbsp; Autonomous CAM Generation</p>
          <div class="ic-badge-row">
            <span class="ic-badge">🔒 Secure Upload</span>
            <span class="ic-badge">🤖 AI-Powered</span>
            <span class="ic-badge">📊 5-Module Pipeline</span>
            <span class="ic-badge">📄 Auto-CAM</span>
            <span class="ic-badge">🔍 Explainable AI</span>
            <span class="ic-badge">⚖️ Cross-Checking</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = render_sidebar()

    # ── Input configuration ──────────────────────────────────────────────────
    st.markdown("### 📥 Input Configuration")
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "📂 1. Document Upload",
        "🌐 3. AI Research Agent",
        "🔍 4. Primary Inspection",
    ])

    with input_tab1:
        saved_files = render_ingestion_tab()
    with input_tab2:
        external_signals = render_research_tab()
    with input_tab3:
        inspection_data = render_inspection_tab()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Run button + real-time status chips ──────────────────────────────────
    col_btn, col_hints = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🚀 Run IntelliCredit Analysis", use_container_width=True)
    with col_hints:
        chips = []
        if config["company_name"]:
            chips.append(f'<span class="ok-chip">✅ {config["company_name"]}</span>')
        else:
            chips.append('<span class="flag-chip">❌ No company name</span>')
        if saved_files:
            chips.append(f'<span class="ok-chip">✅ {len(saved_files)} file(s) ready</span>')
        else:
            chips.append('<span class="flag-chip">❌ No documents</span>')
        if config["perform_web_research"] or st.session_state.get("perform_web_research_override"):
            chips.append('<span class="info-chip">🌐 Research ON</span>')
        if config["include_cam"]:
            chips.append('<span class="info-chip">📄 CAM ON</span>')
        st.markdown(
            f'<div style="padding:8px 0;line-height:2.4;">{" ".join(chips)}</div>',
            unsafe_allow_html=True,
        )

    if run_clicked:
        if not config["company_name"]:
            st.error("❌ Enter a company name in the sidebar.")
            st.stop()
        if not saved_files:
            st.error("❌ Upload at least one financial document.")
            st.stop()

        perform_web_research = (
            config["perform_web_research"]
            or st.session_state.get("perform_web_research_override", False)
        )

        with st.spinner("🔄 Running AI Brain + IntelliCredit pipeline…"):
            try:
                from main import AIBrainPipeline

                pipeline = AIBrainPipeline()
                result = pipeline.run(
                    company=config["company_name"],
                    file_paths=list(saved_files),
                    include_metadata=True,
                    include_retrieval=False,
                    inspection_data=inspection_data,
                    external_signals=external_signals if external_signals else None,
                    include_cam=config["include_cam"],
                    perform_web_research=perform_web_research,
                )
                st.session_state["result"]      = result
                st.session_state["run_success"] = True
                st.toast("✅ Analysis complete!", icon="🏦")
            except Exception as exc:
                st.error(f"❌ Pipeline error: {type(exc).__name__}: {exc}")
                st.exception(exc)
                st.session_state["run_success"] = False

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.get("run_success") and "result" in st.session_state:
        result: Dict = st.session_state["result"]

        if result.get("status") in ("failed", "insufficient_data"):
            st.error(f"⚠️ Analysis failed: {result.get('reason', 'Unknown reason')}")
            for w in result.get("warnings", []):
                if w:
                    st.warning(w)
            st.stop()

        company_disp = result.get("company") or config["company_name"]
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(f"### 📊 Results — **{company_disp}**")

        render_risk_overview(result)
        st.markdown("<br>", unsafe_allow_html=True)

        out1, out2, out3, out4, out5, out6 = st.tabs([
            "💰 Financials (M1)",
            "⚖️ Cross-Check (M2)",
            "🌐 Research (M3)",
            "🔍 Inspection (M4)",
            "🧠 Explainable AI (M5)",
            "📄 CAM Report (M5)",
        ])

        with out1:
            render_financials_tab(result)
        with out2:
            render_cross_check_tab(result)
        with out3:
            render_research_results_tab(result)
        with out4:
            render_inspection_results_tab(result)
        with out5:
            render_explainability_tab(result)
        with out6:
            render_cam_tab(result)

        with st.expander("🛠️ Raw Pipeline Output (JSON)", expanded=False):
            st.json(result)

        st.markdown(
            '<div class="ic-footer">IntelliCredit AI · Draft outputs only · '
            'Not a lending decision · Always verify with original documents</div>',
            unsafe_allow_html=True,
        )

    elif not run_clicked:
        st.markdown(
            """
            <div style="text-align:center;padding:70px 20px;">
              <div style="font-size:5rem;margin-bottom:14px;">🏦</div>
              <div style="font-size:1.45rem;font-weight:800;color:#818cf8;margin-bottom:10px;">Ready to Analyse</div>
              <div style="color:#334155;max-width:460px;margin:0 auto;font-size:0.9rem;line-height:2;">
                1. Enter a <strong style="color:#64748b;">company name</strong> in the sidebar<br>
                2. Upload financial documents in <strong style="color:#64748b;">Document Upload</strong><br>
                3. Optionally configure Research Agent &amp; Inspection<br>
                4. Click <strong style="color:#818cf8;">🚀 Run IntelliCredit Analysis</strong>
              </div>
              <div style="margin-top:28px;color:#1e293b;font-size:0.72rem;">
                IntelliCredit AI · Draft outputs only · Not a lending decision
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
