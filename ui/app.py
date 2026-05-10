from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st
from streamlit_option_menu import option_menu


@dataclass(frozen=True)
class SystemStatus:
    gemma_core: str = "Online"
    local_inference: str = "Active"
    edge_mode: str = "Enabled"


CSS = r"""
<style>
  :root{
    --bg: #070A0F;
    --panel: rgba(255,255,255,0.06);
    --panel-2: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.10);
    --text: rgba(255,255,255,0.92);
    --muted: rgba(255,255,255,0.62);
    --muted2: rgba(255,255,255,0.45);
    --cyan: #27D7FF;
    --orange: #FFB020;
    --red: #FF3B3B;
    --green: #26D07C;
    --shadow: 0 12px 30px rgba(0,0,0,0.55);
    --radius: 18px;
  }

  /* App background */
  .stApp{
    background: radial-gradient(1200px 600px at 35% -10%, rgba(39,215,255,0.10), transparent 60%),
                radial-gradient(900px 500px at 80% 0%, rgba(255,176,32,0.08), transparent 55%),
                radial-gradient(900px 600px at 20% 35%, rgba(255,59,59,0.06), transparent 55%),
                var(--bg);
    color: var(--text);
  }

  /* Reduce default padding a bit */
  section.main > div { padding-top: 1.1rem; }

  /* Sidebar: dark + glow */
  [data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(10,14,22,0.96), rgba(6,8,12,0.98));
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  [data-testid="stSidebar"]::before{
    content:"";
    position:absolute;
    top:0; right:-1px; bottom:0; left:0;
    pointer-events:none;
    box-shadow: inset -1px 0 0 rgba(39,215,255,0.08);
  }

  /* Cards */
  .resqai-card{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px 14px;
  }
  .resqai-card:hover{
    border-color: rgba(39,215,255,0.22);
    box-shadow: 0 16px 36px rgba(0,0,0,0.62);
    transform: translateY(-1px);
    transition: all .18s ease;
  }

  .resqai-hero{
    background: linear-gradient(135deg, rgba(39,215,255,0.10), rgba(255,59,59,0.06));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 18px 18px;
    box-shadow: var(--shadow);
  }
  .resqai-title{
    font-size: 20px;
    font-weight: 800;
    letter-spacing: .2px;
    margin: 0;
  }
  .resqai-subtitle{
    font-size: 12.5px;
    color: var(--muted);
    margin-top: 4px;
    margin-bottom: 0;
  }

  /* Top nav wrapper */
  .resqai-topbar{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 6px 10px;
    box-shadow: 0 10px 22px rgba(0,0,0,0.35);
  }

  /* Status chips */
  .chip{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.05);
    color: var(--muted);
    font-size: 12px;
  }
  .dot{
    width: 8px; height: 8px; border-radius: 999px;
    box-shadow: 0 0 16px rgba(39,215,255,0.35);
    background: var(--cyan);
  }
  .dot.red{ background: var(--red); box-shadow: 0 0 18px rgba(255,59,59,0.42); }
  .dot.orange{ background: var(--orange); box-shadow: 0 0 18px rgba(255,176,32,0.42); }
  .dot.green{ background: var(--green); box-shadow: 0 0 18px rgba(38,208,124,0.40); }

  /* Stream area */
  .stream{
    height: 420px;
    overflow:auto;
    padding: 14px 14px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(0,0,0,0.22);
    box-shadow: inset 0 0 0 1px rgba(39,215,255,0.06);
  }
  .line{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 12.5px;
    color: rgba(255,255,255,0.82);
    line-height: 1.55;
    margin: 0 0 6px 0;
  }
  .tag{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(39,215,255,0.10);
    border: 1px solid rgba(39,215,255,0.22);
    color: rgba(39,215,255,0.92);
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    font-size: 11px;
  }
  .tag.red{
    background: rgba(255,59,59,0.10);
    border-color: rgba(255,59,59,0.24);
    color: rgba(255,200,200,0.95);
  }
  .tag.orange{
    background: rgba(255,176,32,0.10);
    border-color: rgba(255,176,32,0.22);
    color: rgba(255,222,170,0.95);
  }

  /* Pulse indicator */
  .pulse{
    width: 9px; height: 9px;
    border-radius: 999px;
    background: var(--cyan);
    position: relative;
  }
  .pulse::after{
    content:"";
    position:absolute;
    left: 50%; top: 50%;
    width: 9px; height: 9px;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    background: rgba(39,215,255,0.35);
    animation: pulse 1.6s ease-out infinite;
  }
  @keyframes pulse{
    0%{ width: 9px; height: 9px; opacity: 0.9; }
    100%{ width: 30px; height: 30px; opacity: 0.0; }
  }

  /* Make Streamlit buttons look more AI-native */
  .stButton > button{
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.92) !important;
    box-shadow: 0 10px 18px rgba(0,0,0,0.32) !important;
    transition: all .18s ease !important;
  }
  .stButton > button:hover{
    border-color: rgba(39,215,255,0.30) !important;
    transform: translateY(-1px) !important;
  }

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"]{ background: transparent; }
  div[data-testid="stToolbar"]{ visibility: hidden; }
  footer{ visibility: hidden; }
</style>
"""


def _init_state() -> None:
    st.session_state.setdefault("saved_incidents", ["Incident #0412 · Mumbai", "Incident #0409 · Chennai"])
    st.session_state.setdefault("upload_history", ["voice_0412.wav", "flood_0412.jpg"])
    st.session_state.setdefault("simulation_mode", True)
    st.session_state.setdefault(
        "stream_lines",
        [
            ("SYSTEM", "ResQAI workspace ready. Upload inputs to start a new multimodal analysis."),
        ],
    )


def _render_sidebar(status: SystemStatus) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 8px 2px 8px;">
              <div style="font-weight:900;font-size:16px;letter-spacing:0.4px;">ResQAI</div>
              <div style="color:rgba(255,255,255,0.55);font-size:12px;margin-top:2px;">
                Multimodal Disaster Intelligence
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("New Analysis", use_container_width=True):
                st.session_state["stream_lines"] = [
                    ("SYSTEM", "New analysis initialized. Provide inputs and run orchestration."),
                ]
        with col_b:
            st.toggle("Simulation", key="simulation_mode")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("**Saved Incidents**")
        for inc in st.session_state["saved_incidents"][:6]:
            st.caption(inc)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("**Upload History**")
        for item in st.session_state["upload_history"][:6]:
            st.caption(item)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("**System Status**")

        def chip(dot_cls: str, label: str, value: str) -> str:
            return f"<span class='chip'><span class='dot {dot_cls}'></span>{label}: <b style='color:rgba(255,255,255,0.88)'>{value}</b></span>"

        st.markdown(
            "<div style='display:flex;flex-direction:column;gap:8px'>"
            + chip("green", "Gemma Core", status.gemma_core)
            + chip("cyan", "Local Inference", status.local_inference)
            + chip("orange", "Edge Mode", status.edge_mode)
            + "</div>",
            unsafe_allow_html=True,
        )


def _top_nav() -> str:
    st.markdown("<div class='resqai-topbar'>", unsafe_allow_html=True)
    selected = option_menu(
        None,
        [
            "Multimodal Analysis",
            "Vision Intelligence",
            "Voice Intelligence",
            "Geospatial",
            "Memory",
            "Agent Monitoring",
            "RAG Knowledge",
        ],
        icons=["layers", "camera", "mic", "map", "clock-history", "activity", "database"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0px", "background-color": "transparent"},
            "icon": {"color": "rgba(255,255,255,0.75)", "font-size": "14px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "center",
                "margin": "0px 6px",
                "padding": "8px 12px",
                "border-radius": "999px",
                "color": "rgba(255,255,255,0.78)",
                "background-color": "rgba(255,255,255,0.03)",
                "border": "1px solid rgba(255,255,255,0.07)",
            },
            "nav-link-selected": {
                "background-color": "rgba(39,215,255,0.12)",
                "border": "1px solid rgba(39,215,255,0.26)",
                "color": "rgba(255,255,255,0.92)",
                "box-shadow": "0 0 0 1px rgba(39,215,255,0.10) inset",
            },
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def _append_stream(tag: str, text: str) -> None:
    st.session_state["stream_lines"].append((tag, text))


def _render_stream() -> None:
    def tag_badge(tag: str) -> str:
        t = tag.upper()
        if t in {"ALERT", "CRITICAL"}:
            return "<span class='tag red'>CRITICAL</span>"
        if t in {"WARN", "MODERATE"}:
            return "<span class='tag orange'>MODERATE</span>"
        if t in {"SYSTEM"}:
            return "<span class='tag'>SYSTEM</span>"
        return f"<span class='tag'>{t}</span>"

    html_lines = []
    for tag, line in st.session_state["stream_lines"][-120:]:
        html_lines.append(
            f"<div class='line'>{tag_badge(tag)}&nbsp;&nbsp;{st._utils.escape_markdown(line)}</div>"
        )

    st.markdown("<div class='stream'>" + "".join(html_lines) + "</div>", unsafe_allow_html=True)


def _simulate_orchestration() -> None:
    steps = [
        ("SYSTEM", "Orchestration started…"),
        ("SYSTEM", "[Vision Agent activated…]"),
        ("SYSTEM", "Flood probability exceeded 91%."),
        ("SYSTEM", "[Voice Agent activated…]"),
        ("SYSTEM", "Detected trapped civilians in Hindi distress audio."),
        ("SYSTEM", "[RAG Agent retrieving emergency protocols…]"),
        ("SYSTEM", "WHO evacuation guidance retrieved."),
        ("SYSTEM", "[Weather Agent activated…]"),
        ("SYSTEM", "Rainfall intensifying; escalation likely."),
        ("SYSTEM", "[Memory Agent correlating prior incidents…]"),
        ("SYSTEM", "Trend indicates worsening conditions over last 60 minutes."),
        ("SYSTEM", "[Fusion Coordinator synthesizing…]"),
        ("CRITICAL", "Overall Risk: CRITICAL · Rescue Priority: HIGH"),
    ]

    placeholder = st.empty()
    lines = st.session_state["stream_lines"][:]
    for tag, msg in steps:
        lines.append((tag, msg))
        st.session_state["stream_lines"] = lines
        with placeholder.container():
            _render_stream()
        time.sleep(0.18)


def main() -> None:
    st.set_page_config(page_title="ResQAI", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()

    status = SystemStatus()
    _render_sidebar(status)

    selected = _top_nav()

    # Main workspace layout
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="resqai-hero">
              <div class="resqai-title">AI‑Powered Multimodal Disaster Intelligence Workspace</div>
              <div class="resqai-subtitle">
                Conversational orchestration · Vision + Voice + Weather + RAG + Memory · Grounded operational outputs
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='resqai-card'>", unsafe_allow_html=True)
        st.markdown("**Streaming Intelligence**")
        st.caption("Live orchestration messages and fused operational summaries.")
        _render_stream()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='resqai-card'>", unsafe_allow_html=True)
        st.markdown("**Orchestration Status**")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("<span class='chip'><span class='pulse'></span> Agents: Ready</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("<span class='chip'><span class='dot orange'></span> Safety: Monitoring</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("<span class='chip'><span class='dot green'></span> Storage: Online</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='resqai-card'>", unsafe_allow_html=True)
        st.markdown("**Multimodal Input**")
        st.caption("Upload or paste signals. This shell is UI-only (no business logic wired yet).")

        img = st.file_uploader("Flood image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        aud = st.file_uploader("Distress audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        text = st.text_area("Emergency text", placeholder="Describe what’s happening (any language)…", height=90)
        location = st.text_input("Location", placeholder="e.g., Mumbai, IN")

        colx, coly = st.columns([1, 1])
        with colx:
            run = st.button("Run Orchestration", use_container_width=True)
        with coly:
            st.button("Save Incident", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='resqai-card'>", unsafe_allow_html=True)
        st.markdown("**Workspace Mode**")
        st.caption(f"Active tab: `{selected}`")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if st.session_state["simulation_mode"]:
            _simulate_orchestration()
        else:
            _append_stream("SYSTEM", "Orchestration requested (wire backend next).")
            _append_stream("SYSTEM", f"Inputs: image={bool(img)} audio={bool(aud)} text={bool(text.strip())} location={bool(location.strip())}")


if __name__ == "__main__":
    main()

