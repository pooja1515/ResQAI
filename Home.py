import streamlit as st
import time
import os

# Streamlit Page Config must be the first Streamlit command
st.set_page_config(
    page_title="ResQAI - Operational Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components.sidebar import render_sidebar

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject custom CSS
load_css()

# Render Sidebar Navigation
selected_page = render_sidebar()

if selected_page == "Home":
    
    # Optional: Initial loading simulation
    if "initialized" not in st.session_state:
        with st.spinner("Initializing ResQAI Intelligence Core..."):
            time.sleep(1)
        st.session_state["initialized"] = True

    # HERO SECTION
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">ResQAI Intelligence Core</h1>
            <h3 class="hero-subtitle">Operational Disaster Intelligence Platform</h3>
            <p class="hero-text">ResQAI transforms fragmented disaster signals into coordinated operational intelligence using multimodal AI orchestration powered by Gemma.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # OPERATIONAL STATUS CARDS
    st.markdown("<h3 class='section-header'>Agent Fleet Status</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="status-card status-active">
                <div class="card-title">👁️ Vision Agent</div>
                <div class="card-status">● ONLINE & ACTIVE</div>
                <div class="card-detail">Processing UAV streams and satellite imagery...</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="status-card status-active">
                <div class="card-title">🎙️ Voice Agent</div>
                <div class="card-status">● ONLINE & ACTIVE</div>
                <div class="card-detail">Listening to emergency frequencies and extracting entities...</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="status-card status-critical">
                <div class="card-title">🌪️ Weather Agent</div>
                <div class="card-status status-text-critical">● CRITICAL INCIDENT</div>
                <div class="card-detail">Severe cyclone approaching coastal Sector 7</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="status-card status-active">
                <div class="card-title">🧠 Fusion Core</div>
                <div class="card-status">● SYNTHESIZING</div>
                <div class="card-detail">Orchestrating multimodal signals via Gemma...</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # PREDICTIVE ESCALATION RADAR & QUICK NAVIGATION
    col_radar, col_nav = st.columns([1.6, 1])
    
    with col_radar:
        st.markdown("<h3 class='section-header'>Predictive Escalation Radar</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="radar-panel">
                <div class="radar-header">Current Threat Level: <span class="highlight-moderate">MODERATE</span></div>
                <div class="radar-projection">Projection: <span class="highlight-critical">CRITICAL</span> within 6 hours</div>
                <hr class="radar-divider">
                <div class="radar-forecast">
                    <strong>Operational Forecast:</strong>
                    <ul>
                        <li>High probability of flash flooding in lower elevations (85%).</li>
                        <li>Communication network instability detected in Zone B.</li>
                        <li>Recommend staging aerial rescue assets for immediate deployment.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with col_nav:
        st.markdown("<h3 class='section-header'>Quick Actions</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="quick-nav-grid">
                <div class="quick-nav-btn">📊 Multimodal Analysis</div>
                <div class="quick-nav-btn">🌍 Geospatial Command</div>
                <div class="quick-nav-btn">📈 Agent Monitoring</div>
                <div class="quick-nav-btn">📚 RAG Knowledge</div>
            </div>
            """, unsafe_allow_html=True
        )

else:
    # Placeholder for other modules
    st.markdown(
        f"""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title">{selected_page}</h1>
            <p class="hero-text" style="color: #ffb700;">⚠ Module Offline / Under Construction</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
