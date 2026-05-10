import streamlit as st
from streamlit_option_menu import option_menu
import time

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem; margin-top: 1rem;">
                <h1 style="color: #ff4b4b; font-size: 2.2rem; font-weight: 800; margin-bottom: 0; letter-spacing: -1px;">ResQAI</h1>
                <p style="color: #7a7a92; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 2px;">Intelligence Core</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        selected = option_menu(
            menu_title=None,
            options=[
                "Home", 
                "Multimodal Analysis", 
                "Vision Intelligence", 
                "Voice Intelligence", 
                "Geospatial Command", 
                "Memory Timeline", 
                "Agent Monitoring", 
                "RAG Knowledge"
            ],
            icons=[
                "house-door-fill", 
                "cpu-fill", 
                "eye-fill", 
                "mic-fill", 
                "globe", 
                "clock-history", 
                "activity", 
                "database-fill"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#8c8c9e", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.9rem",
                    "text-align": "left",
                    "margin": "0.2rem 0px",
                    "--hover-color": "#202029",
                    "color": "#a0a0b8",
                    "border-radius": "4px"
                },
                "nav-link-selected": {
                    "background-color": "#ff4b4b", 
                    "color": "white", 
                    "font-weight": "600"
                },
            }
        )

        st.markdown("<hr style='border-color: #2d2d3a;'>", unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-header'>SYSTEM HEALTH</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CPU Core", value="32%")
        with col2:
            st.metric(label="RAM", value="12GB")
            
        st.markdown("<div class='sidebar-header' style='margin-top: 1rem;'>ORCHESTRATION</div>", unsafe_allow_html=True)
        st.metric(label="Latency", value="240ms", delta="-12ms", delta_color="inverse")
        
        st.markdown("<hr style='border-color: #2d2d3a;'>", unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-header'>OPERATIONAL MODE</div>", unsafe_allow_html=True)
        st.checkbox("Simulation Mode", value=False)
        st.checkbox("Local Inference (Gemma)", value=True)

        return selected
