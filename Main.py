import base64
import streamlit as st

st.set_page_config(
    page_title="노쇼 프리",
    page_icon=":material/person_check:",
    layout='wide'
)

st.logo("assets/images/LOGO.png", size="large")

pages = st.navigation([
    st.Page("src/views/Dashboard.py", title="대시보드", icon=":material/dashboard:"),
    st.Page("src/views/CustomerList.py", title="고객 관리", icon=":material/group:")
])

pages.run()


# ======== 스타일 설정 ========
st.markdown("""
    <style>
        [data-testid="stSidebarHeader"] {
            align-items: flex-start;
            height: auto;
            padding-top: 2rem;
        }

        [data-testid="stSidebarLogo"] {
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
        }
        
        [data-testid="stSidebarHeader"] > div:first-child {
            width: 100%;
        }

        [data-testid="stSidebarCollapseButton"] {
            position: absolute;
            right: 10px;
            top: 10px;
        }
    </style>
""", 
unsafe_allow_html=True)
