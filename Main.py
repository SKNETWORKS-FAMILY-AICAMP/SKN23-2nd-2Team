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
def load_font(font_path, font_name):
    with open(font_path, "rb") as f:
        data = f.read()
    
    base64_font = base64.b64encode(data).decode()

    font_style = f"""
        <style>
        @font-face {{
            font-family: {font_name};
            src: url(data:font/woff2;base64,{base64_font}) format('woff2');
        }}
        </style>
    """
    st.markdown(font_style, unsafe_allow_html=True)

load_font("assets/fonts/PretendardVariable.woff2", "Pretendard")
load_font("assets/fonts/NanumSquareNeo-Variable.woff2", "NanumSquareNeo")

st.markdown("""
    <style>
        html, body, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, table, div, span, b, [class*="st-"]:not([data-testid="stIconMaterial"]):not(span) {
            font-family: "Pretendard", "NanumSquareNeo", "Source Sans", sans-serif;
        }
            
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
