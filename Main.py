import base64
import streamlit as st

from src.views.modals.weatherModal import render_weather_analysis
from src.views.modals.editInfoModal import render_edit_info_modal
from src.views.modals.messageModal import render_message_sender

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


# ======== 모덜 ========
# 날씨
if "weather_modal_open" not in st.session_state:
    st.session_state.weather_modal_open = False

def on_weather_modal_dismiss():
    st.session_state.weather_modal_open = False

@st.dialog("날씨별 노쇼 예측 상세 분석", width='large', on_dismiss=on_weather_modal_dismiss)
def weather_modal():
    render_weather_analysis()

if st.session_state.weather_modal_open:
    weather_modal()

# 수정
if "open_edit_modal" not in st.session_state:
    st.session_state.open_edit_modal = False

def on_edit_modal_dismiss():
    st.session_state.open_edit_modal = False

@st.dialog("회원 정보 수정", on_dismiss=on_edit_modal_dismiss)
def edit_modal():
    render_edit_info_modal()

if st.session_state.open_edit_modal:
    edit_modal()

# 메세지
if "open_message_modal" not in st.session_state:
    st.session_state.open_message_modal = False

def on_msg_modal_dismiss():
    st.session_state.open_message_modal = False

@st.dialog("메세지 전송", on_dismiss=on_msg_modal_dismiss)
def message_modal():
    render_message_sender()

if st.session_state.open_message_modal:
    message_modal()


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
            
        /* Dashboard - 요일/시간대 */
        .card-title, div[data-testid="stHeadingWithActionElements"] h3 {
            font-size: 20px;
            font-weight: 700;
            margin-left: 1rem;
        }

        table {
            width: 100%;
            border-collapse: separate !important;
            border-spacing: 8px;
        }

        th {
            text-align: center !important;
            font-weight: 600;
            border: 0 !important
        }

        th.time {
            width: 80px;
            text-align: right;
            padding-right: 8px;
            color: #374151;
            border: 0 !important
        }

        td {
            border-radius: 0.25rem;
            padding: 12px !important;
            min-height: 80px;
            vertical-align: top;
        }

        .cell-time {
            font-size: 12px;
            color: #374151;
        }

        .cell-rate {
            font-size: 18px;
            font-weight: 800;
        }

        .low { background: #dcfce7; }
        .mid { background: #fef9c3; }
        .high { background: #fee2e2; }

        .legend {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 20px;
            margin-bottom: 26px;
            font-size: 14px;
        }

        .legend span {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .box {
            width: 14px;
            height: 14px;
            border-radius: 4px;
        }
    </style>
""", 
unsafe_allow_html=True)