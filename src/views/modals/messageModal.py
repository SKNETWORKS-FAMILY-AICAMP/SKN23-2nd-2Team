import time
import streamlit as st
from src.modules.one_hot_module import SPECIALTY_KO_MAP
from src.modules.notification_sms import notification_sms


# =========================
# 메시지 템플릿
# =========================
MESSAGE_TEMPLATES = {
    "예약 안내 (기본)": """[고객명]님,
[예약일시] [전문의] 진료 예약입니다.
예약 시간 10분 전까지 내원 바랍니다.
""",

    "예약 확인 요청": """[고객명]님,
[예약일시] [전문의] 진료 가능하신가요?
✅ 유지 ❌ 변경/취소 필요
회신 부탁드립니다.""",

    "노쇼 경고 (고위험)": """[고객명]님, [예약일시] [전문의] 진료 확인차 연락드립니다.
사전 연락 없이 예약을 지키지 않으실 경우, 향후 예약에 제한이 있을 수 있습니다.
변경/취소 시 미리 연락 바랍니다.""",

    "직접 작성": ""
}


# =========================
# 렌더 함수
# =========================
def render_message_sender():
    customer = st.session_state.selected_customer

    # -------------------------
    # 수신자 정보
    # -------------------------
    with st.container(border=True):
        st.markdown("**수신자 정보**")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- 고객명: **{customer['name']}**")
        with col2:
            st.write(f"- 전문의: {SPECIALTY_KO_MAP[customer['specialty']]}")

        st.write(f"- 예약일시: {customer['appointment_datetime']}")

    # -------------------------
    # 템플릿 선택
    # -------------------------
    template_name = st.selectbox("메시지 템플릿 선택", list(MESSAGE_TEMPLATES.keys()))

    # 세션 초기화
    if "message_content" not in st.session_state:
        st.session_state.message_content = ""

    # 템플릿 변경 시 자동 치환
    raw_template = MESSAGE_TEMPLATES[template_name]
    if raw_template:
        content = (
            raw_template
            .replace("[고객명]", customer["name"])
            .replace("[전문의]", SPECIALTY_KO_MAP[customer['specialty']])
            .replace("[예약일시]", str(customer["appointment_datetime"]))
        )
        st.session_state.message_content = content
    else:
        st.session_state.message_content = ""

    # -------------------------
    # 메시지 입력
    # -------------------------
    message = st.text_area(
        "문자 내용",
        value=st.session_state.message_content,
        height=160,
        max_chars=160
    )

    st.caption(f"{len(message)} / 160자", text_alignment='right')

    # 전송 버튼
    if st.button("전송", type="primary", width='stretch'):
        if not message.strip():
            st.toast("메시지 내용을 입력해주세요.", icon="❌")
        else:
            notification_sms(body = message)
            st.toast(f"{customer['name']}님에게 문자가 발송되었습니다.", icon="✅")
            st.session_state.message_content = ""
            st.session_state.open_message_modal = False
            time.sleep(0.5)
            st.rerun()