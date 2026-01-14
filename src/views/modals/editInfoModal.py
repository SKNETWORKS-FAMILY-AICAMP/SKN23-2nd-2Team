import time
import pandas as pd
import streamlit as st
from src.modules.one_hot_module import SPECIALTY_KO_MAP, _SPECIALTY_CATS_KO, _SPECIALTY_CATS_EN

gender_options = ["남", "여"]

def render_edit_info_modal():
    customer = st.session_state.selected_customer_for_edit

    with st.form("edit_customer_form"):
        name = st.text_input("이름", value=customer['name'], disabled=True)
        new_age = st.number_input("나이", value=customer['age'], min_value=0, max_value=120)

        try:
            gender_index = int(customer['gender'])
        except (ValueError, TypeError):
            try:
                gender_index = gender_options.index(customer['gender'])
            except ValueError:
                gender_index = 0

        new_gender = st.selectbox("성별", gender_options, index=gender_index)

        try:
            initial_index = _SPECIALTY_CATS_EN.index(customer['specialty'])
        except (KeyError, ValueError):
            initial_index = _SPECIALTY_CATS_EN.index("unknown") # Default to 'unknown' if not found

        new_specialty_ko = st.selectbox("진료과목", _SPECIALTY_CATS_KO, index=initial_index)
        new_specialty_en = _SPECIALTY_CATS_EN[_SPECIALTY_CATS_KO.index(new_specialty_ko)]

        try:
            appointment_datetime = pd.to_datetime(customer['appointment_datetime'])
        except:
            appointment_datetime = None

        new_appointment = st.datetime_input("예약시간", value=appointment_datetime)

        if st.form_submit_button("저 장", type="primary", width='stretch'):
            gender_numeric = 0 if new_gender == "남" else 1

            updated_info = {
                "appointment_id": customer['appointment_id'],
                "name": name,
                "age": new_age,
                "gender": gender_numeric,
                "specialty": new_specialty_en,
                "appointment_datetime": new_appointment.strftime("%Y-%m-%d %H:%M:%S"),
            }

            st.session_state.updated_customer_info = updated_info
            st.toast(f"'{name}'님의 정보가 업데이트되었습니다.", icon="✅")
            st.session_state.open_edit_modal = False
            # time.sleep(0.2)
            # st.rerun()
