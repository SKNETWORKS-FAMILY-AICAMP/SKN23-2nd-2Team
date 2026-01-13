import streamlit as st
import pandas as pd

def render_edit_info_modal():
    customer = st.session_state.selected_customer_for_edit

    with st.form("edit_customer_form"):
        new_name = st.text_input("이름", value=customer['name'])
        new_age = st.number_input("나이", value=customer['age'], min_value=0, max_value=120)
        new_gender = st.selectbox("성별", ["남", "여"], index=["남", "여"].index(customer['gender']))
        new_department = st.text_input("진료과목", value=customer['department'])

        try:
            appointment_datetime = pd.to_datetime(customer['appointment'])
        except:
            appointment_datetime = None # Set a default if parsing fails
        new_appointment = st.datetime_input("예약시간", value=appointment_datetime)

        if st.form_submit_button("저 장", width='stretch'):
            updated_info = {
                "id": customer['id'],
                "name": new_name,
                "age": new_age,
                "gender": new_gender,
                "department": new_department,
                "appointment": new_appointment.strftime("%Y-%m-%d %H:%M"),
            }
            
            st.session_state.updated_customer_info = updated_info
            st.toast(f"'{new_name}'님의 정보가 업데이트되었습니다.", icon="✅")
            st.session_state.open_edit_modal = False
            st.rerun()
