import streamlit as st
import pandas as pd
from src.modules.one_hot_module import SPECIALTY_KO_MAP, _SPECIALTY_CATS_EN

def render_edit_info_modal():
    customer = st.session_state.selected_customer_for_edit
    print(customer)

    with st.form("edit_customer_form"):
        new_name = st.text_input("이름", value=customer['name'], disabled=True)
        new_age = st.number_input("나이", value=customer['age'], min_value=0, max_value=120)
        gender_options = ["남", "여"]
        try:
            # Handles case where gender is numeric (0 or 1)
            gender_index = int(customer['gender'])
        except (ValueError, TypeError):
            # Handles case where gender is already a string "남" or "여"
            try:
                gender_index = gender_options.index(customer['gender'])
            except ValueError:
                gender_index = 0 # Failsafe default
        new_gender = st.selectbox("성별", gender_options, index=gender_index)
        specialty_options_ko = list(SPECIALTY_KO_MAP.values())
        specialty_options_en = list(SPECIALTY_KO_MAP.keys())

        # Find the initial index for the selectbox based on the English specialty key
        try:
            initial_specialty_en = customer['specialty']
            initial_index = specialty_options_en.index(initial_specialty_en)
        except (KeyError, ValueError):
            initial_index = specialty_options_en.index("unknown") # Default to 'unknown' if not found

        new_specialty_ko = st.selectbox("진료과목", specialty_options_ko, index=initial_index)
        # Convert selected Korean specialty back to English key for saving
        new_specialty_en = specialty_options_en[specialty_options_ko.index(new_specialty_ko)]

        try:
            appointment_datetime = pd.to_datetime(customer['appointment'])
        except:
            appointment_datetime = None # Set a default if parsing fails
        new_appointment = st.datetime_input("예약시간", value=appointment_datetime)

        if st.form_submit_button("저 장", width='stretch'):
            # Convert gender string back to numeric to maintain data consistency
            gender_numeric = 0 if new_gender == "남" else 1

            updated_info = {
                "id": customer['id'],
                "name": new_name,
                "age": new_age,
                "gender": gender_numeric,
                "specialty": new_specialty_en, # Changed 'department' to 'specialty' and uses English key
                "companion": customer['companion'],
                "appointment": new_appointment.strftime("%Y-%m-%d %H:%M"),
            }
            
            st.session_state.updated_customer_info = updated_info
            st.toast(f"'{new_name}'님의 정보가 업데이트되었습니다.", icon="✅")
            st.session_state.open_edit_modal = False
            st.rerun()
