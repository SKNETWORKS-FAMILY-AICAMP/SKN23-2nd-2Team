import streamlit as st
from src.views.tabs.machineTap import render_machine_learning_tab
from src.views.tabs.deepTap import render_deep_learning_tab

st.set_page_config(
    page_title="모델 성능 확인",
    page_icon=":material/analytics:",
)

st.header("모델 성능 확인하기~")

tab1, tab2 = st.tabs(["머신러닝", "딥러닝"])

with tab1:
    render_machine_learning_tab()

with tab2:
    render_deep_learning_tab()