import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = BASE_DIR / "src" / "artifacts"

def render_machine_learning_tab():
    with open(ARTIFACT_DIR / "xgb_metrics.json", encoding="utf-8") as f:
        xgb_metrics = json.load(f)

    with open(ARTIFACT_DIR / "rf_metrics.json", encoding="utf-8") as f:
        rf_metrics = json.load(f)

    with open(ARTIFACT_DIR / "xgb_threshold.json", encoding="utf-8") as f:
        xgb_threshold = json.load(f)["threshold"]

    with open(ARTIFACT_DIR / "rf_threshold.json", encoding="utf-8") as f:
        rf_threshold = json.load(f)["threshold"]

    with open(ARTIFACT_DIR / "summary_thr_0.185.json", encoding="utf-8") as f:
        summary_list = json.load(f)


    with st.expander("Logistic Regression", expanded=True):
        lr_metrics = summary_list[0]

        _, col1, col2 = st.columns([0.3, 2, 6], gap="medium", vertical_alignment="center")
        
        col1.metric(label="Threshold", value=0.185, width=200)

        with col2:
            st.caption("Metrics")
            st.json(lr_metrics)
        
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            img = ARTIFACT_DIR / "lr_roc_thr_0.185.png"

            if img.exists():
                st.image(str(img), caption="ROC Curve", width="stretch")

        with col2:
            img = ARTIFACT_DIR / "lr_pr_thr_0.185.png"

            if img.exists():
                st.image(str(img), caption="Precision-Recall Curve", width="stretch")

    with st.expander("RandomForest", expanded=True):
        _, col1, col2 = st.columns([0.3, 2, 6], gap="medium", vertical_alignment="center")

        col1.metric(label="Threshold", value=rf_threshold, width=200)

        with col2:
            st.caption("Metrics")
            st.json(rf_metrics)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            img = ARTIFACT_DIR / "rf_roc_curve.png"

            if img.exists():
                st.image(str(img), caption="ROC Curve", width="stretch")

        with col2:
            img = ARTIFACT_DIR / "rf_pr_curve.png"

            if img.exists():
                st.image(str(img), caption="Precision-Recall Curve", width="stretch")

        with st.container(width="stretch"):
            img = ARTIFACT_DIR / "rf_feature_importance.png"

            if img.exists():
                st.image(str(img), caption="Feature Importance", width="stretch")

    with st.expander("XGBoost", expanded=True):
        _, col1, col2 = st.columns([0.3, 2, 6], gap="medium", vertical_alignment="center")

        col1.metric(label="Threshold", value=xgb_threshold, width=200)

        with col2:
            st.caption("Metrics")
            st.json(xgb_metrics)
    
        st.divider()

        col1, col2= st.columns(2)

        with col1:
            img = ARTIFACT_DIR / "xgb_roc_curve.png"

            if img.exists():
                st.image(str(img), caption="ROC Curve", width="stretch")

        with col2:
            img = ARTIFACT_DIR / "xgb_pr_curve.png"

            if img.exists():
                st.image(str(img), caption="Precision-Recall Curve", width="stretch")

        with st.container(width="stretch"):
            img = ARTIFACT_DIR / "xgb_feature_importance.png"

            if img.exists():
                st.image(str(img), caption="Feature Importance", width="stretch")

    with st.expander("LightGBM ", expanded=True):
        lgbm_metrics = summary_list[1]

        _, col1, col2 = st.columns([0.3, 2, 6], gap="medium", vertical_alignment="center")

        col1.metric(label="Threshold", value=0.185, width=200)

        with col2:
            st.caption("Metrics")
            st.json(lgbm_metrics)

        st.divider()

        col1, col2= st.columns(2)

        with col1:
            img = ARTIFACT_DIR / "lgbm_roc_thr_0.185.png"

            if img.exists():
                st.image(str(img), caption="ROC Curve", width="stretch")

        with col2:
            img = ARTIFACT_DIR / "lr_pr_thr_0.185.png"

            if img.exists():
                st.image(str(img), caption="Precision-Recall Curve", width="stretch")

        with st.container(width="stretch"):
            img = ARTIFACT_DIR / "lgbm_importance_top20_thr_0.185.png"

            if img.exists():
                st.image(str(img), caption="Feature Importance", width="stretch")