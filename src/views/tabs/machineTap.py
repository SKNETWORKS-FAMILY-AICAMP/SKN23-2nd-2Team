import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = BASE_DIR / "src" / "artifacts"

def render_machine_learning_tab():
    st.header("머신러닝 모델 성능 분석")

    with open(ARTIFACT_DIR / "xgb_metrics.json", encoding="utf-8") as f:
        xgb_metrics = json.load(f)

    with open(ARTIFACT_DIR / "rf_metrics.json", encoding="utf-8") as f:
        rf_metrics = json.load(f)

    with open(ARTIFACT_DIR / "xgb_threshold.json", encoding="utf-8") as f:
        xgb_threshold = json.load(f)["threshold"]

    with open(ARTIFACT_DIR / "rf_threshold.json", encoding="utf-8") as f:
        rf_threshold = json.load(f)["threshold"]
    
    col1, col2 = st.columns(2)
    
    with col1: 
        col1.metric("XGBoost Threshold", xgb_threshold)
        st.subheader("XGBoost Metrics")
        st.json(xgb_metrics)
    with col2:
        col2.metric("RandomForest Threshold", rf_threshold)
        st.subheader("RandomForest Metrics")
        st.json(rf_metrics)
    
    st.markdown("### 🌲 XGBoost")
    col1, col2, col3 = st.columns(3)

    with col1:
        img = ARTIFACT_DIR / "xgb_roc_curve.png"
        if img.exists():
            st.image(str(img), caption="ROC Curve", use_container_width=True)

    with col2:
        img = ARTIFACT_DIR / "xgb_pr_curve.png"
        if img.exists():
            st.image(str(img), caption="Precision-Recall Curve", use_container_width=True)

    with col3:
        img = ARTIFACT_DIR / "xgb_feature_importance.png"
        if img.exists():
            st.image(str(img), caption="Feature Importance", use_container_width=True)

    # ---- RandomForest ----
    st.markdown("### 🌲 RandomForest")

    col1, col2, col3 = st.columns(3)

    with col1:
        img = ARTIFACT_DIR / "rf_roc_curve.png"
        if img.exists():
            st.image(str(img), caption="ROC Curve", use_container_width=True)

    with col2:
        img = ARTIFACT_DIR / "rf_pr_curve.png"
        if img.exists():
            st.image(str(img), caption="Precision-Recall Curve", use_container_width=True)

    with col3:
        img = ARTIFACT_DIR / "rf_feature_importance.png"
        if img.exists():
            st.image(str(img), caption="Feature Importance", use_container_width=True)