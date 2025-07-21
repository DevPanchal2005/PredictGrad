from tkinter.ttk import Notebook
import streamlit as st
import os
import pandas as pd
import json

st.set_page_config(page_title="Model Selection", page_icon='⚙️', layout='centered')

st.title("⚙️ Model Selection")

st.markdown("""
Welcome to the Model Selection page of **PredictGrad**.

This page provides an overview of the different models used in the project and their performance metrics.

**Key Insights:**
-  **Model :**
-  **Approach :**
-  **Performance Metrics :**
    - **Mean Absolute Error (MAE)**
            """)

def load_model_log(subject):
    df_raw = pd.read_csv(f"EDA\SubjectModels\{subject}_model\model_results_log.csv")
    model_info_df = pd.DataFrame({
        "Model": df_raw.iloc[:, 0],
        "Approach": df_raw.iloc[:, 1:-1].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1),
        "MAE": df_raw.iloc[:, -1].astype(float)
    })

    
           
def show_model(model_name, approach, mae, code="Hello, World!"):
    with st.expander(f"Model: {model_name}"):
        st.markdown(f"**Approach:** {approach}")
        st.metric("Mean Absolute Error (MAE)", mae)
        st.code(code, language="python")

de_tab, math3_tab, fsd_tab, python_tab = st.tabs(["Digital Electronics (DE)", "Math-3", "Full Stack Development (FSD)", "Python"])

with de_tab:
    st.markdown("### Digital Electronics (DE) Models")
    df = load_model_log("de")



with math3_tab:
    st.markdown("### Math-3 Model")
    load_model_log("math3")

with fsd_tab:
    st.markdown("### Full Stack Development (FSD) Model")
    load_model_log("fsd")

with python_tab:
    st.markdown("### Python Model")
    load_model_log("python")


