import streamlit as st
st.set_page_config(page_title="Credits", layout="wide", page_icon='📜')

# CSS for custom styling
st.html("<style> ::selection { color: #FF7300;} </style>")

st.title("Credits")

with st.container(border=True):
    "## Made By :"
    "### **Dev Panchal**"
    st.link_button("LinkedIn Profile", "https://www.linkedin.com/in/dev-panchal-connect/", icon="🔗", use_container_width=True)
    st.link_button("GitHub Profile", "https://github.com/devpanchal2005", icon="🐙", use_container_width=True)

with st.container(border=True):
    """
    ## 🛠️ Technologies Used
    - 📌 **Programming & Libraries :** Python, Streamlit, NumPy, Pandas, Plotly, Scikit-Learn, SciPy, PIL, Matplotlib, Seaborn, SHAP, LightGBM, CatBoost, XGBoost, StatsModels, Optuna, Boruta
    - 💻 **IDE & Development :** VS Code, Jupyter Notebook
    - 🌍 **Version Control :** GitHub (Project Repository)
    - 🤖 **Documentation Assistance :** ChatGPT (Generating Documentation)
    - 📦 **Dependency Management :** pip, requirements.txt
    """
