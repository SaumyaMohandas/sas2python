import streamlit as st
from generator import generate_python_code

st.set_page_config(page_title="SAS to Python Converter", layout="wide")

st.title("🔁 SAS to Optimized Python Converter")
st.markdown("Paste any SAS code below. The app will convert it to clean, concise Python using a GenAI model.")

sas_input = st.text_area("Enter SAS Code:", height=200, placeholder="data mydata;\n   set source;\n   profit = revenue - cost;\nrun;")

if st.button("🚀 Convert to Python"):
    if sas_input.strip():
        with st.spinner("Converting..."):
            result = generate_python_code(sas_input)
        st.code(result, language="python")
    else:
        st.warning("Please enter some SAS code to convert.")
