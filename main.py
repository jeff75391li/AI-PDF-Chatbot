import streamlit as st
from utils import run_query


st.set_page_config(page_title="AI PDF chatbot", layout="wide", page_icon="📓")
with st.sidebar:
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    st.markdown("[Get your API key from here.](https://platform.openai.com/api-keys)")

st.title("AI PDF chatbot")
st.text("The AI chatbot will read the uploaded PDF file and answer your questions related to it.")
uploaded_pdf = st.file_uploader("Upload your PDF file here:", type=["pdf"])
st.divider()
question = st.text_area("Enter your question here:")
submitted = st.button("Ask")

if submitted and not api_key:
    st.warning("Please enter your API key!")
    st.stop()
if submitted and not uploaded_pdf:
    st.warning("Please upload your file!")
    st.stop()
if submitted and not question:
    st.warning("Please enter your question!")
    st.stop()
if submitted:
    with st.spinner("The bot is thinking..."):
        try:
            answer = run_query(uploaded_pdf.name, uploaded_pdf, question, api_key)
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.stop()
    st.write(answer)
