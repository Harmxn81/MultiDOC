import streamlit as st
from utils import load_and_process_docs, answer_query

st.set_page_config(page_title="Multi-Document RAG Chatbot", page_icon="🤖")
st.title("📄 Multi-Document RAG Chatbot")

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF, CSV, TXT, or image files",
    accept_multiple_files=True
)

# Store vectorstore in session so we don't process again after each input
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = load_and_process_docs(uploaded_files)
    st.success("✅ Documents processed successfully!")

# User input for questions
if st.session_state.vectorstore:
    query = st.text_input("💬 Ask a question about your documents:")
    
    if query:
        with st.spinner("Generating answer..."):
            answer = answer_query(st.session_state.vectorstore, query)
        st.write("**Answer:**")
        st.write(answer)
