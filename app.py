import streamlit as st
import os

from ingest import load_and_split_pdf
from vector_store import create_vector_store, load_vector_store
from qa_chain import build_qa_chain
from config import *

st.set_page_config(page_title="Document Q&A Assistant")
st.title("📄 Document Q&A Assistant (Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("data/uploads", exist_ok=True)
    pdf_path = f"data/uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    with st.spinner("Processing document..."):
        chunks = load_and_split_pdf(
            pdf_path,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        vectordb = create_vector_store(
            chunks,
            "chroma_db",
            EMBEDDING_MODEL
        )

    st.success("Document indexed successfully!")

    question = st.text_input("Ask a question about the document:")

    if question:
        vectordb = load_vector_store("chroma_db", EMBEDDING_MODEL)

        qa_chain = build_qa_chain(
            vectordb,
            OLLAMA_LLM,
            TOP_K
        )

        with st.spinner("Generating answer..."):
            answer = qa_chain.run(question)

        st.subheader("Answer")
        st.write(answer)
