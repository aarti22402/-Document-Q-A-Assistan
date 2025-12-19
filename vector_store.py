# vector_store.py

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings


def create_vector_store(chunks, persist_dir, embedding_model):
    """
    Creates and stores embeddings in a local ChromaDB database.
    """

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectordb.persist()
    return vectordb


def load_vector_store(persist_dir, embedding_model):
    """
    Loads an existing ChromaDB vector store.
    """

    embeddings = OllamaEmbeddings(model=embedding_model)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
