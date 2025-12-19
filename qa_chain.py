from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a document-based assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, say exactly:
"I cannot find the answer to that question in the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)
 
def build_qa_chain(vectordb, llm_model, top_k):
    """
    This function connects:
    - Vector database (for retrieval)
    - Local LLM via Ollama (for answering)
    """

    # Step 1: Load the local LLM
    llm = Ollama(model=llm_model)

    # Step 2: Create a retriever from vector DB
    retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k}
    )

    # Step 3: Combine retriever + LLM into a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain

