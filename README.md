**Objective**
A local GenAI-powered Document Question Answering assistant that allows users
to upload a PDF and ask questions.
Runs completely locally using Ollama to ensure data privacy.


**Tech Stack**

Python 3.9+
Ollama (Local LLM)
LangChain
ChromaDB (local)
Sentence Transformers
Streamlit

**Architecture**

Load PDF
Split text into chunks (500 size, 100 overlap)
Create embeddings using open-source model
Store embeddings in ChromaDB
Retrieve top 3–5 chunks
Generate answer using Ollama

**Critical Rule**

The assistant answers ONLY from the uploaded document.
If found → answer returned

If not found →
"I cannot find the answer to that question in the provided document."

**Prerequisites**

Install Ollama and pull a model:
ollama pull mistral

**Installation**

git clone <your-repo-url>
cd document-qa-assistant

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

**Run the App**

streamlit run app.py
