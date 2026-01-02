import os
import requests
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ---- 1) Your Flask LLM (ONLY generation replaced) ----
class FlaskLLM(LLM):
    endpoint: str = Field(default="http://127.0.0.1:5000/generate")
    timeout: int = Field(default=120)

    @property
    def _llm_type(self) -> str:
        return "flask_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        payload = {
            "prompt": prompt,
            # optional params your API may support:
            "temperature": kwargs.get("temperature", 0.2),
            "max_new_tokens": kwargs.get("max_new_tokens", 500),
        }
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        text = data.get("text") or data.get("answer") or data.get("response")
        if not text:
            raise ValueError(f"Flask API response missing text field: {data}")
        return text


# ---- 2) Build index (simple PDF-only) ----
DATA_DIR = Path(__file__).parent / "data"   # put PDFs here
docs = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

from langchain_community.embeddings import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_documents(chunks, emb)

# from langchain_openai import OpenAIEmbeddings
# vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

retriever = vs.as_retriever(search_kwargs={"k": 4})

# ---- 3) RAG chain (Retriever -> Prompt -> FlaskLLM) ----
prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context.\n"
    "If not in context, say: I don't have that information in the documents.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

llm = FlaskLLM(endpoint="http://127.0.0.1:5000/generate")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

q = input("Ask: ").strip()
print("\nANSWER:\n", chain.invoke(q))
