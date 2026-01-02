import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


DATA_DIR = r"C:\Users\THIRU\Desktop\offline_session"   
docs = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader).load()


chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)


vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
vector = vs.as_retriever(search_kwargs={"k": 10})  


def retrieve_bm25_rerank(query: str):
    candidates = vector.invoke(query)              
    bm25 = BM25Retriever.from_documents(candidates)
    bm25.k = 3                                     
    return bm25.invoke(query)


q = "What is the main architecture described?"
top = retrieve_bm25_rerank(q)

for i, d in enumerate(top, 1):
    print(f"\n--- RESULT {i} ---")
    print("Source:", d.metadata.get("source"))
    print(d.page_content[:500])
