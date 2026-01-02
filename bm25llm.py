import os, sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR = Path(__file__).parent  
docs = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader).load()


chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)


vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
vector = vs.as_retriever(search_kwargs={"k": 10}) 


def bm25_rerank(query: str, final_k: int = 3):
    candidates = vector.invoke(query)
    bm25 = BM25Retriever.from_documents(candidates)
    bm25.k = final_k
    return bm25.invoke(query)


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context.\n"
    "If not found in context, say: I don't have that information in the documents.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

q = "What is the main architecture described in the document?"
top_docs = bm25_rerank(q, final_k=3)

context = "\n\n".join([d.page_content for d in top_docs])  
msg = prompt.format_messages(context=context, question=q)
ans = llm.invoke(msg)

print(ans.content)

print("\n--- Sources used ---")
for d in top_docs:
    print(d.metadata.get("source"), "page:", d.metadata.get("page"))
