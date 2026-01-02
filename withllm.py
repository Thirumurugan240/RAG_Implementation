from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os, sys
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. In PowerShell: $env:OPENAI_API_KEY='your_key' then run again.")
DATA_DIR = r"C:\Users\THIRU\Desktop\offline_session"  # <-- change this

docs = (
    DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
)

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 4})
q = "what are all unifid search?"
qa = RetrievalQA.from_chain_type(ChatOpenAI(model="gpt-4.1-mini", temperature=0), retriever=retriever)
result = qa.invoke({"query": q})
print(result["result"])
