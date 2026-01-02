from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = r"C:\Users\THIRU\Desktop\offline_session" 

docs = (
    DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
)

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 4})

q = "What are all unified search?"
for d in retriever.invoke(q):
    print(d.metadata.get("source"), "\n", d.page_content[:400], "\n---\n")
