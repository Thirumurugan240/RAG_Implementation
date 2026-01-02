import os, sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY first")

DATA_DIR = Path(__file__).parent / "data"

docs_local = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
splits_local = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs_local)

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs_local = FAISS.from_documents(splits_local, emb)
retriever_local = vs_local.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

grade_prompt = ChatPromptTemplate.from_template(
    "You are checking if the context is enough to answer.\n"
    "Reply only YES or NO.\n\n"
    "Question: {q}\n\nContext:\n{context}\n"
)
def is_context_enough(q, docs):
    context = "\n\n".join(d.page_content[:800] for d in docs)
    verdict = llm.invoke(grade_prompt.format_messages(q=q, context=context)).content.strip().upper()
    return verdict.startswith("YES")

search = DuckDuckGoSearchResults(output_format="list")

def retrieve_web(q):
    results = search.invoke(q)
    urls = [r["link"] for r in results[:3]]
    web_docs = WebBaseLoader(urls).load()
    web_splits = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(web_docs)
    return web_splits

answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If not in context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {q}\nAnswer:"
)

q = input("Ask: ").strip()

local_hits = retriever_local.invoke(q)

if not is_context_enough(q, local_hits):
    web_hits = retrieve_web(q)
    combined_vs = FAISS.from_documents(local_hits + web_hits, emb) 
    hits = combined_vs.as_retriever(search_kwargs={"k": 4}).invoke(q)
else:
    hits = local_hits

context = "\n\n".join([f"SOURCE: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in hits])
ans = llm.invoke(answer_prompt.format_messages(q=q, context=context)).content

print("\nANSWER:\n", ans)
print("\nSOURCES USED:")
for d in hits:
    print("-", d.metadata.get("source", "unknown"))
