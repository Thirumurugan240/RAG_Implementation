import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

search = DuckDuckGoSearchResults(output_format="list") 

question = input("Ask: ").strip()
results = search.invoke(question)
urls = [r["link"] for r in results[:3]] 

print("\nTop URLs:")
for u in urls:
    print("-", u)

docs = WebBaseLoader(urls).load()

splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
vs = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 4})

hits = retriever.invoke(question)
context = "\n\n".join([f"SOURCE: {d.metadata.get('source')}\n{d.page_content}" for d in hits])

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If not in context, say you don't know.\n\n{context}\n\nQ: {q}\nA:"
)
ans = llm.invoke(prompt.format_messages(context=context, q=question))
print("\nANSWER:\n", ans.content)

print("\nSources used:")
for d in hits:
    print("-", d.metadata.get("source"))
