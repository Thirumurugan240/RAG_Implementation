import os, sys, base64, mimetypes
from pathlib import Path

from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


DATA_DIR = Path(__file__).parent / "data"
client = OpenAI()

def to_data_url(img_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(img_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def image_to_text(img_path: Path) -> str:
    
    resp = client.responses.create(
        model="gpt-4.1-mini",  
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": "Extract any visible text (OCR) and summarize what the image shows. Return plain text."},
                {"type": "input_image", "image_url": to_data_url(img_path)}
            ]
        }]
    )
    return resp.output_text.strip()

docs = []
for pdf in DATA_DIR.rglob("*.pdf"):
    docs.extend(PyPDFLoader(str(pdf)).load()) 

for img in list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.jpeg")) + list(DATA_DIR.rglob("*.webp")):
    text = image_to_text(img)
    docs.append(Document(page_content=text, metadata={"source": str(img), "type": "image"}))

if not docs:
    sys.exit(f"No PDFs/images found in: {DATA_DIR}")

splits = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs)
vs = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 4})

q = input("Ask: ").strip()
hits = retriever.invoke(q)

context = "\n\n".join([f"SOURCE: {d.metadata.get('source')}\n{d.page_content}" for d in hits])

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY this context. If not found, say you don't have it.\n\n{context}\n\nQuestion: {q}"
)

ans = llm.invoke(prompt.format_messages(context=context, q=q))
print("\nANSWER:\n", ans.content)

print("\nSOURCES USED:")
for d in hits:
    print("-", d.metadata.get("source"))
