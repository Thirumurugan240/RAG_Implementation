import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

log = logging.getLogger(__name__)

@dataclass
class RagResult:
    answer: str
    sources: List[str]
    used_web: bool
    latency_ms: int


class CorrectiveRAGService:
    """
    Local PDF RAG -> grade if context enough -> if NO, add web docs -> answer using ONLY context.
    """

    def __init__(
        self,
        data_dir: Path,
        k: int = 4,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_web_urls: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.data_dir = data_dir
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_web_urls = max_web_urls

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.emb = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

        self.grade_prompt = ChatPromptTemplate.from_template(
            "You are checking if the context is enough to answer.\n"
            "Reply only YES or NO.\n\n"
            "Question: {q}\n\nContext:\n{context}\n"
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            "Answer using ONLY the context. If not in context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        )

        self.search = DuckDuckGoSearchResults(output_format="list")

        # Build local index at startup
        self._vs_local = self._build_local_vectorstore()
        self._retriever_local = self._vs_local.as_retriever(search_kwargs={"k": self.k})

    def _build_local_vectorstore(self) -> FAISS:
        if not self.data_dir.exists():
            raise RuntimeError(f"DATA_DIR not found: {self.data_dir}")

        docs_local = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ).load()

        if not docs_local:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        splits_local = self.splitter.split_documents(docs_local)
        log.info("Loaded %d PDFs -> %d chunks", len(docs_local), len(splits_local))
        return FAISS.from_documents(splits_local, self.emb)

    def _is_context_enough(self, q: str, docs) -> bool:
        context = "\n\n".join(d.page_content[:800] for d in docs)
        verdict = (
            self.llm.invoke(self.grade_prompt.format_messages(q=q, context=context))
            .content.strip().upper()
        )
        return verdict.startswith("YES")

    def _retrieve_web(self, q: str) -> Tuple[List[Any], List[str]]:
        """
        Returns (web_splits, urls_used)
        """
        results = self.search.invoke(q) or []
        urls = []
        for r in results:
            link = r.get("link")
            if link and link not in urls:
                urls.append(link)
            if len(urls) >= self.max_web_urls:
                break

        if not urls:
            return [], []

        try:
            web_docs = WebBaseLoader(urls).load()
            web_splits = self.splitter.split_documents(web_docs)
            return web_splits, urls
        except Exception as e:
            # Don't fail the whole request if web fetch fails
            log.warning("Web retrieval failed: %s", e, exc_info=True)
            return [], urls

    @staticmethod
    def _format_context(docs) -> str:
        return "\n\n".join(
            f"SOURCE: {d.metadata.get('source','unknown')}\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def _extract_sources(docs) -> List[str]:
        seen = set()
        out = []
        for d in docs:
            s = d.metadata.get("source", "unknown")
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def ask(self, q: str, allow_web: bool = True) -> RagResult:
        t0 = time.time()
        q = (q or "").strip()
        if not q:
            raise ValueError("Query is empty")

        local_hits = self._retriever_local.invoke(q)

        used_web = False
        hits = local_hits

        if allow_web and not self._is_context_enough(q, local_hits):
            web_hits, _urls = self._retrieve_web(q)
            used_web = True

            # Build a small temporary index from local+web hits (keeps it simple & faithful to your script)
            combined_vs = FAISS.from_documents(list(local_hits) + list(web_hits), self.emb)
            hits = combined_vs.as_retriever(search_kwargs={"k": self.k}).invoke(q)

        context = self._format_context(hits)
        ans = self.llm.invoke(self.answer_prompt.format_messages(q=q, context=context)).content

        ms = int((time.time() - t0) * 1000)
        return RagResult(
            answer=ans,
            sources=self._extract_sources(hits),
            used_web=used_web,
            latency_ms=ms,
        )
