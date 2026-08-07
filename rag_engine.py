import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:
    ChatOpenAI = None
    OpenAIEmbeddings = None

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"

def get_config(key: str, default: str | None = None) -> str | None:
    """خواندن ایمن کلیدها در زمان اجرای تابع."""
    fallbacks = [key, key.upper(), key.lower()]
    if key in ("API_KEY", "OPENAI_API_KEY"):
        fallbacks.extend(["API_KEY", "OPENAI_API_KEY", "api_key"])
    elif key in ("BASE_URL", "OPENAI_BASE_URL"):
        fallbacks.extend(["BASE_URL", "OPENAI_BASE_URL", "base_url"])

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for name in fallbacks:
                if name in st.secrets:
                    return str(st.secrets[name])
                for section in ["secrets", "openai", "DEFAULT"]:
                    if section in st.secrets and name in st.secrets[section]:
                        return str(st.secrets[section][name])
    except Exception:
        pass

    for name in fallbacks:
        val = os.getenv(name)
        if val:
            return val
    return default

class Fallback3072Embeddings(Embeddings):
    """مدل سبک جایگزین بدون محاسبات CPUسوز."""
    dimension = 3072
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension

def get_embeddings():
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    embed_model = get_config("EMBED_MODEL", "text-embedding-3-large")
    
    if api_key and base_url and OpenAIEmbeddings is not None:
        try:
            return OpenAIEmbeddings(
                model=embed_model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                request_timeout=15,
                max_retries=1
            )
        except Exception:
            pass
    return Fallback3072Embeddings()

def load_vectorstore(data_dir: str = "data") -> FAISS:
    embeddings = get_embeddings()
    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        fallback = Fallback3072Embeddings()
        return FAISS.load_local(
            str(INDEX_DIR),
            fallback,
            allow_dangerous_deserialization=True,
        )

PROMPT = ChatPromptTemplate.from_template(
    """تو یک دستیار پژوهشی هستی که فقط بر پایه متن مرجع زیر پاسخ می‌دهی.
متن مرجع:
{context}
پرسش: {question}
پاسخ:"""
)

def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def get_llm():
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    model_name = get_config("MODEL_NAME", "gpt-4o-mini")
    
    if ChatOpenAI is not None and api_key:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=15,
            max_retries=1
        )
    return None

def build_rag_chain(k: int = 5, vectorstore: FAISS | None = None):
    vectorstore = vectorstore or load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def build_answer(inputs):
        api_key = get_config("API_KEY")
        if not api_key:
            return "⚠️ **کلید API در تنظیمات پیدا نشد.**"
        try:
            llm = get_llm()
            if not llm:
                return "⚠️ **امکان ساخت مدل چت وجود ندارد.**"
            
            prompt_value = PROMPT.invoke({
                "context": _format_docs(inputs["source_documents"]),
                "question": inputs["question"],
            })
            res = llm.invoke(prompt_value)
            return StrOutputParser().invoke(res)
        except Exception as exc:
            return f"⚠️ **خطا در برقراری ارتباط با مدل:** `{exc}`"

    return RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnableLambda(lambda inputs: {
        "question": inputs["question"],
        "source_documents": inputs["source_documents"],
        "answer": build_answer(inputs)
    })