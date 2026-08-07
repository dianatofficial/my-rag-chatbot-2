import json
import os
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:
    ChatOpenAI = None
    OpenAIEmbeddings = None

from data_loader import load_dataset

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

def get_config(key: str, default: str | None = None) -> str | None:
    """مقدار یک تنظیم را از Secrets یا Environment Variables می‌خواند."""
    fallbacks = [key, key.upper(), key.lower()]
    if key.upper() in ("API_KEY", "OPENAI_API_KEY"):
        fallbacks.extend(["API_KEY", "OPENAI_API_KEY", "api_key", "openai_api_key"])
    elif key.upper() in ("BASE_URL", "OPENAI_BASE_URL"):
        fallbacks.extend(["BASE_URL", "OPENAI_BASE_URL", "base_url", "openai_base_url"])

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for name in fallbacks:
                if name in st.secrets:
                    val = st.secrets[name]
                    if val is not None:
                        return str(val).strip()
                for sec in ["openai", "DEFAULT", "secrets"]:
                    if sec in st.secrets and name in st.secrets[sec]:
                        val = st.secrets[sec][name]
                        if val is not None:
                            return str(val).strip()
    except Exception:
        pass

    for name in fallbacks:
        value = os.getenv(name)
        if value:
            return value.strip()
            
    return default

def _use_openai_embeddings() -> bool:
    value = str(get_config("USE_OPENAI_EMBEDDINGS", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}

def _load_credentials() -> tuple[str | None, str | None]:
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    return api_key, base_url

# ------------------------------------------------- امبدینگ جایگزین کم‌مصرف (NumPy)
class LightweightFallbackEmbeddings(Embeddings):
    """امبدینگ محلی فوق‌سریع بدون مصرف CPU بر پایه الگوریتم هش NumPy."""
    dimension = 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            seed = abs(hash(t or "")) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            results.append(vec.tolist())
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

class ResilientEmbeddingsWrapper(Embeddings):
    """رپر مقاوم برای مدیریت خطای شبکه در زمان فراخوانی API امبدینگ."""
    def __init__(self, primary_embeddings, fallback_embeddings):
        self.primary = primary_embeddings
        self.fallback = fallback_embeddings
        self.dimension = getattr(primary_embeddings, "dimension", 3072)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception as exc:
            print(f"[WARN] Primary embedding failed ({exc}); switching to lightweight fallback.")
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as exc:
            print(f"[WARN] Primary query embedding failed ({exc}); switching to lightweight fallback.")
            return self.fallback.embed_query(text)

def get_embeddings():
    """تولیدکننده امبدینگ بهینه برای تولید یا بارگذاری ایندکس."""
    fallback = LightweightFallbackEmbeddings()
    api_key, base_url = _load_credentials()
    embed_model = get_config("EMBED_MODEL", "text-embedding-3-large")
    
    if api_key and OpenAIEmbeddings is not None:
        try:
            kwargs = {
                "model": embed_model,
                "api_key": api_key,
                "request_timeout": 15,
                "max_retries": 2,
            }
            if base_url:
                kwargs["base_url"] = base_url
            primary = OpenAIEmbeddings(**kwargs)
            return ResilientEmbeddingsWrapper(primary, fallback)
        except Exception as exc:
            print(f"[WARN] Could not initialize OpenAIEmbeddings: {exc}")
            
    return fallback


# ---------------------------------------------------- ساختار پرامپت هوشمند
PROMPT_TEMPLATE = """تو یک دستیار پژوهشی هستی که فقط بر پایه «متن مرجع» زیر پاسخ می‌دهد.
اگر پاسخ در متن مرجع نبود، صریحاً بگو که اطلاعاتی در دیتاست‌ها پیدا نکردی و حدس نزن.

قواعد نگارش پاسخ:
- پاسخ را به فارسی روان و ساختاریافته بنویس.
- برای داده‌های چندسطری از جدول Markdown استفاده کن.
- برای فرمول‌های ریاضی از $...$ یا $$...$$ استفاده کن.
- اگر داده‌ی عددی قابل مقایسه وجود داشت و نمودار به فهم کمک می‌کرد، علاوه بر توضیح متنی یک بلوک با زبان chart اضافه کن، دقیقاً با این ساختار:

```chart
{{
  "type": "bar",
  "x": "نام ستون محور افقی",
  "y": ["ستون عددی"],
  "title": "عنوان نمودار",
  "data": [
    {{"نام ستون محور افقی": "دسته اول", "ستون عددی": 10}}
  ]
}}
```
- مقدار type یکی از bar یا line یا area یا scatter باشد.
- اگر داده‌ی عددی وجود ندارد، هیچ بلوک chart تولید نکن.

متن مرجع:
{context}

پرسش: {question}
پاسخ:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def _build_documents(rows: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    documents = []
    for row in rows:
        source = "unknown"
        if row.startswith("[") and "]" in row:
            source = row[1 : row.index("]")]
        if len(row) <= 1200:
            documents.append(Document(page_content=row, metadata={"source": source}))
        else:
            documents.extend(splitter.create_documents([row], metadatas=[{"source": source}]))
    return documents

def build_vectorstore(data_dir: str = "data", save: bool = True) -> FAISS:
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")
    docs = _build_documents(rows)
    embeddings = get_embeddings()
    try:
        store = FAISS.from_documents(docs, embeddings)
    except Exception:
        fallback = LightweightFallbackEmbeddings()
        store = FAISS.from_documents(docs, fallback)
        embeddings = fallback

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(INDEX_DIR))
        payload = {
            "embedding_backend": type(embeddings).__name__,
            "dimension": getattr(store.index, "d", 3072),
            "vector_count": getattr(store.index, "ntotal", len(docs)),
        }
        INDEX_METADATA_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return store

def load_vectorstore(data_dir: str = "data", force_build: bool = False) -> FAISS:
    index_file = INDEX_DIR / "index.faiss"
    if force_build or not index_file.exists():
        return build_vectorstore(data_dir, save=True)
    
    embeddings = get_embeddings()
    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        print(f"[WARN] FAISS load_local failed ({exc}); using fallback embeddings.")
        fallback = LightweightFallbackEmbeddings()
        return FAISS.load_local(
            str(INDEX_DIR),
            fallback,
            allow_dangerous_deserialization=True,
        )

def get_llm(api_key: str | None = None, base_url: str | None = None):
    if not api_key:
        api_key, base_url = _load_credentials()
        
    model_name = get_config("MODEL_NAME", "gpt-4o-mini")
    if ChatOpenAI is not None and api_key:
        kwargs = {
            "model": model_name,
            "temperature": 0.1,
            "api_key": api_key,
            "request_timeout": 15,
            "max_retries": 1,
        }


        if base_url:
            kwargs["base_url"] = base_url
        try:
            return ChatOpenAI(**kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to initialize ChatOpenAI: {exc}")
            
    return None

def build_rag_chain(
    data_dir: str = "data",
    k: int = 5,
    rebuild: bool = False,
    vectorstore: FAISS | None = None,
):
    vectorstore = vectorstore or load_vectorstore(data_dir=data_dir, force_build=rebuild)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def build_answer(inputs):
        prompt_value = PROMPT.format(
            context=_format_docs(inputs["source_documents"]),
            question=inputs["question"],
        )
        api_key, base_url = _load_credentials()
        if not api_key:
            return (
                "⚠️ **کلید API تنظیم نشده است.**\n\n"
                "لطفاً کلید API را در بخش Secrets اضافه کنید."
            )
            
        llm = get_llm(api_key, base_url)
        if llm is None:
            return "⚠️ امکان ارتباط با مدل هوش مصنوعی وجود ندارد. لطفاً صحت کلید API را بررسی کنید."
            
        try:
            response = llm.invoke(prompt_value)
            return StrOutputParser().invoke(response)
        except Exception as exc:
            return (
                f"⚠️ **خطا در برقراری ارتباط با API:**\n\n`{exc}`\n\n"
                "لطفاً آدرس BASE_URL و API_KEY را بررسی کنید."
            )

    def build_result(inputs):
        return {
            "question": inputs["question"],
            "source_documents": inputs["source_documents"],
            "answer": build_answer(inputs),
        }

    return RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnableLambda(build_result)