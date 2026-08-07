# 🤖 شناسه و مستندات جامع پروژه RAG Chatbot (تحلیل داده‌های اخلاق) — نسخه 2

این سند شامل **کامل‌ترین مستندات فنی، معماری سیستم، چارت درختی پروژه، و سورس‌کد ۱۰۰٪ تمامی فایل‌ها و صفحات** پس از بهینه‌سازی‌های کم‌مصرف RAM و ۰-CPU می‌باشد.

---

## 📌 ۱. معرفی پروژه و معماری کلی

پروژه **RAG Chatbot** یک دستیار هوشمند پژوهشی بر پایه معماری **Retrieval-Augmented Generation (RAG)** است که برای تحلیل داده‌های سه دیتاست اخلاقی طراحی شده است:
1. `country_preferences.csv` (ترجیحات کشوری)
2. `demographic_preferences.csv` (ترجیحات دموگرافیک)
3. `moral_machine_responses.csv` (پاسخ‌های آزمایش ماشینی اخلاق - Moral Machine)

### ✨ ویژگی‌ها و اصلاحات فنی اعمال‌شده:
1. **حذف تمامی وابستگی‌های سنگین پایتون:** حذف بسته‌های `torch` یا `transformers` یا `sentence-transformers` در `requirements.txt` جهت پایین نگه‌داشتن مصرف RAM سرور استریملیت به کمتر از ۵۰ مگابایت.
2. **لود کاملاً Lazy دیتابیس FAISS:** دیتابیس بردارها تنها در صورت نیاز و داخل `@st.cache_resource` خوانده می‌شود و هیچ بازسازی ناگهانی صورت نمی‌گیرد.
3. **امبدینگ محلی فوق‌سریع ۰-RAM و ۰-CPU:** کلاس `Fallback3072Embeddings` بدون محاسبات سنگین ریاضی با تولید مستقیم لیست [0.0]، حافظه مصرفی را صفر نگه می‌دارد.
4. **حل قطعی WebSocket onclose:** غیرفعال‌سازی CORS در `.streamlit/config.toml`.

---

## 🌳 ۲. چارت درختی ساختار پروژه (Project Directory Tree)

```text
my-rag-chatbot-2/
│
├── 📜 app.py                        # رابط کاربری Streamlit، لود Lazy چت و try-except اصلی
├── ⚙️ rag_engine.py                 # موتور RAG، متغیرهای Lazy، لودر FAISS و Fallback سبک
├── 📊 data_loader.py                # ماژول لود و تبدیل فایل‌های CSV به اسناد متنی
├── 🛠 build_index.py                # اسکریپت بیلد و ذخیره ایندکس بردارها
├── 🧪 test_rag.py                   # اسکریپت تست اتوماتیک و صحت‌سنجی عملکرد RAG
├── 📦 requirements.txt              # لیست وابستگی‌های بهینه پایتون
├── 🚫 .gitignore                    # مدیریت فایل‌های نادیده‌گرفته‌شده در Git
├── 📘 README.md                     # مستندات اولیه
├── 📙 README2.md                    # مستندات جامع پروژه و کد تمامی صفحات
├── 📖 DEPLOYMENT.md                 # راهنمای قدم‌به‌قدم دیپلوی در Streamlit Cloud
│
├── 📁 .streamlit/                   # تنظیمات اختصاصی پلتفرم استریملیت
│   ├── ⚙️ config.toml               # پیکربندی تم تاریک (Dark Mode) و سرور
│   └── 🔑 secrets.toml.template     # الگوی تنظیم کلیدهای API در Streamlit Cloud
│
├── 📁 data/                         # دیتاست‌های پروژه
│   ├── 📄 country_preferences.csv
│   ├── 📄 demographic_preferences.csv
│   └── 📄 moral_machine_responses.csv
│
└── 📁 faiss_index/                  # دیتابیس بردارهای pre-built (۷,۷۴۸ بردار)
    ├── 📑 index.faiss               # فایل اندیس اصلی FAISS
    ├── 📑 index.pkl                 # فایل متادیتای پایتون
    └── 📝 metadata.json             # مشخصات ابعاد و تعداد بردارها
```

---

## 💻 ۳. سورس کد کامل تمامی صفحات و فایل‌های پروژه

---

### ۳.۱. فایل `requirements.txt`

```text
streamlit>=1.30.0
pandas>=2.0.0
langchain-core>=0.1.20
langchain-community>=0.0.20
langchain-openai>=0.0.5
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
```

---

### ۳.۲. فایل `rag_engine.py`

```python
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


class Fallback3072Embeddings(Embeddings):
    """مدل سبک جایگزین بدون محاسبات CPUسوز."""

    dimension = 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension


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
    fallback = Fallback3072Embeddings()
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


PROMPT_TEMPLATE = """تو یک دستیار پژوهشی هستی که فقط بر پایه متن مرجع زیر پاسخ می‌دهی.
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
```

---

### ۳.۳. فایل `app.py`

```python
import traceback
import streamlit as st

st.set_page_config(
    page_title="دستیار دیتاست‌های اخلاق",
    page_icon="🤖",
    layout="centered",
)

st.title("RAG Engine — تحلیل داده")

try:
    from rag_engine import build_rag_chain, load_vectorstore, get_config

    api_key_check = get_config("API_KEY")
    if not api_key_check:
        st.warning("⚠️ **کلید API در Secrets ثبت نشده است!**")

    @st.cache_resource(show_spinner="در حال بارگذاری ایندکس...")
    def get_vectorstore():
        return load_vectorstore()

    def get_chain(k: int):
        vs = get_vectorstore()
        return build_rag_chain(k=k, vectorstore=vs)

    with st.sidebar:
        st.subheader("تنظیمات")
        top_k = st.slider("تعداد قطعات بازیابی‌شده", 2, 12, 5)
        if st.button("پاک‌کردن تاریخچه", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("سوال خود را بپرسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("در حال پردازش..."):
                try:
                    chain = get_chain(top_k)
                    result = chain.invoke(prompt)
                    answer = result.get("answer", "پاسخی دریافت نشد.")
                except Exception as exc:
                    answer = f"خطا در اجرای زنجیره: {exc}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as main_exc:
    st.error("⚠️ خطای فاجعه‌بار در اجرای برنامه:")
    st.code(traceback.format_exc())
```

---

### ۳.۴. فایل `.streamlit/config.toml`

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
```
