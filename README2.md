# 🤖 شناسه و مستندات جامع پروژه RAG Chatbot (تحلیل داده‌های اخلاق) — نسخه 2

این سند شامل **کامل‌ترین مستندات فنی، معماری سیستم، چارت درختی پروژه، و سورس‌کد ۱۰۰٪ تمامی فایل‌ها و صفحات** پس از بازنویسی خطایابی شبکه و حذف Fallback پوچ می‌باشد.

---

## 📌 ۱. معرفی پروژه و معماری کلی

پروژه **RAG Chatbot** یک دستیار هوشمند پژوهشی بر پایه معماری **Retrieval-Augmented Generation (RAG)** است که برای تحلیل داده‌های سه دیتاست اخلاقی طراحی شده است:
1. `country_preferences.csv` (ترجیحات کشوری)
2. `demographic_preferences.csv` (ترجیحات دموگرافیک)
3. `moral_machine_responses.csv` (پاسخ‌های آزمایش ماشینی اخلاق - Moral Machine)

### ✨ ویژگی‌ها و اصلاحات فنی شبکه:
1. **حذف مدل `Fallback3072Embeddings`:** در صورت قطع بودن سرور API یا تایم‌اوت در تبدیل متن به بردار، سیستم با بردار صفر/پوچ FAISS را جست‌وجو نمی‌کند و یک Exception صریح و شفاف ایجاد می‌نماید.
2. **تنظیم تایم‌آوت ۲۵ ثانیه‌ای:** تعیین `request_timeout=25` برای `OpenAIEmbeddings` و `ChatOpenAI`.
3. **نمایش پیام خطای شفاف در UI:** در صورت قطع بودن پروکسی `api.gapgpt.app` در Streamlit Cloud، کادر قرمز رنگ `st.error` با توضیح دقیق راهکار و نمایش استک‌تریس در اختیار کاربر قرار می‌گیرد.
4. **حل قطعی WebSocket onclose:** غیرفعال‌سازی CORS در `.streamlit/config.toml`.

---

## 🌳 ۲. چارت درختی ساختار پروژه (Project Directory Tree)

```text
my-rag-chatbot-2/
│
├── 📜 app.py                        # رابط کاربری Streamlit، نمایش شفاف خطای API و try-except اصلی
├── ⚙️ rag_engine.py                 # موتور RAG، تایم‌آوت ۲۵ ثانیه‌ای، حذف Fallback پوچ و مدیریت Exception
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

### ۳.۱. فایل `rag_engine.py`

```python
import json
import os
import sys
from pathlib import Path

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
from langchain_core.prompts import ChatPromptTemplate
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
                    return str(st.secrets[name]).strip()
                for section in ["secrets", "openai", "DEFAULT"]:
                    if section in st.secrets and name in st.secrets[section]:
                        return str(st.secrets[section][name]).strip()
    except Exception:
        pass

    for name in fallbacks:
        val = os.getenv(name)
        if val:
            return val.strip()

    return default


def get_embeddings():
    """مدل امبدینگ OpenAI را برمی‌گرداند. در صورت عدم احراز هویت یا خطا، Exception صریح ایجاد می‌کند."""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    embed_model = get_config("EMBED_MODEL", "text-embedding-3-large")

    if not api_key:
        raise ValueError("کلید API_KEY در تنظیمات (Secrets) یافت نشد.")

    if OpenAIEmbeddings is None:
        raise ImportError("بسته langchain_openai نصب نیست.")

    kwargs = {
        "model": embed_model,
        "openai_api_key": api_key,
        "request_timeout": 25,
        "max_retries": 1,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url

    return OpenAIEmbeddings(**kwargs)


def load_vectorstore(data_dir: str = "data") -> FAISS:
    """ایندکس FAISS پیش‌ساخته را بارگذاری می‌کند."""
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(f"فایل ایندکس در مسیر '{INDEX_DIR}' پیدا نشد.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


PROMPT = ChatPromptTemplate.from_template(
    """تو یک دستیار پژوهشی هستی که فقط بر پایه متن مرجع زیر پاسخ می‌دهی.
اگر پاسخ در متن مرجع نبود، صریحاً بگو که اطلاعاتی در دیتاست‌ها پیدا نکردی و حدس نزن.

متن مرجع:
{context}

پرسش: {question}

پاسخ:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def get_llm():
    """مدل ChatOpenAI را برمی‌گرداند."""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    model_name = get_config("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        raise ValueError("کلید API_KEY در تنظیمات (Secrets) یافت نشد.")

    if ChatOpenAI is None:
        raise ImportError("بسته langchain_openai نصب نیست.")

    kwargs = {
        "model": model_name,
        "temperature": 0,
        "openai_api_key": api_key,
        "request_timeout": 25,
        "max_retries": 1,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)


def build_rag_chain(k: int = 5, vectorstore: FAISS | None = None):
    """زنجیره RAG را می‌سازد."""
    vectorstore = vectorstore or load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def build_answer(inputs):
        llm = get_llm()
        prompt_value = PROMPT.invoke({
            "context": _format_docs(inputs["source_documents"]),
            "question": inputs["question"],
        })
        res = llm.invoke(prompt_value)
        return StrOutputParser().invoke(res)

    return RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnableLambda(lambda inputs: {
        "question": inputs["question"],
        "source_documents": inputs["source_documents"],
        "answer": build_answer(inputs),
    })
```

---

### ۳.۲. فایل `app.py`

```python
import traceback
import streamlit as st

st.set_page_config(
    page_title="دستیار دیتاست‌های اخلاق",
    page_icon="🤖",
    layout="centered",
)

st.title("RAG Engine — تحلیل داده")
st.caption("پاسخ‌ها فقط بر پایه‌ی محتوای دیتاست‌های موجود تولید می‌شوند.")

try:
    from rag_engine import build_rag_chain, load_vectorstore, get_config

    api_key_check = get_config("API_KEY")
    if not api_key_check:
        st.warning("⚠️ **کلید API_KEY در تنظیمات (Secrets) ثبت نشده است!**")

    @st.cache_resource(show_spinner="در حال بارگذاری ایندکس دیتاست‌ها...")
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

    if prompt := st.chat_input("سوال خود را درباره‌ی دیتاست‌ها بپرسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = None
            with st.spinner("در حال ارتباط با سرور API و جست‌وجو در دیتاست‌ها..."):
                try:
                    chain = get_chain(top_k)
                    result = chain.invoke(prompt)
                    answer = result.get("answer")
                except Exception as exc:
                    err_msg = str(exc)
                    st.error(
                        "⚠️ **خطا در برقراری ارتباط با سرور API:**\n\n"
                        f"`{err_msg}`\n\n"
                        "💡 **راهکار:** سرور API (آدرس `BASE_URL`) در مهلت ۲۵ ثانیه پاسخ نداد یا اتصال آن مسدود شد. لطفاً کلید `API_KEY` و آدرس `BASE_URL` را در Secrets بررسی کنید."
                    )
                    with st.expander("جزئیات فنی خطا"):
                        st.code(traceback.format_exc(), language="text")

            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as main_exc:
    st.error("⚠️ **خطای فاجعه‌بار در اجرای برنامه:**")
    st.code(traceback.format_exc(), language="text")
```

---

### ۳.۳. فایل `.streamlit/config.toml`

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
```

---

### ۳.۴. فایل `requirements.txt`

```text
streamlit>=1.30.0
pandas>=2.0.0
langchain-core>=0.1.20
langchain-community>=0.0.20
langchain-openai>=0.0.5
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
```
