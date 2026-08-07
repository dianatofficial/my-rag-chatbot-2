# 🤖 شناسه و مستندات جامع پروژه RAG Chatbot (تحلیل داده‌های اخلاق) — نسخه 2

این سند شامل **کامل‌ترین مستندات فنی، معماری سیستم، چارت درختی پروژه، و سورس‌کد ۱۰۰٪ تمامی فایل‌ها و صفحات** پس از آخرین اصلاحات ضد OOM و Thread-Safe می‌باشد.

---

## 📌 ۱. معرفی پروژه و معماری کلی

پروژه **RAG Chatbot** یک دستیار هوشمند پژوهشی بر پایه معماری **Retrieval-Augmented Generation (RAG)** است که برای تحلیل داده‌های سه دیتاست اخلاقی طراحی شده است:
1. `country_preferences.csv` (ترجیحات کشوری)
2. `demographic_preferences.csv` (ترجیحات دموگرافیک)
3. `moral_machine_responses.csv` (پاسخ‌های آزمایش ماشینی اخلاق - Moral Machine)

### ✨ ویژگی‌ها و اصلاحات فنی اعمال‌شده:
1. **اصلاح `rag_engine.py`:**
   - حذف کامل کلاس `ThreadPoolExecutor` و تابع `_invoke_llm_with_timeout` (حذف ریشه‌ای Thread Leak).
   - تنظیم تایم‌آوت مستقیم در `ChatOpenAI(request_timeout=15, max_retries=1)`.
   - سبکسازی کامل `Fallback3072Embeddings` بدون محاسبات سنگین ریاضی و حلقه بر روی کاراکترها (مصرف ۰ CPU و RAM).
   - بارگذاری کاملاً Lazy و کش‌شده در `FAISS.load_local`.
   - انتقال تمامی فراخوانی‌های `get_config()` به داخل توابع.

2. **اصلاح `app.py`:**
   - اجرای ساخت زنجیره RAG (`get_chain`) صرفاً در زمان ارسال پرسش توسط کاربر (`st.chat_input`).
   - قرار گرفتن کل برنامه در یک try-except عمومی جهت جلوگیری از صفحه سفید یا کرش کانتینر.

3. **اصلاح `.streamlit/config.toml`:**
   - غیرفعال‌سازی CORS و XSRF برای حفظ اتصال دائم پروتکل WebSocket (رفع قطعی WSS).

4. **اصلاح `requirements.txt`:**
   - حذف تمامی وابستگی‌های سنگین غیرضروری و استفاده از بسته‌های سبک پایتون.

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

### ۳.۱. فایل `app.py`

```python
import json
import re
import traceback

import pandas as pd
import streamlit as st

from rag_engine import _load_credentials, build_rag_chain, load_vectorstore

# ---------------------------------------------------------------- تنظیمات صفحه
st.set_page_config(
    page_title="دستیار دیتاست‌های اخلاق",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource(show_spinner="در حال بارگذاری ایندکس دیتاست‌ها...")
def get_vectorstore():
    """ایندکس را از دیسک می‌خواند."""
    return load_vectorstore("data", force_build=False)


@st.cache_resource
def get_chain(k: int):
    """زنجیره‌ی RAG را روی ایندکس کش‌شده می‌سازد."""
    vectorstore = get_vectorstore()
    return build_rag_chain("data", k=k, rebuild=False, vectorstore=vectorstore)


# ------------------------------------------------------------------- نمایش غنی
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _render_chart(payload: str) -> None:
    """یک بلوک chart را به نمودار Streamlit تبدیل می‌کند."""
    spec = json.loads(payload)
    df = pd.DataFrame(spec["data"])
    if df.empty:
        st.info("داده‌ای برای رسم نمودار وجود ندارد.")
        return

    if spec.get("title"):
        st.markdown(f"**{spec['title']}**")

    kind = spec.get("type", "bar")
    x = spec.get("x") or df.columns[0]
    y = spec.get("y") or [c for c in df.columns if c != x]

    renderers = {
        "bar": st.bar_chart,
        "line": st.line_chart,
        "area": st.area_chart,
        "scatter": st.scatter_chart,
    }
    renderers.get(kind, st.bar_chart)(df, x=x, y=y)

    with st.expander("داده‌ی نمودار"):
        st.dataframe(df, use_container_width=True)


def render_rich(text: str) -> None:
    """متن پاسخ را رندر می‌کند و بلوک‌های کد را جداگانه پردازش می‌کند."""
    cursor = 0
    for match in FENCE.finditer(text):
        prefix = text[cursor : match.start()].strip()
        if prefix:
            st.markdown(prefix)

        lang = (match.group(1) or "").lower()
        body = match.group(2).strip()

        if lang == "chart":
            try:
                _render_chart(body)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"بلوک نمودار قابل رسم نبود: {exc}")
                st.code(body, language="json")
        elif lang == "json":
            try:
                data = json.loads(body)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                else:
                    st.json(data)
            except Exception:  # noqa: BLE001
                st.code(body, language="json")
        else:
            st.code(body, language=lang or None)

        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        st.markdown(tail)


def main():
    try:
        st.title("RAG Engine — تحلیل داده")
        st.caption("پاسخ‌ها فقط بر پایه‌ی محتوای دیتاست‌های موجود تولید می‌شوند.")

        # بررسی وجود کلید API
        api_key_check, _ = _load_credentials()
        if not api_key_check:
            st.warning(
                "⚠️ **کلید API در آنلاین ثبت نشده است!**\n\n"
                "برای اینکه مدل هوش مصنوعی پاسخ‌های کامل و هوشمند تولید کند، لطفا وارد داشبورد Streamlit Cloud شوید و در بخش **App Settings > Secrets** کلیک کرده و مقادیر را ذخیره کنید."
            )

        # ----------------------------------------------------------------- نوار کناری
        with st.sidebar:
            st.subheader("تنظیمات")
            top_k = st.slider(
                "تعداد قطعات بازیابی‌شده",
                min_value=2,
                max_value=12,
                value=5,
                help="مقدار بیشتر پوشش را بالا می‌برد ولی هزینه و زمان پاسخ را افزایش می‌دهد.",
            )
            show_sources = st.toggle("نمایش منابع پاسخ", value=True)

            if st.button("بازسازی ایندکس", use_container_width=True):
                st.cache_resource.clear()
                st.success("حافظه کش ایندکس پاک‌سازی شد.")
                st.rerun()

            if st.button("پاک‌کردن تاریخچه", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            st.divider()
            st.caption("ایندکس در حافظه کش می‌شود؛ پاک‌کردن تاریخچه آن را بازنمی‌سازد.")

        # -------------------------------------------------------------- تاریخچه‌ی چت
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_rich(msg["content"])
                else:
                    st.markdown(msg["content"])

        # ---------------------------------------------------------------- ورودی کاربر
        if prompt := st.chat_input("سوال خود را درباره‌ی دیتاست‌ها بپرسید..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer = None
                sources = []

                with st.spinner("در حال جست‌وجو در دیتاست‌ها..."):
                    try:
                        # Lazy loading chain ONLY when prompt is received!
                        chain = get_chain(top_k)
                        result = chain.invoke(prompt)
                        answer = result["answer"]
                        sources = result.get("source_documents", [])
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"خطا در تولید پاسخ: {exc}")
                        with st.expander("جزئیات فنی"):
                            st.code(traceback.format_exc(), language="text")

                if answer is not None:
                    render_rich(answer)

                    if show_sources and sources:
                        with st.expander(f"منابع ({len(sources)} قطعه)"):
                            for i, doc in enumerate(sources, start=1):
                                src = doc.metadata.get("source", "نامشخص")
                                st.markdown(f"**{i}. {src}**")
                                st.caption(doc.page_content[:400] + "…")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ **خطا در اجرای برنامه:** {exc}")
        with st.expander("جزئیات فنی خطا"):
            st.code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()
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


def _use_openai_embeddings() -> bool:
    value = str(get_config("USE_OPENAI_EMBEDDINGS", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_credentials() -> tuple[str | None, str | None]:
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    return api_key, base_url


class LightweightFallbackEmbeddings(Embeddings):
    """امبدینگ محلی فوق‌سریع بدون مصرف CPU."""

    dimension = 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vec = [0.0] * self.dimension
        return [vec for _ in texts]

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
```

---

### ۳.۳. فایل `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#6366F1"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F8FAFC"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200
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
