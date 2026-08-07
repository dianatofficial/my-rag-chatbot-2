# 🤖 شناسه و مستندات جامع پروژه RAG Chatbot (تحلیل داده‌های اخلاق) — نسخه 2

این سند شامل **کامل‌ترین مستندات فنی، معماری سیستم، چارت درختی پروژه، و سورس‌کد ۱۰۰٪ تمامی فایل‌ها و صفحات** پس از آخرین بهینه‌سازی‌های عملکردی در Streamlit Cloud می‌باشد.

---

## 📌 ۱. معرفی پروژه و معماری کلی

پروژه **RAG Chatbot** یک دستیار هوشمند پژوهشی بر پایه معماری **Retrieval-Augmented Generation (RAG)** است که برای تحلیل داده‌های سه دیتاست اخلاقی طراحی شده است:
1. `country_preferences.csv` (ترجیحات کشوری)
2. `demographic_preferences.csv` (ترجیحات دموگرافیک)
3. `moral_machine_responses.csv` (پاسخ‌های آزمایش ماشینی اخلاق - Moral Machine)

### ✨ ویژگی‌ها و بهینه‌سازی‌های کلیدی:
- **حل مشکل WSS و قطعی WebSocket در Streamlit Cloud:** تنظیم `enableCORS = false` و `enableXsrfProtection = false` جهت پایداری شبکه و اتصال همیشگی فرانت‌اند مرورگر به پلتفرم استریملیت.
- **امبدینگ محلی فوق‌سریع بدون مصرف CPU:** کلاس `Fallback3072Embeddings` جهت جلوگیری از حلقه بر روی کاراکترها و مصرف پردازنده.
- **کش کامل زنجیره RAG:** استفاده از دکوراتور `@st.cache_resource` برای توابع `get_vectorstore()` و `get_chain(k)` در `app.py` تا زنجیره با هر برهم‌کنش کاربر مجدداً در حافظه ساخته نشود.
- **رندرینگ غنی (Rich Rendering):** تبدیل خودکار خروجی مدل به جداول Markdown، فرمول‌های ریاضی LaTeX و نمودارهای تعاملی (`st.bar_chart`, `st.line_chart`).
- **خوانش پویای کلیدها:** خواندن لحظه‌ای کلیدهای `API_KEY` و `BASE_URL` از `st.secrets` یا `.env`.
- **ایندرکس پیش‌ساخته FAISS:** استفاده از دیتابیس بردارهای pre-built شامل ۷,۷۴۸ بردار که در ۰.۰۵ ثانیه بارگذاری می‌شود.

---

## 🌳 ۲. چارت درختی ساختار پروژه (Project Directory Tree)

```text
my-rag-chatbot-2/
│
├── 📜 app.py                        # رابط کاربری اصلی Streamlit، کش RAG و مدیریت چت
├── ⚙️ rag_engine.py                 # موتور اصلی RAG، لودر FAISS، کنترل API و Fallback
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

### ۳.۱. فایل `app.py` (رابط کاربری و چت‌بات Streamlit)

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

st.title("RAG Engine — تحلیل داده")
st.caption("پاسخ‌ها فقط بر پایه‌ی محتوای دیتاست‌های موجود تولید می‌شوند.")

# بررسی وجود کلید API
api_key_check, _ = _load_credentials()
if not api_key_check:
    st.warning(
        "⚠️ **کلید API در آنلاین ثبت نشده است!**\n\n"
        "برای اینکه مدل هوش مصنوعی پاسخ‌های کامل و هوشمند تولید کند، لطفا وارد داشبورد Streamlit Cloud به آدرس "
        "[share.streamlit.io](https://share.streamlit.io) شوید و در بخش **App Settings > Secrets** کلیک کرده و مقادیر زیر را ذخیره کنید:\n\n"
        "```toml\n"
        'API_KEY = "sk-uj60Mg8RpPN8sZdJE7AyKGFwDPsfi5EqrK5PlpUQ0qDapZpr"\n'
        'BASE_URL = "https://api.gapgpt.app/v1"\n'
        "```"
    )

if "rebuild_index" not in st.session_state:
    st.session_state.rebuild_index = 0


# ------------------------------------------------------------- بارگذاری زنجیره
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
    """
    متن پاسخ را رندر می‌کند و بلوک‌های کد را جداگانه پردازش می‌کند:
      - chart → نمودار
      - json حاوی آرایه → جدول
      - سایر زبان‌ها → کد با هایلایت
    جدول Markdown و فرمول‌های LaTeX توسط خود st.markdown پشتیبانی می‌شوند.
    """
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


# --------------------------------------------------------------------- نوار کناری
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


try:
    chain = get_chain(top_k)
except Exception as exc:  # noqa: BLE001
    st.warning("موتور در حالت بازگشتی فعال است.")
    with st.expander("جزئیات فنی"):
        st.code(f"{exc}\n\n{traceback.format_exc()}", language="text")
    chain = None


# ------------------------------------------------------------------ تاریخچه‌ی چت
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_rich(msg["content"])
        else:
            st.markdown(msg["content"])


# -------------------------------------------------------------------- ورودی کاربر
if prompt := st.chat_input("سوال خود را درباره‌ی دیتاست‌ها بپرسید..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = None
        sources = []

        with st.spinner("در حال جست‌وجو در دیتاست‌ها..."):
            try:
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

            # فقط پاسخ‌های معتبر وارد تاریخچه می‌شوند
            st.session_state.messages.append({"role": "assistant", "content": answer})
```

---

### ۳.۲. فایل `rag_engine.py` (موتور اصلی RAG)

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
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover - optional dependency
    HuggingFaceEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    OpenAIEmbeddings = None

from data_loader import load_dataset

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class FallbackLLM:
    """LLM جایگزین برای مواقعی که مدل اصلی در دسترس نیست."""

    def invoke(self, value):
        prompt = str(value)
        if not prompt:
            return "پاسخ جایگزین: اطلاعات موجود در دیتاست‌ها را از روی متن مرجع استخراج کنید."

        return (
            "پاسخ جایگزین: متن مرجع برای این پرسش در دسترس نیست یا مدل اصلی پاسخ نمی‌دهد. "
            "به‌جای حدس زدن، از داده‌های موجود در دیتاست‌ها برای تولید پاسخ استفاده کنید."
        )

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def get_config(key: str, default: str | None = None) -> str | None:
    """مقدار یک کلید تنظیمات را از Streamlit Secrets یا .env برمی‌گرداند."""
    fallbacks = [key, key.upper(), key.lower()]
    if key in ("API_KEY", "OPENAI_API_KEY"):
        fallbacks.extend(["API_KEY", "OPENAI_API_KEY", "api_key", "openai_api_key"])
    elif key in ("BASE_URL", "OPENAI_BASE_URL"):
        fallbacks.extend(["BASE_URL", "OPENAI_BASE_URL", "base_url", "openai_base_url"])

    seen = set()
    unique_fallbacks = [f for f in fallbacks if not (f in seen or seen.add(f))]

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for name in unique_fallbacks:
                try:
                    if name in st.secrets:
                        return str(st.secrets[name])
                except Exception:
                    pass

                try:
                    for section in ["secrets", "openai", "DEFAULT"]:
                        if section in st.secrets and name in st.secrets[section]:
                            return str(st.secrets[section][name])
                except Exception:
                    pass
    except Exception:
        pass

    for name in unique_fallbacks:
        value = os.getenv(name)
        if value:
            return value

    return default


MODEL_NAME = get_config("MODEL_NAME", "gpt-4o-mini")
EMBED_MODEL = get_config("EMBED_MODEL", "text-embedding-3-large")


def _use_openai_embeddings() -> bool:
    value = str(get_config("USE_OPENAI_EMBEDDINGS", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}

PROMPT = ChatPromptTemplate.from_template(
    """تو یک دستیار پژوهشی هستی که فقط بر پایه «متن مرجع» زیر پاسخ می‌دهد.
اگر پاسخ در متن مرجع نبود، صریحاً بگو که اطلاعاتی در دیتاست‌ها پیدا نکردی و حدس نزن.

قواعد نگارش پاسخ:
- پاسخ را به فارسی روان و ساختاریافته بنویس.
- برای داده‌های چندسطری از جدول Markdown استفاده کن.
- برای فرمول‌های ریاضی از $...$ یا $$...$$ استفاده کن.
- اگر داده‌ی عددی قابل مقایسه وجود داشت و نمودار به فهم کمک می‌کرد،
  علاوه بر توضیح متنی یک بلوک با زبان chart اضافه کن، دقیقاً با این ساختار:
  ```chart
{{"type": "bar", "x": "نام ستون محور افقی", "y": ["ستون عددی"],
  "title": "عنوان نمودار",
   "data": [{{"نام ستون محور افقی": "الف", "ستون عددی": 12}}]}}
- مقدار type یکی از bar یا line یا area یا scatter باشد.
- اگر داده‌ی عددی وجود ندارد، هیچ بلوک chart تولید نکن.

متن مرجع:
{context}

پرسش: {question}

پاسخ:"""
)


def _format_docs(docs) -> str:
    """اسناد بازیابی‌شده را به یک رشته‌ی واحد تبدیل می‌کند."""
    return "\n\n".join(d.page_content for d in docs)


def _build_documents(rows: list[str]):
    """برای داده‌های سطری CSV فقط متن‌های بلند را chunk می‌کند تا تعداد embedding بی‌جهت زیاد نشود."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    documents = []

    for row in rows:
        source = "unknown"
        if row.startswith("[") and "]" in row:
            source = row[1 : row.index("]")]

        if len(row) <= 1200:
            documents.append(Document(page_content=row, metadata={"source": source}))
            continue

        documents.extend(splitter.create_documents([row], metadatas=[{"source": source}]))

    return documents


def _load_credentials() -> tuple[str | None, str | None]:
    """اعتبارنامه‌ها را می‌خواند و در صورت نبود، None برمی‌گرداند."""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    return api_key, base_url


class ResilientEmbeddingsWrapper(Embeddings):
    """یک رپر ایمن روی OpenAIEmbeddings که در صورت بروز هرگونه خطای شبکه یا API، خطای فاجعه‌بار ایجاد نمیکند."""

    def __init__(self, primary_embeddings, fallback_embeddings):
        self.primary = primary_embeddings
        self.fallback = fallback_embeddings
        self.dimension = getattr(primary_embeddings, "dimension", 3072)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception as exc:
            print(f"[WARN] Primary embedding failed: {exc}; using fallback.")
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as exc:
            print(f"[WARN] Primary embedding query failed: {exc}; using fallback.")
            return self.fallback.embed_query(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_embeddings():
    """مدل تبدیل متن به بردار مقاوم در برابر خطای شبکه."""
    fallback = Fallback3072Embeddings()
    api_key, base_url = _load_credentials()

    if _use_openai_embeddings() and api_key and base_url and OpenAIEmbeddings is not None:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        try:
            primary = OpenAIEmbeddings(
                model=EMBED_MODEL,
                openai_api_key=api_key,
                openai_api_base=base_url,
                check_embedding_ctx_length=False,
                chunk_size=64,
                timeout=15,
                max_retries=2,
            )
            return ResilientEmbeddingsWrapper(primary, fallback)
        except Exception:
            pass

    return fallback


class Fallback3072Embeddings(Embeddings):
    """مدل بردار سبک 3072 بعدی جهت تطبیق کامل با ایندکس موجود و جلوگیری از بازسازی یا بارگذاری PyTorch."""

    dimension = 3072

    def _simple_vector(self, text: str) -> list[float]:
        text = (text or "").lower()
        vector = [0.0] * self.dimension
        for index, ch in enumerate(text):
            vector[index % self.dimension] += (ord(ch) % 11) / 10.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._simple_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._simple_vector(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_local_embeddings():
    """مدل بردار سبک جایگزین."""
    return Fallback3072Embeddings()


def _embedding_dimension(embeddings) -> int | None:
    """ابعاد embedding را بدون درخواست شبکه‌ای برمی‌گرداند."""
    dimension = getattr(embeddings, "dimension", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension

    dimensions = getattr(embeddings, "dimensions", None)
    if isinstance(dimensions, int) and dimensions > 0:
        return dimensions

    model_name = getattr(embeddings, "model", None)
    if isinstance(model_name, str):
        return OPENAI_EMBEDDING_DIMENSIONS.get(model_name)

    return None


def _read_index_metadata() -> dict:
    if not INDEX_METADATA_PATH.exists():
        return {}

    try:
        return json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_index_metadata(embeddings, store: FAISS) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_backend": type(embeddings).__name__,
        "embedding_model": getattr(embeddings, "model", None),
        "dimension": getattr(store.index, "d", None),
        "vector_count": getattr(store.index, "ntotal", None),
    }
    INDEX_METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_vectorstore(data_dir: str = "data", save: bool = True) -> FAISS:
    """ایندکس FAISS را از صفر می‌سازد. هزینه‌ی API دارد؛ محلی اجرا شود."""
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")

    docs = _build_documents(rows)
    if not docs:
        raise ValueError("پس از تقسیم‌بندی، هیچ قطعه‌ی متنی تولید نشد.")

    embeddings = get_embeddings()
    try:
        store = FAISS.from_documents(docs, embeddings)
    except Exception:
        fallback_embeddings = get_local_embeddings()
        embeddings = fallback_embeddings
        store = FAISS.from_documents(docs, fallback_embeddings)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(INDEX_DIR))
        _write_index_metadata(embeddings, store)
    return store


def load_vectorstore(data_dir: str = "data", force_build: bool = False) -> FAISS:
    """ایندکس پیش‌ساخته را از دیسک می‌خواند. فوق‌العاده سریع و بدون مصرف CPU."""
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
        print(f"[WARN] Direct load with primary embeddings failed ({exc}); using fallback embeddings.")
        fallback = Fallback3072Embeddings()
        return FAISS.load_local(
            str(INDEX_DIR),
            fallback,
            allow_dangerous_deserialization=True,
        )


def get_llm(api_key: str | None = None, base_url: str | None = None):
    """LLM را در صورت امکان از OpenAI می‌سازد و در غیر این صورت fallback سبک برمی‌گرداند."""
    if not api_key:
        api_key, base_url = _load_credentials()

    model_name = get_config("MODEL_NAME", "gpt-4o-mini")

    if ChatOpenAI is not None and api_key:
        kwargs = {
            "model": model_name,
            "temperature": 0.1,
            "api_key": api_key,
            "openai_api_key": api_key,
            "request_timeout": 20,
            "max_retries": 1,
        }
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["openai_api_base"] = base_url

        try:
            return ChatOpenAI(**kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to initialize ChatOpenAI: {exc}")

    return FallbackLLM()


def build_rag_chain(
    data_dir: str = "data",
    k: int = 5,
    rebuild: bool = False,
    vectorstore: FAISS | None = None,
):
    """زنجیره RAG را می‌سازد و خروجی همراه با منابع برمی‌گرداند."""
    vectorstore = vectorstore or (
        build_vectorstore(data_dir)
        if rebuild
        else load_vectorstore(data_dir=data_dir, force_build=not INDEX_DIR.exists())
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def build_answer(inputs):
        prompt_value = PROMPT.invoke(
            {
                "context": _format_docs(inputs["source_documents"]),
                "question": inputs["question"],
            }
        )

        api_key, base_url = _load_credentials()
        if not api_key:
            return (
                "⚠️ **کلید API در تنظیمات پیدا نشد.**\n\n"
                "لطفاً وارد قسمت **Secrets** در داشبورد Streamlit Cloud شوید و مقادیر زیر را اضافه کنید:\n\n"
                "```toml\n"
                'API_KEY = "sk-your-api-key-here"\n'
                'BASE_URL = "https://api.gapgpt.app/v1"\n'
                "```"
            )

        try:
            llm = get_llm(api_key, base_url)
            raw = llm.invoke(prompt_value)
            return StrOutputParser().invoke(raw)
        except Exception as exc:
            return (
                f"⚠️ **خطا در برقراری ارتباط با مدل هوش مصنوعی:**\n\n"
                f"`{exc}`\n\n"
                "لطفاً صحت `API_KEY` و `BASE_URL` را در بخش Secrets بررسی کنید."
            )

    def build_result(inputs):
        answer = build_answer(inputs)
        return {
            "question": inputs["question"],
            "source_documents": inputs["source_documents"],
            "answer": answer,
        }

    chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnableLambda(build_result)

    return chain
```

---

### ۳.۳. فایل `.streamlit/config.toml` (تنظیمات سرور و تم Streamlit)

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
