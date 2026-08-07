# راهنمای کامل دیپلوی پروژه RAG Chatbot در Streamlit Community Cloud

این راهنما تمامی مراحل لازم برای انتشار چت‌بات تحلیل داده‌های اخلاق بر روی پلتفرم آنلاین **Streamlit Community Cloud** را به صورت گام به گام توضیح می‌دهد.

---

## 📋 پیش‌نیازها

1. **حساب کاربری GitHub**
2. **حساب کاربری Streamlit Community Cloud** (متصل به گیت‌هاب شما از طریق [share.streamlit.io](https://share.streamlit.io))
3. **کلید API سرویس هوش مصنوعی** (`API_KEY` و `BASE_URL`)

---

## 🚀 گام اول: مقداردهی اولیه Git و ساخت مخزن در GitHub

اگر پروژه هنوز مخزن گیت ندارد، در ترمینال پروژه دستورات زیر را اجرا کنید:

```bash
# 1. ساخت ریپوزیتوری گیت
git init

# 2. افزودن تمام فایل‌ها به استیج
git add .

# 3. ثبت اولین کامیت
git commit -m "Prepare RAG chatbot for Streamlit Cloud deployment"

# 4. تغییر نام شاخه به main
git branch -M main

# 5. اتصال به مخزن گیت‌هاب شما (آدرس مخزن خود را جایگزین کنید)
git remote add origin https://github.com/USERNAME/my-rag-chatbot.push.git

# 6. ارسال پروژه به گیت‌هاب
git push -u origin main
```

---

## ⚙️ گام دوم: ساخت برنامه در Streamlit Cloud

1. وارد سایت [share.streamlit.io](https://share.streamlit.io) شوید.
2. بر روی دکمه **Create app** (یا **New app**) کلیک کنید.
3. گزینه **I already have an app** را انتخاب کنید.
4. اطلاعات زیر را وارد کنید:
   - **Repository:** نام مخزن شما (مثلاً `USERNAME/my-rag-chatbot`)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (یک آدرس دلخواه برای برنامه‌تان انتخاب کنید)

---

## 🔑 گام سوم: تنظیم کلیدهای API (Streamlit Secrets)

قبل از کلیک روی دکمه Deploy، حتماً کلیدهای API را در استریملیت تنظیم کنید:

1. در صفحه تنظیمات برنامه (یا بخش **Advanced settings...**)، روی قسمت **Secrets** کلیک کنید.
2. محتوای زیر را کپی کرده و کلید واقعی خود را قرار دهید:

```toml
API_KEY = "sk-uj60Mg8RpPN8sZdJE7AyKGFwDPsfi5EqrK5PlpUQ0qDapZpr"
BASE_URL = "https://api.gapgpt.app/v1"
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
USE_OPENAI_EMBEDDINGS = "1"
```

3. روی دکمه **Save** کلیک کنید.

---

## 🎯 گام چهارم: انتشار (Deploy)

روی دکمه **Deploy!** کلیک کنید.

Streamlit به صورت خودکار:
- محیط پایتون را آماده می‌سازد.
- وابستگی‌های درون `requirements.txt` را نصب می‌کند.
- فایل‌های بردار و دیتابیس `faiss_index` را بارگذاری کرده و برنامه‌ی شما را آنلاین اجرا می‌نماید.

---

## 🛠 نکاتی برای نگهداری و عیب‌یابی

- **به‌روزرسانی داده‌ها:** اگر فایل‌های CSV در پوشه `data` تغییر کنند، می‌توانید از نوار کناری برنامه روی **بازسازی ایندکس** کلیک کنید یا پروژه را کامیت کرده و `git push` نمایید.
- **مشاهده لاگ‌ها:** در صورت بروز هرگونه مشکل، از گوشه سمت راست پایین صفحه Streamlit Cloud روی بخش **Manage app > View logs** کلیک کنید تا خطاهای سرور را ببینید.
