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
