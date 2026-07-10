# Streamlit 기본 예시
import streamlit as st
st.title("면접 준비 에이전트")
user_input = st.text_input("질문을 입력하세요")
if st.button("전송"):
    st.write(f"입력한 내용: {user_input}")