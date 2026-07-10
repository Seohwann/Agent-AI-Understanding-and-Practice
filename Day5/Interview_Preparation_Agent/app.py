"""
app.py - 면접 준비 에이전트 Streamlit UI
------------------------------------------------------------
- 사이드바: 자소서/채용공고 입력 + "정보 저장" 버튼
- 메인: 채팅 인터페이스 (st.session_state 로 대화 기록 유지)
- st.spinner 로 로딩 표시
- agent.py 의 run_interview_coach() 호출

[실행]
uv run streamlit run app.py
"""

import streamlit as st

from agent import run_interview_coach, reset_memory


# ------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------
st.set_page_config(page_title="면접 준비 에이전트", page_icon="💼")
st.title("💼 면접 준비 에이전트")
st.caption("자소서와 채용공고를 저장한 뒤, 예상 질문 생성과 답변 피드백을 받아보세요.")


# ------------------------------------------------------------
# session_state 초기화
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []      # 화면에 표시할 대화 기록
if "resume" not in st.session_state:
    st.session_state.resume = ""        # 저장된 자소서
if "job_posting" not in st.session_state:
    st.session_state.job_posting = ""   # 저장된 채용공고
if "info_saved" not in st.session_state:
    st.session_state.info_saved = False # 정보 저장 여부


# ------------------------------------------------------------
# 사이드바: 자소서 / 채용공고 입력
# ------------------------------------------------------------
with st.sidebar:
    st.header("📄 지원 정보 입력")

    resume_input = st.text_area(
        "자기소개서",
        value=st.session_state.resume,
        height=220,
        placeholder="자기소개서 전문을 붙여넣으세요.",
    )

    job_input = st.text_area(
        "채용공고",
        value=st.session_state.job_posting,
        height=220,
        placeholder="지원하려는 채용공고를 붙여넣으세요.",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("정보 저장", use_container_width=True):
            if not resume_input.strip() or not job_input.strip():
                st.warning("자소서와 채용공고를 모두 입력해 주세요.")
            else:
                st.session_state.resume = resume_input.strip()
                st.session_state.job_posting = job_input.strip()
                st.session_state.info_saved = True
                st.success("정보가 저장되었습니다.")

    with col2:
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            reset_memory()  # 에이전트 Memory 도 함께 비움
            st.success("대화가 초기화되었습니다.")

    st.divider()

    # 저장 상태 표시
    if st.session_state.info_saved:
        st.info("✅ 자소서·채용공고가 저장되어 있습니다.")
    else:
        st.warning("⚠️ 아직 정보가 저장되지 않았습니다.")

    st.markdown(
        "**이렇게 물어보세요**\n\n"
        "- 내 자소서와 공고를 분석해줘\n"
        "- 예상 면접 질문 5개 만들어줘\n"
        "- 방금 질문에 이렇게 답했어: (답변) → 피드백 부탁해"
    )


# ------------------------------------------------------------
# 이전 대화 기록 렌더링
# ------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------------------------------------
# 채팅 입력 및 응답 생성
# ------------------------------------------------------------
if user_input := st.chat_input("무엇을 도와드릴까요?"):

    # 사용자 메시지 표시 및 기록
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 저장된 자소서·공고가 있으면 프롬프트에 함께 실어 보냄
    if st.session_state.info_saved:
        agent_input = (
            f"{user_input}\n\n"
            f"[참고: 지원자 자소서]\n{st.session_state.resume}\n\n"
            f"[참고: 채용공고]\n{st.session_state.job_posting}"
        )
    else:
        agent_input = user_input

    # 에이전트 호출 (로딩 표시)
    with st.chat_message("assistant"):
        with st.spinner("면접 코치가 분석 중입니다..."):
            response = run_interview_coach(agent_input)
        st.markdown(response)

    # 응답 기록
    st.session_state.messages.append({"role": "assistant", "content": response})