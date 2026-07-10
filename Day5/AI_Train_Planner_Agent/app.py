"""
planner_app.py - AI 학습 플래너 Streamlit UI
------------------------------------------------------------
- 사이드바: 과목명, 시험 날짜(date_input), 하루 학습 시간(slider)
- 시험까지 남은 날짜(D-day) 자동 계산 및 표시
- 대화 기록 초기화 버튼
- st.session_state 로 대화 기록 유지
- st.spinner 로 로딩 표시
- planner_agent.py 의 run_planner() 호출

[실행]
uv run streamlit run planner_app.py
"""

from datetime import date, timedelta

import streamlit as st

from agent import run_planner


# ------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------
st.set_page_config(page_title="AI 학습 플래너", page_icon="📚")
st.title("📚 AI 학습 플래너")
st.caption("학습 계획 수립, 최신 자료 검색, 진도 체크를 도와드립니다.")


# ------------------------------------------------------------
# session_state 초기화
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------
# 사이드바: 학습 정보 입력
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 학습 정보")

    subject = st.text_input(
        "과목명",
        value="파이썬",
        placeholder="예: 파이썬, 정보처리기사",
    )

    exam_date = st.date_input(
        "시험 날짜",
        value=date.today() + timedelta(days=30),
        min_value=date.today(),
    )

    study_hours = st.slider(
        "하루 학습 가능 시간",
        min_value=0.5,
        max_value=12.0,
        value=3.0,
        step=0.5,
        format="%.1f 시간",
    )

    st.divider()

    # --- 시험까지 남은 날짜 자동 계산 ---
    remaining = (exam_date - date.today()).days
    total_hours = remaining * study_hours

    if remaining > 0:
        st.metric("시험까지", f"D-{remaining}")
    elif remaining == 0:
        st.metric("시험까지", "D-DAY")
    else:
        st.metric("시험까지", f"D+{abs(remaining)}")

    st.caption(f"확보 가능한 총 학습 시간: 약 {total_hours:.0f}시간")

    st.divider()

    # --- 대화 기록 초기화 버튼 ---
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.success("대화 기록이 초기화되었습니다.")
        st.rerun()

    st.markdown(
        "**이렇게 물어보세요**\n\n"
        "- 시험까지 학습 계획 짜줘\n"
        "- 이 과목 최신 자료 찾아줘\n"
        "- 오늘 반복문이랑 함수 공부했어"
    )


# ------------------------------------------------------------
# 이전 대화 기록 렌더링
# ------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("agent"):
            st.caption(f"담당: {msg['agent']}")
        st.markdown(msg["content"])


# ------------------------------------------------------------
# 채팅 입력 및 응답 생성
# ------------------------------------------------------------
if user_input := st.chat_input("무엇을 도와드릴까요?"):

    if not subject.strip():
        st.warning("사이드바에서 과목명을 입력해 주세요.")
        st.stop()

    # 사용자 메시지 표시 및 기록
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 에이전트 호출 (로딩 표시)
    with st.chat_message("assistant"):
        with st.spinner("학습 플래너가 분석 중입니다..."):
            result = run_planner(
                user_input=user_input,
                subject=subject.strip(),
                exam_date=exam_date.strftime("%Y-%m-%d"),
                study_hours=study_hours,
            )

        # Supervisor가 선택한 에이전트 안내
        st.caption(f"담당: {result['agent']}")
        st.markdown(result["answer"])

    # 응답 기록
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "agent": result["agent"],
    })