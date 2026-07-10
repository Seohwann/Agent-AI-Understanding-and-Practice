"""
agent.py - AI 학습 플래너 (LangGraph Supervisor 패턴)
------------------------------------------------------------
구조:
    START -> supervisor ─┬─ "plan"     -> plan_agent     ─┐
                         ├─ "news"     -> news_agent     ─┼─> final -> END
                         └─ "progress" -> progress_agent ─┘

노드:
  - supervisor_node     : 요청을 분석해 담당 에이전트 선택
  - plan_agent_node     : 과목/시험일/학습시간 기반 일별 학습 계획 생성
  - news_agent_node     : NewsData API로 과목 관련 최신 학습 자료 검색
  - progress_agent_node : 오늘 학습 내용 입력받아 진도 체크 및 피드백
  - final_node          : 결과 취합 후 최종 답변 생성

[설치]
uv add langgraph langchain-openai python-dotenv requests streamlit
"""

import os
from datetime import date
from typing import TypedDict, Literal

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")  # 없으면 mock 으로 자동 전환


# ------------------------------------------------------------
# 1. LLM (gpt-4o-mini)
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


# ------------------------------------------------------------
# 2. 상태 정의 (TypedDict)
# ------------------------------------------------------------
class PlannerState(TypedDict):
    user_input: str        # 사용자 요청
    subject: str           # 과목명
    exam_date: str         # 시험 날짜 (YYYY-MM-DD)
    study_hours: float     # 하루 학습 가능 시간
    next_agent: str        # supervisor 가 선택한 에이전트
    plan_result: str       # 학습 계획 결과
    news_result: str       # 학습 자료 검색 결과
    progress_result: str   # 진도 체크 피드백
    final_answer: str      # 최종 취합 답변


AGENT_LABEL = {
    "plan": "학습 계획 에이전트",
    "news": "학습 자료 검색 에이전트",
    "progress": "진도 체크 에이전트",
}


# ------------------------------------------------------------
# 3. 뉴스 검색 (NewsData API / 키 없으면 mock)
# ------------------------------------------------------------
def fetch_news(keyword: str) -> str:
    if not NEWSDATA_API_KEY:
        return (
            f"(mock) '{keyword}' 관련 최신 자료:\n"
            f"- {keyword} 입문자를 위한 학습 로드맵이 공개되었다.\n"
            f"- {keyword} 분야 최신 트렌드를 정리한 리포트가 발표되었다.\n"
            "※ NEWSDATA_API_KEY를 .env에 넣으면 실제 뉴스가 검색됩니다."
        )

    try:
        resp = requests.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": NEWSDATA_API_KEY, "q": keyword, "language": "ko"},
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        return f"뉴스 요청 중 오류가 발생했습니다: {e}"

    if data.get("status") != "success":
        return f"뉴스 검색 실패: {data.get('results') or data}"

    articles = data.get("results", [])[:5]
    if not articles:
        return f"'{keyword}' 관련 자료를 찾지 못했습니다."

    lines = [f"'{keyword}' 관련 최신 자료:"]
    for i, art in enumerate(articles, 1):
        title = art.get("title") or "제목 없음"
        link = art.get("link") or ""
        lines.append(f"{i}. {title}\n   {link}")
    return "\n".join(lines)


def days_left(exam_date: str) -> int:
    """시험까지 남은 일수를 계산합니다."""
    try:
        y, m, d = map(int, exam_date.split("-"))
        return (date(y, m, d) - date.today()).days
    except Exception:
        return 0


# ------------------------------------------------------------
# 4. supervisor_node : 요청 분석 후 담당 에이전트 선택
# ------------------------------------------------------------
def supervisor_node(state: PlannerState) -> dict:
    print("\n[supervisor] 요청 분석 중...")

    choice = llm.invoke(
        "당신은 AI 학습 플래너 시스템의 관리자입니다.\n"
        "사용자 요청을 읽고 아래 담당자 중 하나를 골라 그 이름만 출력하세요.\n\n"
        "- plan     : 학습 계획 수립, 일정 짜기, 시간 배분을 원하는 경우\n"
        "- news     : 과목 관련 최신 자료, 뉴스, 트렌드를 찾는 경우\n"
        "- progress : 오늘 공부한 내용을 보고하고 진도 점검·피드백을 원하는 경우\n\n"
        "다른 말 없이 plan, news, progress 중 하나만 출력하세요.\n\n"
        f"[사용자 요청]\n{state['user_input']}"
    ).content.strip().lower()

    if choice not in AGENT_LABEL:
        choice = "plan"  # 안전장치

    print(f"[supervisor] '{AGENT_LABEL[choice]}'에게 작업을 전달합니다.")
    return {"next_agent": choice}


# ------------------------------------------------------------
# 5. 에이전트 노드
# ------------------------------------------------------------
def plan_agent_node(state: PlannerState) -> dict:
    """과목, 시험 날짜, 하루 학습 시간을 바탕으로 일별 학습 계획을 생성합니다."""
    print("[plan_agent] 학습 계획 생성 중...")

    remaining = days_left(state["exam_date"])
    total_hours = remaining * state["study_hours"]

    result = llm.invoke(
        "당신은 학습 코치입니다. 아래 조건으로 일별 학습 계획을 세워 주세요.\n"
        "- 시험일까지 남은 기간을 초반(개념)·중반(문제풀이)·후반(복습/모의고사)으로 배분\n"
        "- 각 구간의 기간, 목표, 하루 단위 학습 내용을 표 형태로 정리\n"
        "- 마지막에 '주의할 점' 2~3가지 제시\n\n"
        f"과목: {state['subject']}\n"
        f"시험일: {state['exam_date']} (D-{remaining})\n"
        f"하루 학습 가능 시간: {state['study_hours']}시간\n"
        f"총 확보 가능 학습 시간: 약 {total_hours:.0f}시간\n\n"
        f"[사용자 요청]\n{state['user_input']}"
    ).content

    return {"plan_result": result}


def news_agent_node(state: PlannerState) -> dict:
    """NewsData API로 과목 관련 최신 학습 자료를 검색합니다."""
    print("[news_agent] 학습 자료 검색 중...")

    # 과목명을 우선 검색어로 사용
    keyword = state["subject"] or llm.invoke(
        "다음 요청에서 검색 키워드 하나만 출력하세요. 설명 없이 단어만.\n\n"
        f"{state['user_input']}"
    ).content.strip()

    return {"news_result": fetch_news(keyword)}


def progress_agent_node(state: PlannerState) -> dict:
    """오늘 학습한 내용을 바탕으로 진도를 체크하고 피드백합니다."""
    print("[progress_agent] 진도 체크 중...")

    remaining = days_left(state["exam_date"])

    result = llm.invoke(
        "당신은 학습 코치입니다. 학습자가 오늘 공부한 내용을 보고했습니다.\n"
        "아래 형식으로 피드백해 주세요.\n"
        "1) 오늘 학습에 대한 평가 (잘한 점 먼저)\n"
        "2) 남은 기간을 고려한 진도 적정성 판단\n"
        "3) 내일 학습 권장 사항 (구체적으로)\n"
        "학습자가 의욕을 잃지 않도록 격려하는 어조를 유지하세요.\n\n"
        f"과목: {state['subject']}\n"
        f"시험일: {state['exam_date']} (D-{remaining})\n"
        f"하루 학습 가능 시간: {state['study_hours']}시간\n\n"
        f"[오늘 학습 보고]\n{state['user_input']}"
    ).content

    return {"progress_result": result}


# ------------------------------------------------------------
# 6. final_node : 결과 취합 후 최종 답변 생성
# ------------------------------------------------------------
def final_node(state: PlannerState) -> dict:
    print("[final] 결과 취합 중...")

    parts = []
    if state.get("plan_result"):
        parts.append(f"[학습 계획]\n{state['plan_result']}")
    if state.get("news_result"):
        parts.append(f"[학습 자료]\n{state['news_result']}")
    if state.get("progress_result"):
        parts.append(f"[진도 피드백]\n{state['progress_result']}")

    collected = "\n\n".join(parts) if parts else "수집된 결과가 없습니다."

    final = llm.invoke(
        "당신은 AI 학습 플래너입니다. 아래 담당자 결과를 바탕으로 "
        "학습자에게 도움이 되는 최종 답변을 한국어로 정리하세요.\n"
        "결과에 없는 내용은 지어내지 말고, 마크다운으로 읽기 좋게 구성하세요.\n\n"
        f"[사용자 요청]\n{state['user_input']}\n\n"
        f"[담당자 결과]\n{collected}"
    ).content

    return {"final_answer": final}


# ------------------------------------------------------------
# 7. 조건 엣지 및 그래프 구성
# ------------------------------------------------------------
def route_agent(state: PlannerState) -> Literal["plan", "news", "progress"]:
    return state["next_agent"]


builder = StateGraph(PlannerState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("plan", plan_agent_node)
builder.add_node("news", news_agent_node)
builder.add_node("progress", progress_agent_node)
builder.add_node("final", final_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {"plan": "plan", "news": "news", "progress": "progress"},
)

# 각 에이전트는 supervisor 로 돌아가지 않고 바로 final 로 이동
builder.add_edge("plan", "final")
builder.add_edge("news", "final")
builder.add_edge("progress", "final")
builder.add_edge("final", END)

graph = builder.compile()


# ------------------------------------------------------------
# 8. 외부 호출용 함수 (app.py 에서 import)
# ------------------------------------------------------------
def run_planner(
    user_input: str,
    subject: str,
    exam_date: str,
    study_hours: float,
) -> dict:
    """학습 플래너 그래프를 실행하고 결과를 반환합니다.

    Returns:
        {"answer": 최종 답변, "agent": 선택된 에이전트 라벨}
    """
    try:
        result = graph.invoke({
            "user_input": user_input,
            "subject": subject,
            "exam_date": exam_date,
            "study_hours": study_hours,
            "next_agent": "",
            "plan_result": "",
            "news_result": "",
            "progress_result": "",
            "final_answer": "",
        })
        return {
            "answer": result["final_answer"],
            "agent": AGENT_LABEL.get(result["next_agent"], "알 수 없음"),
        }
    except Exception as e:
        return {"answer": f"오류가 발생했습니다: {e}", "agent": "오류"}


# ------------------------------------------------------------
# 9. 터미널 단독 실행 시 테스트용
# ------------------------------------------------------------
if __name__ == "__main__":
    print("AI 학습 플래너 (종료하려면 '종료' 입력)")
    while True:
        text = input("\n> ").strip()
        if text == "종료":
            break
        if not text:
            continue
        out = run_planner(text, "파이썬", "2026-08-30", 3.0)
        print(f"\n[{out['agent']}]\n{out['answer']}")