"""
LangGraph 뉴스 검색·요약 그래프
------------------------------------------------------------
구조:
    START -> search_node -> (조건 분기)
                              ├─ 뉴스 있음 -> summarize_node -> end_node -> END
                              └─ 뉴스 없음 ------------------> end_node -> END

- AgentState(TypedDict): keyword, news, summary, is_done
- search_node    : 키워드로 뉴스 검색 (임시 반환값)
- summarize_node : 검색된 뉴스를 gpt-4o-mini로 요약
- end_node       : 작업 완료 표시 (is_done=True)

[설치] uv add langgraph langchain-openai python-dotenv
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env 에서 OPENAI_API_KEY)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. LLM (gpt-4o-mini)
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ------------------------------------------------------------
# 2. 상태 정의 (TypedDict)
#    그래프의 노드들이 공유하며 읽고 쓰는 데이터 구조입니다.
# ------------------------------------------------------------
class AgentState(TypedDict):
    keyword: str        # 검색 키워드
    news: list[str]     # 검색된 뉴스 목록
    summary: str        # 요약 결과
    is_done: bool       # 작업 완료 여부


# ------------------------------------------------------------
# 3. 노드 구현
#    각 노드는 state를 받아, 변경할 필드만 dict로 반환합니다.
# ------------------------------------------------------------
def search_node(state: AgentState) -> dict:
    """키워드로 뉴스를 검색합니다. (임시 반환값)"""
    keyword = state["keyword"]
    print(f"\n[search_node] '{keyword}' 뉴스 검색 중...")

    # --- 임시(mock) 데이터. 실제로는 뉴스 API를 호출하면 됩니다. ---
    mock_db = {
        "인공지능": [
            "국내 기업들이 생성형 AI 도입을 확대하며 업무 자동화에 나섰다.",
            "정부가 AI 반도체 연구개발에 대규모 예산을 투입한다고 발표했다.",
            "AI 저작권 논란이 이어지며 관련 법제 정비 논의가 활발하다.",
        ],
        "경제": [
            "한국은행이 기준금리를 동결하며 물가 안정을 우선하겠다고 밝혔다.",
            "수출 회복세에 힘입어 무역수지가 흑자로 전환됐다.",
        ],
    }
    news = mock_db.get(keyword, [])  # 없는 키워드면 빈 리스트

    print(f"[search_node] {len(news)}건 검색됨")
    return {"news": news}


def summarize_node(state: AgentState) -> dict:
    """검색된 뉴스를 LLM으로 요약합니다."""
    print("[summarize_node] 요약 중...")

    articles = "\n".join(f"- {n}" for n in state["news"])
    prompt = (
        f"다음은 '{state['keyword']}' 관련 뉴스입니다.\n"
        "핵심 내용을 3줄 이내로 요약해 주세요.\n\n"
        f"{articles}"
    )
    summary = llm.invoke(prompt).content
    return {"summary": summary}


def end_node(state: AgentState) -> dict:
    """작업 완료를 표시합니다."""
    print("[end_node] 작업 완료")

    # 검색 결과가 없어 요약을 건너뛴 경우를 대비
    summary = state.get("summary") or f"'{state['keyword']}' 관련 뉴스를 찾지 못했습니다."
    return {"summary": summary, "is_done": True}


# ------------------------------------------------------------
# 4. 조건 분기 함수
#    반환한 문자열이 다음에 실행할 노드 이름으로 매핑됩니다.
# ------------------------------------------------------------
def route_after_search(state: AgentState) -> str:
    """뉴스가 있으면 요약으로, 없으면 바로 종료 노드로 보냅니다."""
    return "summarize" if state["news"] else "end"


# ------------------------------------------------------------
# 5. 그래프 구성 및 컴파일
# ------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("search", search_node)
builder.add_node("summarize", summarize_node)
builder.add_node("end", end_node)

builder.add_edge(START, "search")

# search 이후 조건 분기: route_after_search 의 반환값 -> 노드 이름
builder.add_conditional_edges(
    "search",
    route_after_search,
    {"summarize": "summarize", "end": "end"},
)

builder.add_edge("summarize", "end")
builder.add_edge("end", END)

graph = builder.compile()


# ------------------------------------------------------------
# 6. 실행
# ------------------------------------------------------------
def main():
    print("=" * 50)
    print(" LangGraph 뉴스 검색·요약")
    print(" (검색 가능한 키워드 예: 인공지능, 경제)")
    print("=" * 50)

    keyword = input("\n검색할 키워드를 입력하세요: ").strip()
    if not keyword:
        print("키워드가 비어 있습니다.")
        return

    # 초기 상태로 그래프 실행
    result = graph.invoke({
        "keyword": keyword,
        "news": [],
        "summary": "",
        "is_done": False,
    })

    print("\n" + "=" * 50)
    print(" [요약 결과]")
    print("=" * 50)
    print(result["summary"])
    print(f"\n완료 여부(is_done): {result['is_done']}")


if __name__ == "__main__":
    main()