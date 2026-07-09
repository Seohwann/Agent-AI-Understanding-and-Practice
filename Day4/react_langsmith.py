"""
LangGraph ReAct 패턴 에이전트
------------------------------------------------------------
구조 (순환 루프):

    START -> think ─┬─ next_action == "tool" -> act ─┐
                    │                                 │  (다시 think 로)
                    └─ next_action == "end"  -> END   │
                        ▲─────────────────────────────┘

- AgentState(TypedDict): messages, next_action, tool_result
- think_node : LLM이 상황을 판단하고 다음 행동(tool / end)을 결정  (Reasoning)
- act_node   : 뉴스 검색 Tool 실행 (임시 반환값)                    (Acting)
- 조건 엣지  : "tool" -> act_node, "end" -> 그래프 종료
- act_node 완료 후 다시 think_node 로 복귀 (반복)

[설치] uv add langgraph langchain-openai python-dotenv
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env)
#    OPENAI_API_KEY   : 필수
#    LANGSMITH_API_KEY: 있으면 추적(tracing) 활성화
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "react-agent")
    print("LangSmith 추적이 활성화되었습니다.")
else:
    print("LANGSMITH_API_KEY가 없어 추적 없이 실행합니다.")


# ------------------------------------------------------------
# 1. Tool 정의 (뉴스 검색 - 임시 반환값)
#    docstring 이 곧 LLM 이 읽는 도구 설명이 됩니다.
# ------------------------------------------------------------
@tool
def search_news(keyword: str) -> str:
    """키워드로 최신 뉴스를 검색할 때 사용합니다.
    사용자가 특정 주제의 뉴스, 기사, 최근 소식을 물어볼 때 호출하세요.

    Args:
        keyword: 검색할 키워드 (예: '인공지능', '경제')
    """
    # --- 임시(mock) 데이터. 실제 뉴스 API로 교체 가능 ---
    mock_db = {
        "인공지능": [
            "국내 기업들이 생성형 AI 도입을 확대하며 업무 자동화에 나섰다.",
            "정부가 AI 반도체 연구개발에 대규모 예산을 투입한다고 발표했다.",
        ],
        "경제": [
            "한국은행이 기준금리를 동결하며 물가 안정을 우선하겠다고 밝혔다.",
            "수출 회복세에 힘입어 무역수지가 흑자로 전환됐다.",
        ],
    }
    news = mock_db.get(keyword, [])
    if not news:
        return f"'{keyword}' 관련 뉴스를 찾지 못했습니다."
    return f"'{keyword}' 관련 뉴스:\n" + "\n".join(f"- {n}" for n in news)


tools = [search_news]
tools_by_name = {t.name: t for t in tools}


# ------------------------------------------------------------
# 2. LLM (gpt-4o-mini) + 도구 바인딩
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)


# ------------------------------------------------------------
# 3. 상태 정의 (TypedDict)
#    messages 에는 add_messages 리듀서를 붙여, 반환값이 덮어쓰기가 아닌
#    "누적(append)"이 되도록 합니다. -> 대화 맥락이 계속 쌓임
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_action: str   # "tool" 또는 "end"
    tool_result: str   # 마지막 도구 실행 결과


SYSTEM_PROMPT = (
    "당신은 뉴스를 조사해 답변하는 어시스턴트입니다. "
    "필요하면 search_news 도구로 뉴스를 검색하고, "
    "충분한 정보를 얻으면 도구를 더 부르지 말고 한국어로 최종 답변을 정리하세요."
)


# ------------------------------------------------------------
# 4. think_node (Reasoning)
#    LLM 이 상황을 보고 "도구를 쓸지 / 끝낼지"를 스스로 결정합니다.
# ------------------------------------------------------------
def think_node(state: AgentState) -> dict:
    print("\n[think_node] 다음 행동을 판단 중...")

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # 응답에 tool_calls 가 있으면 도구를 쓰겠다는 뜻
    if response.tool_calls:
        called = ", ".join(tc["name"] for tc in response.tool_calls)
        print(f"[think_node] 판단: 도구 사용 ({called})")
        next_action = "tool"
    else:
        print("[think_node] 판단: 최종 답변 생성")
        next_action = "end"

    return {"messages": [response], "next_action": next_action}


# ------------------------------------------------------------
# 5. act_node (Acting)
#    think_node 가 요청한 도구를 실제로 실행합니다.
# ------------------------------------------------------------
def act_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]  # tool_calls 를 담은 AIMessage

    tool_messages = []
    last_result = ""

    for call in last_message.tool_calls:
        name = call["name"]
        args = call["args"]
        print(f"[act_node] 도구 실행: {name}({args})")

        selected = tools_by_name[name]
        result = selected.invoke(args)
        last_result = result

        # 도구 결과는 반드시 ToolMessage 로 되돌려줘야 LLM 이 읽을 수 있음
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )

    return {"messages": tool_messages, "tool_result": last_result}


# ------------------------------------------------------------
# 6. 조건 엣지
#    next_action 값에 따라 act_node 로 가거나 그래프를 종료합니다.
# ------------------------------------------------------------
def route(state: AgentState) -> str:
    return state["next_action"]  # "tool" 또는 "end"


# ------------------------------------------------------------
# 7. 그래프 구성 및 컴파일 (순환 구조)
# ------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("think", think_node)
builder.add_node("act", act_node)

builder.add_edge(START, "think")

builder.add_conditional_edges(
    "think",
    route,
    {"tool": "act", "end": END},
)

# 핵심: act 가 끝나면 다시 think 로 돌아가 반복 (ReAct 루프)
builder.add_edge("act", "think")

graph = builder.compile()


# ------------------------------------------------------------
# 8. 실행
# ------------------------------------------------------------
def main():
    print("=" * 50)
    print(" LangGraph ReAct 에이전트 (뉴스 검색)")
    print(" (검색 가능한 키워드 예: 인공지능, 경제)")
    print("=" * 50)

    keyword = input("\n검색할 키워드를 입력하세요: ").strip()
    if not keyword:
        print("키워드가 비어 있습니다.")
        return

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=f"{keyword} 관련 최신 뉴스를 알려줘.")],
            "next_action": "",
            "tool_result": "",
        },
        # 무한 루프 방지 (think<->act 반복 상한)
        {"recursion_limit": 10},
    )

    print("\n" + "=" * 50)
    print(" [최종 답변]")
    print("=" * 50)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()