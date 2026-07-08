"""
MBTI 콘텐츠 추천 + 뉴스/날씨 도구 에이전트  (LangChain 1.0 버전)
------------------------------------------------------------
LangChain 1.0부터 create_tool_calling_agent / AgentExecutor 가 제거되어
langchain.agents.create_agent 로 에이전트를 만듭니다.

도구:
  1) recommend_content : MBTI 유형별 콘텐츠(영화/책/음악) 추천 (LLM 체인)
  2) search_news       : NewsData API로 최신 뉴스 검색
  3) get_weather       : 날씨 조회 (임시 반환값)
"""

import os

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent  # ← 1.0 방식


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. 공통 LLM (gpt-4o-mini)
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ------------------------------------------------------------
# 2. MBTI 추천 체인  (PromptTemplate -> LLM -> Parser)
# ------------------------------------------------------------
_list_parser = CommaSeparatedListOutputParser()

_mbti_prompt = PromptTemplate(
    template=(
        "당신은 성격에 맞는 콘텐츠를 추천하는 전문가입니다.\n"
        "MBTI 유형이 '{mbti}'인 사람에게 어울리는 {content_type} 5가지를 추천해 주세요.\n"
        "부연 설명 없이 제목만 답변하세요.\n"
        "{format_instructions}"
    ),
    input_variables=["mbti", "content_type"],
    partial_variables={
        "format_instructions": _list_parser.get_format_instructions()
    },
)

mbti_chain = _mbti_prompt | llm | _list_parser


# ------------------------------------------------------------
# 3. Tool 정의 (@tool 데코레이터)
#    docstring이 그대로 LLM이 읽는 "description"이 됩니다.
# ------------------------------------------------------------
@tool
def recommend_content(mbti: str, content_type: str) -> str:
    """MBTI 유형에 어울리는 콘텐츠를 추천할 때 사용합니다.
    사용자가 특정 MBTI(예: INFP, ESTJ)에게 맞는 영화·책·음악 추천을
    요청할 때 호출하세요.

    Args:
        mbti: MBTI 유형 (예: 'INFP')
        content_type: 콘텐츠 유형 ('영화', '책', '음악' 중 하나)
    """
    items = mbti_chain.invoke({
        "mbti": mbti.upper(),
        "content_type": content_type,
    })
    # 체인은 리스트를 반환하므로, 에이전트에 넘길 땐 문자열로 정리
    return f"[{mbti.upper()}] {content_type} 추천: " + ", ".join(items)


@tool
def search_news(query: str) -> str:
    """최신 뉴스를 검색할 때 사용합니다.
    사용자가 특정 키워드·주제에 대한 최근 뉴스, 기사, 소식을 물어볼 때 호출하세요.

    Args:
        query: 검색할 키워드 (예: '인공지능', '경제')
    """
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return "NEWSDATA_API_KEY가 .env에 설정되어 있지 않습니다."

    url = "https://newsdata.io/api/1/latest"
    params = {"apikey": api_key, "q": query, "language": "ko"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        return f"뉴스 요청 중 오류가 발생했습니다: {e}"

    if data.get("status") != "success":
        return f"뉴스 검색 실패: {data.get('results') or data}"

    articles = data.get("results", [])[:5]
    if not articles:
        return f"'{query}' 관련 뉴스를 찾지 못했습니다."

    lines = [f"'{query}' 관련 최신 뉴스:"]
    for i, art in enumerate(articles, 1):
        title = art.get("title") or "제목 없음"
        link = art.get("link") or ""
        lines.append(f"{i}. {title}\n   {link}")
    return "\n".join(lines)


@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회할 때 사용합니다.
    사용자가 날씨, 기온, 강수 여부 등을 물어볼 때 호출하세요.

    Args:
        city: 날씨를 조회할 도시 이름 (예: '서울')
    """
    # TODO: 실제 날씨 API 연동 예정. 현재는 임시(mock) 반환값.
    return f"{city}의 현재 날씨: 맑음 / 기온 23℃ / 습도 45% (임시 데이터)"


tools = [recommend_content, search_news, get_weather]


# ------------------------------------------------------------
# 4. 에이전트 구성 (LangChain 1.0: create_agent)
#    - model 에는 ChatOpenAI 인스턴스를 그대로 넘길 수 있습니다.
#    - system_prompt 로 시스템 메시지를 지정합니다.
# ------------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "당신은 사용자를 돕는 한국어 비서입니다. "
        "필요할 때 제공된 도구를 사용해 답변하세요. "
        "도구가 필요 없는 일반 대화는 그냥 대답하면 됩니다."
    ),
)


# ------------------------------------------------------------
# 5. 대화 루프 (while True, '종료' 입력 시 종료)
# ------------------------------------------------------------
def main():
    print("=" * 55)
    print(" MBTI 추천 · 뉴스 · 날씨 에이전트")
    print(" 예) 'INFP한테 어울리는 영화 추천해줘'")
    print("     'AI 관련 최신 뉴스 알려줘'")
    print("     '서울 날씨 어때?'")
    print(" (종료하려면 '종료' 입력)")
    print("=" * 55)

    while True:
        user_input = input("\n무엇을 도와드릴까요? > ").strip()

        if user_input == "종료":
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            continue

        try:
            # create_agent 로 만든 에이전트는 messages 형식으로 입력받습니다.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            # 마지막 메시지가 최종 답변입니다.
            answer = result["messages"][-1].content
            print("\n[답변]")
            print(answer)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()