"""
LangGraph 취업 코치 멀티 에이전트 (Supervisor 패턴)
------------------------------------------------------------
구조:
    START -> supervisor ─┬─ "news"   -> news_agent   ─┐
                         ├─ "rag"    -> rag_agent    ─┤
                         └─ "resume" -> resume_agent ─┴─> final -> END
    (각 에이전트는 Supervisor로 돌아가지 않고 바로 final 로 이동)

추가 기능:
  1) Supervisor가 선택한 에이전트를 사용자에게 안내 메시지로 출력
  2) 최종 취업 코칭 결과를 txt 파일로 저장
  3) 임시 반환값 대신 실제 Tool 연동
     - news_agent : NewsData API (키 없으면 mock 자동 전환)
     - rag_agent  : job_posting*.txt 기반 FAISS RAG 파이프라인

[설치]
uv add langchain langchain-openai langchain-community langgraph faiss-cpu python-dotenv requests
"""

import os
import glob
from datetime import datetime
from typing import TypedDict, Literal

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. LLM & Embeddings
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ------------------------------------------------------------
# 2. [기능 3-a] 실제 RAG 파이프라인 구축
#    job_posting*.txt 를 모두 읽어 하나의 FAISS 인덱스로 만듭니다.
# ------------------------------------------------------------
DATA_PATTERN = "job_posting*.txt"


def build_vectorstore() -> FAISS | None:
    paths = sorted(glob.glob(DATA_PATTERN))
    if not paths:
        print(f"[경고] '{DATA_PATTERN}' 파일이 없어 RAG 없이 실행합니다.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks: list[Document] = []

    for path in paths:
        docs = TextLoader(path, encoding="utf-8").load()
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source"] = os.path.basename(path)
        all_chunks.extend(chunks)

    print(f"RAG 인덱싱 완료: {len(paths)}개 파일, {len(all_chunks)}개 청크\n")
    return FAISS.from_documents(all_chunks, embeddings)


print("채용공고 인덱싱 중...")
vectorstore = build_vectorstore()


# ------------------------------------------------------------
# 3. [기능 3-b] 실제 뉴스 검색 Tool (NewsData API)
#    NEWSDATA_API_KEY 가 없으면 mock 데이터로 자동 전환합니다.
# ------------------------------------------------------------
MOCK_NEWS = {
    "인공지능": [
        "국내 기업들이 생성형 AI 도입을 확대하며 업무 자동화에 나섰다.",
        "정부가 AI 반도체 연구개발에 대규모 예산을 투입한다고 발표했다.",
    ],
    "채용": [
        "IT 업계 개발자 채용이 경력직 중심으로 재편되고 있다.",
        "주요 기업들이 직무 중심 수시채용 비중을 늘리고 있다.",
    ],
}


def fetch_news(keyword: str) -> str:
    """NewsData API로 뉴스를 검색합니다. 키가 없으면 mock 데이터를 사용합니다."""
    api_key = os.getenv("NEWSDATA_API_KEY")

    if not api_key:
        news = MOCK_NEWS.get(keyword, [])
        if not news:
            # 키워드가 mock DB에 없으면 가장 가까운 항목 대신 안내
            return f"(mock) '{keyword}' 관련 뉴스를 찾지 못했습니다."
        return f"(mock) '{keyword}' 관련 뉴스:\n" + "\n".join(f"- {n}" for n in news)

    try:
        resp = requests.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": api_key, "q": keyword, "language": "ko"},
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        return f"뉴스 요청 중 오류가 발생했습니다: {e}"

    if data.get("status") != "success":
        return f"뉴스 검색 실패: {data.get('results') or data}"

    articles = data.get("results", [])[:5]
    if not articles:
        return f"'{keyword}' 관련 뉴스를 찾지 못했습니다."

    lines = [f"'{keyword}' 관련 최신 뉴스:"]
    for i, art in enumerate(articles, 1):
        lines.append(f"{i}. {art.get('title', '제목 없음')}")
    return "\n".join(lines)


# ------------------------------------------------------------
# 4. 상태 정의 (TypedDict)
# ------------------------------------------------------------
class SupervisorState(TypedDict):
    user_input: str      # 사용자 요청
    next_agent: str      # supervisor가 선택한 에이전트
    news_result: str     # 뉴스 에이전트 결과
    rag_result: str      # 채용공고 RAG 에이전트 결과
    resume_result: str   # 자소서 에이전트 결과
    final_answer: str    # 최종 취합 답변


AGENT_LABEL = {
    "news": "뉴스 검색 에이전트",
    "rag": "채용공고 분석 에이전트",
    "resume": "자소서 피드백 에이전트",
}


# ------------------------------------------------------------
# 5. supervisor_node : 요청을 분석해 담당 에이전트를 선택
# ------------------------------------------------------------
def supervisor_node(state: SupervisorState) -> dict:
    print("\n[supervisor] 요청 분석 중...")

    prompt = (
        "당신은 취업 코칭 시스템의 관리자입니다.\n"
        "사용자 요청을 읽고 아래 세 담당자 중 하나를 골라, 그 이름만 정확히 출력하세요.\n\n"
        "- news   : 채용 시장, 산업 동향 등 뉴스·최신 소식을 묻는 경우\n"
        "- rag    : 채용공고의 자격요건, 연봉, 복지, 절차 등을 묻는 경우\n"
        "- resume : 자기소개서 첨삭, 강점·개선점 피드백을 원하는 경우\n\n"
        "다른 말 없이 news, rag, resume 중 하나만 출력하세요.\n\n"
        f"[사용자 요청]\n{state['user_input']}"
    )
    choice = llm.invoke(prompt).content.strip().lower()

    # 안전장치: 예상 밖의 답이면 rag 로 기본 라우팅
    if choice not in AGENT_LABEL:
        choice = "rag"

    # [기능 1] 어떤 에이전트를 선택했는지 사용자에게 안내
    print(f"[supervisor] '{AGENT_LABEL[choice]}'에게 작업을 전달합니다.")

    return {"next_agent": choice}


# ------------------------------------------------------------
# 6. 각 에이전트 노드 (실제 Tool 연동)
# ------------------------------------------------------------
def news_agent_node(state: SupervisorState) -> dict:
    """뉴스 검색 에이전트: 요청에서 키워드를 뽑아 실제 뉴스 API 호출."""
    print("[news_agent] 뉴스 검색 중...")

    keyword = llm.invoke(
        "다음 요청에서 뉴스 검색에 쓸 핵심 키워드 하나만 출력하세요. "
        "설명 없이 단어만 출력하세요.\n\n"
        f"요청: {state['user_input']}"
    ).content.strip()

    result = fetch_news(keyword)
    return {"news_result": result}


def rag_agent_node(state: SupervisorState) -> dict:
    """채용공고 분석 에이전트: FAISS에서 관련 청크를 검색해 근거로 사용."""
    print("[rag_agent] 채용공고 검색 중...")

    if vectorstore is None:
        return {"rag_result": "채용공고 파일이 없어 분석할 수 없습니다."}

    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(state["user_input"])
    if not docs:
        return {"rag_result": "관련 채용공고 내용을 찾지 못했습니다."}

    context = "\n\n".join(
        f"[출처: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
    )
    answer = llm.invoke(
        "아래 [채용공고 발췌]만을 근거로 질문에 답하세요. "
        "발췌에 없는 내용은 추측하지 마세요.\n\n"
        f"[채용공고 발췌]\n{context}\n\n"
        f"[질문]\n{state['user_input']}"
    ).content

    return {"rag_result": answer}


def resume_agent_node(state: SupervisorState) -> dict:
    """자소서 피드백 에이전트: 요청에 담긴 자소서를 분석."""
    print("[resume_agent] 자소서 피드백 생성 중...")

    feedback = llm.invoke(
        "당신은 10년 경력의 취업 컨설턴트입니다.\n"
        "아래 요청에 담긴 자기소개서의 강점 3가지와 개선점 3가지를 "
        "구체적인 근거와 함께 제시하세요.\n"
        "자소서 본문이 없다면 자소서를 붙여 달라고 안내하세요.\n\n"
        f"[요청]\n{state['user_input']}"
    ).content

    return {"resume_result": feedback}


# ------------------------------------------------------------
# 7. final_node : 각 에이전트 결과를 취합해 최종 답변 생성
# ------------------------------------------------------------
def final_node(state: SupervisorState) -> dict:
    print("[final] 결과 취합 중...")

    parts = []
    if state.get("news_result"):
        parts.append(f"[뉴스 검색 결과]\n{state['news_result']}")
    if state.get("rag_result"):
        parts.append(f"[채용공고 분석 결과]\n{state['rag_result']}")
    if state.get("resume_result"):
        parts.append(f"[자소서 피드백]\n{state['resume_result']}")

    collected = "\n\n".join(parts) if parts else "수집된 결과가 없습니다."

    final = llm.invoke(
        "당신은 취업 코치입니다. 아래 담당자 분석 결과를 바탕으로 "
        "사용자에게 도움이 되는 최종 답변을 한국어로 정리하세요.\n"
        "결과에 없는 내용은 지어내지 마세요.\n\n"
        f"[사용자 요청]\n{state['user_input']}\n\n"
        f"[담당자 분석 결과]\n{collected}"
    ).content

    return {"final_answer": final}


# ------------------------------------------------------------
# 8. 조건 엣지: supervisor 의 선택에 따라 라우팅
# ------------------------------------------------------------
def route_agent(state: SupervisorState) -> Literal["news", "rag", "resume"]:
    return state["next_agent"]


# ------------------------------------------------------------
# 9. 그래프 구성 및 컴파일
# ------------------------------------------------------------
builder = StateGraph(SupervisorState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("news", news_agent_node)
builder.add_node("rag", rag_agent_node)
builder.add_node("resume", resume_agent_node)
builder.add_node("final", final_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {"news": "news", "rag": "rag", "resume": "resume"},
)

# 각 에이전트는 supervisor 로 돌아가지 않고 바로 final 로 이동
builder.add_edge("news", "final")
builder.add_edge("rag", "final")
builder.add_edge("resume", "final")
builder.add_edge("final", END)

graph = builder.compile()


# ------------------------------------------------------------
# 10. [기능 2] 결과를 txt 파일로 저장
# ------------------------------------------------------------
def save_result(user_input: str, agent: str, answer: str) -> str:
    filename = f"coaching_{datetime.now():%Y%m%d_%H%M%S}.txt"
    content = (
        f"[취업 코칭 결과] {datetime.now():%Y-%m-%d %H:%M}\n"
        f"담당 에이전트: {AGENT_LABEL.get(agent, agent)}\n"
        + "=" * 50 + "\n"
        f"[요청]\n{user_input}\n\n"
        f"[최종 답변]\n{answer}\n"
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(filename)


# ------------------------------------------------------------
# 11. 메인 루프 ('종료' 입력 시 종료)
# ------------------------------------------------------------
def main():
    print("=" * 55)
    print(" 취업 코치 멀티 에이전트 (Supervisor)")
    print(" 예) '요즘 채용 시장 뉴스 알려줘'      -> 뉴스")
    print("     '백엔드 공고 연봉이 얼마야?'      -> 채용공고")
    print("     '내 자소서 첨삭해줘: (자소서 내용)' -> 자소서")
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
            result = graph.invoke({
                "user_input": user_input,
                "next_agent": "",
                "news_result": "",
                "rag_result": "",
                "resume_result": "",
                "final_answer": "",
            })
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        answer = result["final_answer"]
        print("\n" + "=" * 55)
        print(" [최종 답변]")
        print("=" * 55)
        print(answer)

        # [기능 2] 결과 저장 여부 확인
        if input("\n결과를 파일로 저장할까요? (y/n) > ").strip().lower() == "y":
            path = save_result(user_input, result["next_agent"], answer)
            print(f"저장 완료 → {path}")


if __name__ == "__main__":
    main()