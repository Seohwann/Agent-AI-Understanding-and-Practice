"""
LangGraph 채용공고 RAG 에이전트 (확장판)
------------------------------------------------------------
추가된 기능:
  1) 여러 개의 txt 파일을 하나의 FAISS 벡터 DB에 통합 저장 (출처 메타데이터 부착)
  2) 검색할 chunk 개수(k)를 사용자가 입력/변경 가능
  3) RAG 검색 결과를 근거로 자소서를 개선하는 기능 (improve 노드)

그래프 구조:
    START -> retrieve ─┬─ mode == "qa"      -> generate -> END
                       └─ mode == "improve" -> improve  -> END

메뉴:
  1) 질문하기 (RAG Q&A)
  2) 자소서 입력
  3) 자소서 개선 (RAG 검색 결과 기반)
  4) 검색 chunk 개수(k) 변경
  5) 종료

[설치]
uv add langchain langchain-openai langchain-community langgraph faiss-cpu python-dotenv
"""

import os
import glob
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env 에서 OPENAI_API_KEY)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. LLM & Embeddings
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ------------------------------------------------------------
# 2. [기능 1] 여러 txt 파일 -> 하나의 FAISS 벡터 DB
#    각 청크에 source(파일명) 메타데이터를 넣어, 어느 공고에서 나온
#    내용인지 답변에서 구분할 수 있게 합니다.
# ------------------------------------------------------------
DATA_PATTERN = "job_posting*.txt"  # job_posting.txt, job_posting_data.txt ...


def build_vectorstore() -> FAISS:
    file_paths = sorted(glob.glob(DATA_PATTERN))
    if not file_paths:
        raise FileNotFoundError(
            f"'{DATA_PATTERN}' 패턴에 해당하는 파일이 없습니다. "
            "채용공고 txt 파일을 같은 폴더에 준비해 주세요."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks: list[Document] = []
    for path in file_paths:
        docs = loader_docs = TextLoader(path, encoding="utf-8").load()
        chunks = splitter.split_documents(docs)

        # 출처 메타데이터 부착 (파일명 기준)
        filename = os.path.basename(path)
        for c in chunks:
            c.metadata["source"] = filename

        all_chunks.extend(chunks)
        print(f"  - {filename}: {len(chunks)}개 청크")

    print(f"총 {len(file_paths)}개 파일, {len(all_chunks)}개 청크")

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    print("벡터 저장소(FAISS) 생성 완료\n")
    return vectorstore


print("문서 로드 및 인덱싱 중...")
vectorstore = build_vectorstore()


# ------------------------------------------------------------
# 3. 상태 정의 (TypedDict)
# ------------------------------------------------------------
class RAGState(TypedDict):
    question: str                   # 사용자 질문 (또는 개선 시 검색 쿼리)
    retrieved_docs: list[Document]  # 검색된 문서 청크
    answer: str                     # 최종 결과 (답변 또는 개선된 자소서)
    top_k: int                      # [기능 2] 검색할 chunk 개수
    mode: str                       # "qa" 또는 "improve"
    resume: str                     # [기능 3] 개선 대상 자소서


def format_context(docs: list[Document]) -> str:
    """검색된 청크를 출처와 함께 문자열로 정리합니다."""
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    )


# ------------------------------------------------------------
# 4. 노드 구현
# ------------------------------------------------------------
def retrieve_node(state: RAGState) -> dict:
    """질문과 관련된 문서를 벡터 DB에서 검색합니다. (k는 상태에서 결정)"""
    k = state["top_k"]
    print(f"[retrieve_node] 관련 문서 검색 중... (k={k})")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(state["question"])

    sources = {d.metadata.get("source") for d in docs}
    print(f"[retrieve_node] {len(docs)}개 청크 검색됨 (출처: {', '.join(sources)})")
    return {"retrieved_docs": docs}


def generate_node(state: RAGState) -> dict:
    """검색된 문서를 근거로 답변을 생성합니다. (Q&A 모드)"""
    print("[generate_node] 답변 생성 중...")

    docs = state["retrieved_docs"]
    if not docs:
        return {"answer": "관련 내용을 채용공고에서 찾지 못했습니다."}

    prompt = (
        "당신은 채용공고 내용을 안내하는 어시스턴트입니다.\n"
        "아래 [채용공고 발췌]만을 근거로 질문에 답하세요.\n"
        "발췌에 없는 내용은 추측하지 말고, 공고에서 확인할 수 없다고 답하세요.\n"
        "여러 공고의 내용이 섞여 있다면 어느 공고의 내용인지 함께 밝히세요.\n\n"
        f"[채용공고 발췌]\n{format_context(docs)}\n\n"
        f"[질문]\n{state['question']}"
    )
    return {"answer": llm.invoke(prompt).content}


def improve_node(state: RAGState) -> dict:
    """[기능 3] 검색된 공고 내용을 근거로 자소서를 개선합니다."""
    print("[improve_node] 자소서 개선 중...")

    docs = state["retrieved_docs"]
    if not docs:
        return {"answer": "관련 공고 내용을 찾지 못해 개선할 수 없습니다."}

    prompt = (
        "당신은 10년 경력의 취업 컨설턴트입니다.\n"
        "아래 [채용공고 발췌]에서 핵심 역량과 키워드를 파악한 뒤, "
        "[원본 자소서]를 그 공고에 맞게 개선해 주세요.\n\n"
        "출력 형식:\n"
        "1) 공고에서 추출한 핵심 키워드\n"
        "2) 원본 자소서의 개선점 3가지\n"
        "3) 개선된 자소서 전문\n\n"
        "주의: 원본에 없는 경력이나 사실을 지어내지 마세요.\n\n"
        f"[채용공고 발췌]\n{format_context(docs)}\n\n"
        f"[원본 자소서]\n{state['resume']}"
    )
    return {"answer": llm.invoke(prompt).content}


# ------------------------------------------------------------
# 5. 조건 분기: mode 에 따라 generate 또는 improve 로 이동
# ------------------------------------------------------------
def route_by_mode(state: RAGState) -> str:
    return state["mode"]  # "qa" 또는 "improve"


# ------------------------------------------------------------
# 6. 그래프 구성 및 컴파일
# ------------------------------------------------------------
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("improve", improve_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve",
    route_by_mode,
    {"qa": "generate", "improve": "improve"},
)
builder.add_edge("generate", END)
builder.add_edge("improve", END)

graph = builder.compile()


# ------------------------------------------------------------
# 7. 실행 상태 및 유틸
# ------------------------------------------------------------
store = {"resume": None, "top_k": 3}


def read_multiline(label: str) -> str:
    print(f"{label}를 입력하세요. (여러 줄 가능 / 다 쓰면 마지막 줄에 END 입력)")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def print_menu():
    print("\n" + "=" * 50)
    print(" 채용공고 RAG 에이전트")
    print(" 1) 질문하기 (공고 검색 Q&A)")
    print(" 2) 자소서 입력")
    print(" 3) 자소서 개선 (공고 기반)")
    print(f" 4) 검색 chunk 개수 변경 (현재 k={store['top_k']})")
    print(" 5) 종료")
    print("=" * 50)


def run_graph(question: str, mode: str) -> str:
    result = graph.invoke({
        "question": question,
        "retrieved_docs": [],
        "answer": "",
        "top_k": store["top_k"],
        "mode": mode,
        "resume": store["resume"] or "",
    })
    return result["answer"]


# ------------------------------------------------------------
# 8. 메인 루프 ('종료' 또는 5 입력 시 종료)
# ------------------------------------------------------------
def main():
    while True:
        print_menu()
        choice = input("선택 > ").strip()

        if choice in ("5", "종료"):
            print("프로그램을 종료합니다.")
            break

        elif choice == "1":
            question = input("\n질문 > ").strip()
            if question == "종료":
                print("프로그램을 종료합니다.")
                break
            if not question:
                continue
            print("\n[답변]")
            print(run_graph(question, mode="qa"))

        elif choice == "2":
            store["resume"] = read_multiline("자소서")
            print("자소서가 저장되었습니다.")

        elif choice == "3":
            if not store["resume"]:
                print("먼저 자소서(2)를 입력해 주세요.")
                continue
            # 자소서 내용을 검색 쿼리로 사용 -> 가장 관련 있는 공고 부분을 찾음
            query = input(
                "\n어떤 공고/직무 기준으로 개선할까요? (예: 백엔드 자격요건) > "
            ).strip()
            if not query:
                query = store["resume"][:200]  # 입력이 없으면 자소서 앞부분으로 검색

            print("\n[개선 결과]")
            print(run_graph(query, mode="improve"))

        elif choice == "4":
            val = input("검색할 chunk 개수(k)를 입력하세요 (1~10) > ").strip()
            if val.isdigit() and 1 <= int(val) <= 10:
                store["top_k"] = int(val)
                print(f"k가 {store['top_k']}(으)로 변경되었습니다.")
            else:
                print("1에서 10 사이의 숫자를 입력해 주세요.")

        else:
            print("1 ~ 5 중에서 선택해 주세요.")


if __name__ == "__main__":
    main()