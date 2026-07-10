"""
자소서 리뷰 에이전트 (확장판)
------------------------------------------------------------
메뉴:
  1) 자소서 입력
  2) 채용공고 입력          (여러 개 추가 저장 가능)
  3) 리뷰 시작              (결과를 txt 파일로 저장)
  4) 면접 질문 생성          (예상 면접 질문 5개)
  5) 채용공고 비교           (자소서와 가장 잘 맞는 공고 추천)
  6) 종료

구성:
- create_tool_calling_agent + AgentExecutor (verbose=True)
- ConversationBufferMemory 로 대화 기록 유지
- gpt-4o-mini 사용, while True 루프, '종료' 또는 6 입력 시 종료

[설치] uv add langchain langchain-classic langchain-openai python-dotenv
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 레거시 API: 1.0에서는 langchain_classic, 그 이전 버전은 langchain ---
try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # LangChain < 1.0
    from langchain.agents import AgentExecutor, create_tool_calling_agent

try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:  # LangChain < 1.0
    from langchain.memory import ConversationBufferMemory


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. LLM (gpt-4o-mini)
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ------------------------------------------------------------
# 2. Tool 정의 (@tool)
# ------------------------------------------------------------
@tool
def review_resume(resume: str) -> str:
    """자기소개서의 강점과 개선점을 피드백할 때 사용합니다.
    사용자가 자소서 리뷰, 첨삭, 강점/약점 분석을 요청할 때 호출하세요.

    Args:
        resume: 리뷰할 자기소개서 전문
    """
    prompt = (
        "다음 자기소개서를 분석해 주세요.\n"
        "1) 강점 3가지 (근거 포함)\n"
        "2) 개선점 3가지 (구체적인 수정 방향 포함)\n"
        "항목별로 명확히 구분해 답변하세요.\n\n"
        f"[자기소개서]\n{resume}"
    )
    return llm.invoke(prompt).content


@tool
def analyze_job_posting(job_posting: str) -> str:
    """채용공고에서 핵심 역량과 키워드를 추출할 때 사용합니다.
    사용자가 채용공고 분석, 요구 역량·키워드 파악을 요청할 때 호출하세요.

    Args:
        job_posting: 분석할 채용공고 전문
    """
    prompt = (
        "다음 채용공고를 분석해 주세요.\n"
        "1) 핵심 요구 역량 (우선순위 순)\n"
        "2) 자소서에 반영하면 좋은 핵심 키워드 목록\n"
        "3) 이 공고가 중요하게 보는 인재상 요약\n\n"
        f"[채용공고]\n{job_posting}"
    )
    return llm.invoke(prompt).content


@tool
def improve_resume(resume: str, job_posting: str) -> str:
    """채용공고에 맞춰 자기소개서를 개선한 버전을 생성할 때 사용합니다.
    사용자가 특정 공고에 맞춘 자소서 개선/재작성을 요청할 때 호출하세요.

    Args:
        resume: 원본 자기소개서
        job_posting: 맞춤 대상 채용공고
    """
    prompt = (
        "아래 채용공고의 핵심 역량과 키워드를 반영하여, "
        "원본 자기소개서를 개선한 버전을 작성해 주세요.\n"
        "- 공고의 핵심 키워드를 자연스럽게 녹일 것\n"
        "- 구체적 경험과 성과 중심으로 서술할 것\n"
        "- 원본에 없는 경력이나 사실을 지어내지 말 것\n\n"
        f"[채용공고]\n{job_posting}\n\n"
        f"[원본 자기소개서]\n{resume}"
    )
    return llm.invoke(prompt).content


@tool
def generate_interview_questions(resume: str, job_posting: str) -> str:
    """저장된 자소서와 채용공고를 바탕으로 예상 면접 질문을 생성할 때 사용합니다.
    사용자가 면접 대비, 예상 질문 생성을 요청할 때 호출하세요.

    Args:
        resume: 지원자의 자기소개서
        job_posting: 지원 대상 채용공고
    """
    prompt = (
        "다음 자소서와 채용공고를 바탕으로, 이 지원자가 실제 면접에서 받을 "
        "가능성이 높은 예상 면접 질문 5개를 생성해 주세요.\n"
        "- 각 질문마다 '왜 이 질문이 나올지' 의도를 한 줄로 덧붙일 것\n"
        "- 자소서의 약한 부분(수치 부족 등)과 공고의 핵심 역량을 반영할 것\n\n"
        f"[자소서]\n{resume}\n\n"
        f"[채용공고]\n{job_posting}"
    )
    return llm.invoke(prompt).content


@tool
def recommend_best_job_posting(resume: str, job_postings: str) -> str:
    """여러 채용공고 중 자소서와 가장 잘 맞는 공고를 추천할 때 사용합니다.
    사용자가 여러 공고 비교, 가장 적합한 공고 추천을 요청할 때 호출하세요.

    Args:
        resume: 지원자의 자기소개서
        job_postings: 비교할 여러 채용공고 (각 공고가 제목과 함께 나열된 텍스트)
    """
    prompt = (
        "다음 자소서와 여러 채용공고를 비교해 주세요.\n"
        "1) 각 공고별 적합도(상/중/하)와 근거를 간단히 정리\n"
        "2) 자소서와 가장 잘 맞는 공고 1개를 최종 추천하고 이유를 명확히 설명\n\n"
        f"[자소서]\n{resume}\n\n"
        f"[채용공고 목록]\n{job_postings}"
    )
    return llm.invoke(prompt).content


tools = [
    review_resume,
    analyze_job_posting,
    improve_resume,
    generate_interview_questions,
    recommend_best_job_posting,
]


# ------------------------------------------------------------
# 3. system_instruction (요청 문구 그대로)
# ------------------------------------------------------------
system_instruction = (
    "당신은 10년 경력의 취업 컨설턴트입니다. "
    "자소서를 분석하고 채용공고에 맞게 개선하는 것이 전문입니다.\n"
    "- 자소서의 강점과 개선점을 구체적으로 피드백\n"
    "- 채용공고의 핵심 역량과 키워드를 추출\n"
    "- 추출한 키워드를 반영해 개선된 자소서 생성"
)


# ------------------------------------------------------------
# 4. 프롬프트 + Memory + 에이전트
# ------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


# ------------------------------------------------------------
# 5. 저장 상태 (자소서 1개 + 채용공고 여러 개)
# ------------------------------------------------------------
store = {"resume": None, "job_postings": []}  # job_postings: [{"name","text"}]


def print_menu():
    print("\n" + "=" * 45)
    print(" 자소서 리뷰 & 채용공고 맞춤 에이전트")
    print(" 1) 자소서 입력")
    print(" 2) 채용공고 입력 (여러 개 가능)")
    print(" 3) 리뷰 시작 (결과 txt 저장)")
    print(" 4) 면접 질문 생성")
    print(" 5) 채용공고 비교 (가장 잘 맞는 공고 추천)")
    print(" 6) 종료")
    print("=" * 45)
    if store["job_postings"]:
        print(f" (저장된 공고 {len(store['job_postings'])}개)")


def read_multiline(label: str) -> str:
    """여러 줄 텍스트 입력. 마지막 줄에 END 만 입력하면 종료됩니다."""
    print(f"{label}를 입력하세요. (여러 줄 가능 / 다 쓰면 마지막 줄에 END 입력)")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def select_posting():
    """저장된 공고가 여러 개면 하나를 선택하게 하고, 하나면 그것을 반환."""
    postings = store["job_postings"]
    if len(postings) == 1:
        return postings[0]
    print("\n저장된 채용공고:")
    for i, p in enumerate(postings, 1):
        print(f"  {i}) {p['name']}")
    while True:
        sel = input("공고 번호 선택 > ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(postings):
            return postings[int(sel) - 1]
        print("올바른 번호를 입력해 주세요.")


def format_all_postings() -> str:
    return "\n\n".join(f"### {p['name']}\n{p['text']}" for p in store["job_postings"])


def save_to_txt(content: str, prefix: str = "review") -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(filename)


# ------------------------------------------------------------
# 6. 메인 루프
# ------------------------------------------------------------
def main():
    while True:
        print_menu()
        choice = input("선택 > ").strip()

        if choice in ("6", "종료"):
            print("프로그램을 종료합니다.")
            break

        elif choice == "1":
            store["resume"] = read_multiline("자소서")
            print("자소서가 저장되었습니다.")

        elif choice == "2":
            name = f"공고{len(store['job_postings']) + 1}"
            text = read_multiline(f"채용공고({name})")
            store["job_postings"].append({"name": name, "text": text})
            print(f"{name}이(가) 저장되었습니다. (현재 {len(store['job_postings'])}개)")

        elif choice == "3":
            if not store["resume"] or not store["job_postings"]:
                print("자소서(1)와 채용공고(2)를 먼저 입력해 주세요.")
                continue
            posting = select_posting()
            request = (
                "저장된 자소서를 리뷰하고, 채용공고를 분석한 뒤, "
                "공고에 맞춰 개선된 자소서를 만들어 주세요.\n\n"
                f"[자소서]\n{store['resume']}\n\n"
                f"[채용공고]\n{posting['text']}"
            )
            result = agent_executor.invoke({"input": request})
            output = result["output"]

            print("\n" + "=" * 45)
            print(" [최종 결과]")
            print("=" * 45)
            print(output)

            # 결과를 txt 파일로 저장
            record = (
                f"[리뷰 결과] {datetime.now():%Y-%m-%d %H:%M}\n"
                f"대상 공고: {posting['name']}\n"
                + "=" * 45 + "\n" + output
            )
            path = save_to_txt(record, prefix="review")
            print(f"\n결과를 저장했습니다 → {path}")

        elif choice == "4":
            if not store["resume"] or not store["job_postings"]:
                print("자소서(1)와 채용공고(2)를 먼저 입력해 주세요.")
                continue
            posting = select_posting()
            request = (
                "다음 자소서와 채용공고를 바탕으로 예상 면접 질문 5개를 생성해 주세요.\n\n"
                f"[자소서]\n{store['resume']}\n\n"
                f"[채용공고]\n{posting['text']}"
            )
            result = agent_executor.invoke({"input": request})
            print("\n[예상 면접 질문]")
            print(result["output"])

        elif choice == "5":
            if not store["resume"]:
                print("먼저 자소서(1)를 입력해 주세요.")
                continue
            if len(store["job_postings"]) < 2:
                print("비교하려면 채용공고를 2개 이상 입력해 주세요. (메뉴 2)")
                continue
            request = (
                "다음 자소서와 여러 채용공고를 비교해, 자소서와 가장 잘 맞는 공고를 "
                "추천하고 이유를 설명해 주세요.\n\n"
                f"[자소서]\n{store['resume']}\n\n"
                f"[채용공고 목록]\n{format_all_postings()}"
            )
            result = agent_executor.invoke({"input": request})
            print("\n[공고 비교 및 추천]")
            print(result["output"])

        else:
            print("1 ~ 6 중에서 선택해 주세요.")


if __name__ == "__main__":
    main()