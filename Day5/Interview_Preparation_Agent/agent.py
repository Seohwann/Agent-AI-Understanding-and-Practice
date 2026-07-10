"""
agent.py - 면접 준비 에이전트 (LangChain)
------------------------------------------------------------
Tool:
  1) analyze_resume_and_job(resume, job_posting) : 자소서·채용공고 분석
  2) generate_interview_questions(analysis)      : 예상 면접 질문 5가지 생성
  3) feedback_answer(question, answer)           : STAR 기법 기준 답변 피드백

구성:
  - create_tool_calling_agent + AgentExecutor (verbose=True)
  - ConversationBufferMemory 로 대화 기록 유지
  - gpt-4o-mini 사용
  - run_interview_coach(user_input) 으로 외부(app.py)에서 호출

[설치]
uv add langchain langchain-classic langchain-openai python-dotenv streamlit
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 레거시 API: LangChain 1.0 이후는 langchain_classic, 이전은 langchain ---
try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_tool_calling_agent

try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.memory import ConversationBufferMemory


# ------------------------------------------------------------
# 0. 환경 변수 로드 (.env 에서 OPENAI_API_KEY)
# ------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ------------------------------------------------------------
# 1. LLM (gpt-4o-mini)
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# ------------------------------------------------------------
# 2. Tool 정의 (@tool)
#    docstring 이 곧 LLM 이 읽는 도구 설명(description)이 됩니다.
# ------------------------------------------------------------
@tool
def analyze_resume_and_job(resume: str, job_posting: str) -> str:
    """자기소개서와 채용공고를 함께 분석할 때 사용합니다.
    사용자가 자소서와 공고의 매칭도, 강점, 부족한 역량 파악을 원할 때 호출하세요.

    Args:
        resume: 지원자의 자기소개서 전문
        job_posting: 지원 대상 채용공고 전문
    """
    prompt = (
        "다음 자소서와 채용공고를 분석해 주세요.\n"
        "1) 채용공고의 핵심 요구 역량과 키워드\n"
        "2) 자소서에서 공고와 잘 맞는 강점\n"
        "3) 자소서에 부족하거나 보강이 필요한 역량\n"
        "4) 면접에서 집중적으로 검증될 만한 지점\n\n"
        f"[자기소개서]\n{resume}\n\n"
        f"[채용공고]\n{job_posting}"
    )
    return llm.invoke(prompt).content


@tool
def generate_interview_questions(analysis: str) -> str:
    """분석 결과를 바탕으로 예상 면접 질문 5가지를 생성할 때 사용합니다.
    사용자가 면접 대비, 예상 질문 목록을 요청할 때 호출하세요.

    Args:
        analysis: analyze_resume_and_job 으로 얻은 분석 결과
    """
    prompt = (
        "다음 분석 결과를 바탕으로 실제 면접에서 나올 법한 예상 질문 5가지를 "
        "생성해 주세요.\n"
        "- 각 질문마다 '이 질문의 의도'를 한 줄로 덧붙일 것\n"
        "- 채용공고의 핵심 역량과 자소서 내용을 반드시 반영할 것\n"
        "- 압박 질문과 경험 검증 질문을 골고루 섞을 것\n\n"
        f"[분석 결과]\n{analysis}"
    )
    return llm.invoke(prompt).content


@tool
def feedback_answer(question: str, answer: str) -> str:
    """면접 답변을 STAR 기법 기준으로 평가하고 피드백할 때 사용합니다.
    사용자가 자신의 답변에 대한 피드백, 첨삭을 요청할 때 호출하세요.

    Args:
        question: 면접 질문
        answer: 지원자가 작성한 답변
    """
    prompt = (
        "다음 면접 답변을 STAR 기법 기준으로 평가해 주세요.\n\n"
        "평가 항목:\n"
        "- Situation(상황): 배경과 맥락이 명확한가\n"
        "- Task(과제): 본인의 역할과 목표가 드러나는가\n"
        "- Action(행동): 구체적으로 무엇을 했는지 설명하는가\n"
        "- Result(결과): 성과가 수치나 근거로 뒷받침되는가\n\n"
        "출력 형식:\n"
        "1) 항목별 평가 (각 항목 상/중/하 + 근거)\n"
        "2) 잘한 점 (지원자가 자신감을 가질 수 있도록)\n"
        "3) 개선이 필요한 부분 + 구체적인 개선 예시 문장\n\n"
        f"[면접 질문]\n{question}\n\n"
        f"[지원자 답변]\n{answer}"
    )
    return llm.invoke(prompt).content


tools = [analyze_resume_and_job, generate_interview_questions, feedback_answer]


# ------------------------------------------------------------
# 3. system_instruction (요청하신 문구 그대로)
# ------------------------------------------------------------
system_instruction = (
    "당신은 10년 경력의 대기업 인사담당자이자 면접 코치입니다. "
    "자소서와 채용공고를 분석해 실제 면접에서 나올 법한 질문을 생성하고 "
    "답변에 구체적이고 건설적인 피드백을 제공합니다.\n"
    "- 예상 질문은 채용공고의 핵심 역량과 자소서 내용을 반영\n"
    "- 답변 피드백은 STAR 기법 기준으로 평가\n"
    "- 개선이 필요한 부분은 구체적인 개선 예시와 함께 제시\n"
    "- 지원자가 자신감을 가질 수 있도록 강점도 함께 언급"
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

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


# ------------------------------------------------------------
# 5. 외부 호출용 함수 (app.py 에서 import)
# ------------------------------------------------------------
def run_interview_coach(user_input: str) -> str:
    """사용자 입력을 에이전트에 전달하고 최종 답변을 반환합니다."""
    try:
        result = agent_executor.invoke({"input": user_input})
        return result["output"]
    except Exception as e:
        return f"오류가 발생했습니다: {e}"


def reset_memory() -> None:
    """대화 기록을 초기화합니다."""
    memory.clear()


# ------------------------------------------------------------
# 6. 터미널에서 단독 실행 시 테스트용
# ------------------------------------------------------------
if __name__ == "__main__":
    print("면접 준비 에이전트 (종료하려면 '종료' 입력)")
    while True:
        user_input = input("\n> ").strip()
        if user_input == "종료":
            break
        if not user_input:
            continue
        print("\n" + run_interview_coach(user_input))