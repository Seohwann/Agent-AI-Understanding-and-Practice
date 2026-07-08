"""
MBTI 콘텐츠 추천 프로그램
------------------------------------------------------------
LangChain + OpenAI(gpt-4o-mini)를 활용해 사용자의 MBTI 유형과
콘텐츠 유형(영화/책/음악)에 맞는 추천 목록을 리스트로 출력합니다.

체인 구조: PromptTemplate -> LLM(gpt-4o-mini) -> CommaSeparatedListOutputParser
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


# 1. .env 파일에서 OpenAI API Key 로드
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")


# 2. 출력 파서: 콤마로 구분된 결과를 파이썬 리스트로 변환
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()


# 3. 프롬프트 템플릿: MBTI 유형 + 콘텐츠 유형을 입력받음
prompt = PromptTemplate(
    template=(
        "당신은 사람의 성격에 맞는 콘텐츠를 추천하는 전문가입니다.\n"
        "MBTI 유형이 '{mbti}'인 사람에게 어울리는 {content_type} 5가지를 추천해 주세요.\n"
        "부연 설명 없이 제목만 간단히 답변하세요.\n"
        "{format_instructions}"
    ),
    input_variables=["mbti", "content_type"],
    partial_variables={"format_instructions": format_instructions},
)


# 4. LLM: gpt-4o-mini 모델
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# 5. Chain 연결: PromptTemplate -> LLM -> Parser
chain = prompt | llm | output_parser


def main():
    print("=" * 50)
    print(" MBTI 콘텐츠 추천 프로그램")
    print(" (종료하려면 '종료' 를 입력하세요)")
    print("=" * 50)

    # 6. while True 루프로 계속 이어지도록 구현
    while True:
        mbti = input("\nMBTI 유형을 입력하세요 (예: INFP): ").strip()
        if mbti == "종료":
            print("프로그램을 종료합니다.")
            break

        content_type = input("콘텐츠 유형을 입력하세요 (영화/책/음악): ").strip()
        if content_type == "종료":
            print("프로그램을 종료합니다.")
            break

        if not mbti or not content_type:
            print("입력값이 비어 있습니다. 다시 입력해 주세요.")
            continue

        # 7. 추천 결과 출력
        try:
            result = chain.invoke({
                "mbti": mbti.upper(),
                "content_type": content_type,
            })
        except Exception as e:
            print(f"추천을 가져오는 중 오류가 발생했습니다: {e}")
            continue

        print(f"\n[{mbti.upper()}] 님께 추천하는 {content_type} 목록:")
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item}")


if __name__ == "__main__":
    main()