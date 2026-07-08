import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from news_tool import search_news   # 위에서 만든 함수 가져오기

# .env에서 API 키 불러오기
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("❌ .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    exit()

# 클라이언트 생성
client = genai.Client(api_key=gemini_api_key)

# AI 역할 설정 (요청하신 내용 그대로 사용)
system_instruction = """당신은 뉴스 요약 전문가입니다.
검색된 뉴스를 바탕으로 핵심 내용을 요약해주세요.
- 각 뉴스의 핵심 내용을 2~3문장으로 요약
- 중립적이고 객관적인 표현 사용
- 중요도 순으로 정렬해서 제시
- 출처(뉴스 제목)를 함께 표시"""

# 대화 기록이 유지되는 chat 세션 생성
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction
    ),
)

print("📰 뉴스 검색·요약 에이전트입니다!")
print("검색하고 싶은 키워드를 입력하세요.")
print("(종료하려면 '종료'를 입력하세요)\n")

# 대화 루프
while True:
    keyword = input("키워드: ").strip()

    # 종료 조건
    if keyword == "종료":
        print("에이전트: 이용해 주셔서 감사합니다. 좋은 하루 되세요! 👋")
        break

    if not keyword:
        continue

    # 1) NewsData API로 뉴스 검색
    print("🔍 뉴스를 검색하는 중...")
    news_data = search_news(keyword)

    # 검색 실패 / 결과 없음 처리
    if news_data.startswith("관련 뉴스를 찾을 수 없습니다") or \
       news_data.startswith("뉴스 검색 중 오류"):
        print(f"에이전트: {news_data}\n")
        continue

    # 2) 검색된 뉴스를 LLM에 전달해 요약 요청
    prompt = f"""다음은 '{keyword}' 키워드로 검색된 뉴스입니다.
요약 원칙에 따라 정리해주세요.

{news_data}"""

    try:
        response = chat.send_message(prompt)
        print(f"에이전트:\n{response.text}\n")
    except Exception as e:
        print(f"⚠️ 요약 중 오류가 발생했습니다: {e}\n")