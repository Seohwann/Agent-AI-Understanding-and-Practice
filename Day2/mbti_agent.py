import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    exit()

# 클라이언트 생성
client = genai.Client(api_key=api_key)

# AI의 역할 설정 (요청하신 내용 그대로 사용)
system_instruction = """당신은 MBTI 전문가이자 콘텐츠 큐레이터입니다.
사용자의 MBTI 유형에 맞는 영화, 책, 음악을 추천해주세요.
추천 시 다음 원칙을 따르세요:
- 각 MBTI 유형의 성격 특성을 반영한 추천
- 추천 이유를 MBTI 특성과 연결해서 설명
- 친근하고 공감 가는 말투 사용
- 콘텐츠별로 최소 2개 이상 추천
- 가독성이 좋게 보일 수 있도록 각 문장별로 줄바꿈하고, 대주제가 바뀌는 부분은 줄바꿈 2번을 진행
- 마크다운 형식이 아닌 일반 텍스트로 답변
- 한국 사용자에게 맞는 추천을 제공할 수 있도록 한국 내에 존재하는 컨텐츠를 기반으로 답변
- 사용자의 MBTI 유형과 궁합이 잘 맞을 것 같은, 안 맞을 것 같은 MBTI를 각 1개씩 제공"""

# 대화 기록이 유지되는 chat 세션 생성
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction
    ),
)

print("🎬 MBTI 콘텐츠 추천 에이전트입니다!")
print("MBTI 유형과 원하는 콘텐츠(영화/책/음악)를 알려주세요.")
print("예) 'INFP인데 영화 추천해줘'")
print("(종료하려면 '종료'를 입력하세요)\n")

# 대화 루프
while True:
    user_input = input("나: ").strip()

    # 종료 조건
    if user_input == "종료":
        print("AI: 즐거운 감상 되세요! 다음에 또 추천해드릴게요 😊")
        break

    # 빈 입력은 건너뛰기
    if not user_input:
        continue

    # API 호출 (실패 시 예외처리)
    try:
        response = chat.send_message(user_input)
        print(f"AI: {response.text}\n")
    except Exception as e:
        print(f"⚠️ API 호출 중 오류가 발생했습니다: {e}\n")