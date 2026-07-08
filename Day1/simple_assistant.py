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

# AI의 역할 설정 (여기서는 여행 전문가로 설정)
system_instruction = (
    "당신은 단호한 여행 전문가입니다. "
    "사용자가 가고 싶어하는 여행지에 대해서 최대한"
    "낭만적인 여행이 될 수 있도록 안내해 주세요. "
    "답변은 너무 길지 않게, 핵심 위주로 해주세요."
)

# 대화 기록이 유지되는 chat 세션 생성
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction
    ),
)

print("🌍 여행 전문가 AI 어시스턴트입니다. 무엇이든 물어보세요!")
print("(종료하려면 '종료'를 입력하세요)\n")

# 대화 루프
while True:
    user_input = input("나: ").strip()

    # 종료 조건
    if user_input == "종료":
        print("AI: 즐거운 여행 되세요! 👋")
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