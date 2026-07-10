# ngrok_run.py — 프로젝트 폴더에 미리 저장
from pyngrok import ngrok
tunnel = ngrok.connect(8501)
print(f"공개 URL: {tunnel.public_url}")
input("종료하려면 Enter를 누르세요...")
ngrok.kill()