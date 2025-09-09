#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import requests
print("1")
# Ollama 서버 주소
API_URL = "http://localhost:11434/api/generate"
print("2")
# 사용자 프롬프트 입력
prompt = "Explain black holes in simple terms."
print("3")
# 요청 데이터
payload = {
    "model": "llama3",        # 사용할 모델 이름 (ollama run 한 이름)
    "prompt": prompt,
    "stream": False           # True면 스트리밍 응답
}
print("4")
# POST 요청
response = requests.post(API_URL, json=payload)
print("5")
# 응답 출력
print(response.json()['response'])