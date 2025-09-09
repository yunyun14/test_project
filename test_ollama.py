# chat_llama3_ollama.py
# 로컬 Ollama 서버(기본: http://localhost:11434)에 llama3로 멀티턴 채팅하기
import requests, json, sys

API_URL = "http://localhost:11434/api/chat"
MODEL   = "llama3"      # 예: "llama3", "llama3:8b", "llama3:70b" 등 설치된 태그 사용

# 대화 이력 (system + user/assistant 메시지 누적)
messages = [
    {"role": "system", "content": "당신은 친절하고 간결한 한국어 조수입니다."}
]

def chat_once(user_text: str, stream: bool = True, temperature: float = 0.7):
    """
    user_text를 대화 이력에 추가하고, Ollama /api/chat 호출.
    stream=True면 토큰 단위로 바로바로 출력, False면 완성본 한번에 반환.
    """
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            # 필요시 힌트:
            # "num_ctx": 8192, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1
        }
    }

    if stream:
        # 서버가 보내는 JSON 라인을 순차적으로 받아서 content만 즉시 출력
        with requests.post(API_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                # data 예: {"message":{"role":"assistant","content":"..."}, "done":false}
                msg = data.get("message", {}).get("content", "")
                if msg:
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    full.append(msg)
                if data.get("done"):
                    break
            print()  # 줄바꿈
            assistant_text = "".join(full)
    else:
        # 완성본을 한 번에 받기
        r = requests.post(API_URL, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        assistant_text = data["message"]["content"]
        print(assistant_text)

    # 대화 이력에 assistant 응답도 누적
    messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text

if __name__ == "__main__":
    print("🔗 모델:", MODEL)
    print("대화를 시작하세요. 종료: Ctrl+C")
    try:
        while True:
            user = input("\n사용자> ").strip()
            if not user:
                continue
            chat_once(user, stream=True)
    except (KeyboardInterrupt, EOFError):
        print("\n[대화 종료]")
