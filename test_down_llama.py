
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 로컬에 다운로드한 LLaMA 모델 경로
local_dir = r"C:\han\models\llama-3.1-8b"

# 토크나이저와 모델 불러오기
tok = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # load_in_4bit=True,  # (옵션) VRAM 적으면 bitsandbytes 설치 후 활성화
)

# 대화 이력 저장
history = [
    {"role": "system", "content": "당신은 친절한 한국어 조수입니다."}
]

def chat(messages, max_new_tokens=200):
    """대화 이력을 기반으로 모델 응답 생성"""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("💬 LLaMA 대화 시작 (종료: Ctrl+C)")
    try:
        while True:
            user_msg = input("\n👤 사용자> ").strip()
            if not user_msg:
                continue
            history.append({"role": "user", "content": user_msg})
            answer = chat(history)
            print(f"🤖 LLaMA> {answer}")
            history.append({"role": "assistant", "content": answer})
    except (KeyboardInterrupt, EOFError):
        print("\n대화를 종료합니다.")
