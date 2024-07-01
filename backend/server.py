from transformers import AutoTokenizer
import transformers 
import torch

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

# Modle Import
model = "PY007/TinyLlama-1.1B-Chat-v0.1"
print(f"{model} setting precessing...")
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Done.")

# Run API Server
app = FastAPI()

# Define json Obj
class Sequences(BaseModel):
    pre_prompt: str = None
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.7
    max_new_tokens: int = 100

# Use Instead Of Real DataBase
sequence_settings = {}

@app.get("/ping")
def read_root():
    return {"message": "Pong"}

# LLM Envrionment Setting Path
@app.post("/setting/")
async def update_setting(sequences: Sequences):
    sequence_settings['current'] = sequences.dict()
    return sequence_settings['current']

# LLM Question Path
@app.post("/question/")
async def generate_response(prompt: str):
    if 'current' not in sequence_settings:
        return {"error": "Settings not configured, please set them first."}

    current_settings = sequence_settings['current']

    formatted_prompt = f"Human: answering follow comment {current_settings['pre_prompt']} question is that {prompt} \n Assistant: "

    # Error Handling
    try:
        # Load LLM Setting Values
        result = pipeline(
            formatted_prompt,
            do_sample=current_settings['do_sample'], # 확률적 샘플링 시 고려되는 상위 N개의 후보
            top_k=current_settings['top_k'],  # 확률적 샘플링 시 고려되는 상위 N%의 후보
            top_p=current_settings['top_p'], # 답변 횟수
            num_return_sequences=1, # 반복 토큰 방지. 1이상이면 대게 많이 줄어듦
            repetition_penalty=2.0, # 한 번에 생성할 최대 토큰 수
            max_new_tokens=current_settings['max_new_tokens'],
        )
        if result and len(result) > 0:
            # Parse The Result
            generated_text = result[0]['generated_text'].split("\n Assistant:")[1]
            return {generated_text}
        else:
            return {"error": "No text was generated."}
    except Exception as e:
        return {"error": str(e)}