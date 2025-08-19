
import time
import openai
import random
from config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_KEY, SIMULATED_LATENCY_SECONDS

openai.api_key = OPENAI_API_KEY

def query_model(prompt: str) -> str:
    if USE_OPENAI:
        return real_openai_model(prompt)
    else:
        return mock_model(prompt)

def mock_model(prompt: str) -> str:
    time.sleep(SIMULATED_LATENCY_SECONDS)
    return f"response for: {prompt}"

def real_openai_model(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "error"
