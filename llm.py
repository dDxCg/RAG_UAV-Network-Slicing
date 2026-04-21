from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()

api_key = os.getenv("MINIMAX")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

def generate_response(prompt):
    response = client.chat.completions.create(
        model="minimax/minimax-m2.5:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    res =  response.choices[0].message.content
    return _clean_response(res)

def _clean_response(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return text


if __name__ == "__main__":
    from prompt_builder import build_prompt_zs, build_prompt_fs
    
    env = {"arrival_rate": 4.5, "departure_rate": 0.5, 
           "walk_prob": 0.45, "vehicle_prob": 0.3, "staionary_prob": 0.25,
            "embb_prob": 0.05, "urllc_prob": 0.05, "mmtc_prob": 0.9, 
            "qos_demand": 0.22, "m_stress": 0.325, "dynamic_stress": 0.2925,
            "pressure": 0.0005, 
            "raw_jain": 0.5691, "jain_m": 0.6665, "jain_s": 0.4859, "jain": 0.0}
    prompt = build_prompt_fs(env)
    res = generate_response(prompt)
    print(res)
    print(type(res))