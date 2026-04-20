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
    from prompt_builder import build_prompt_zs
    
    env = {"arrival_rate": 2.5, "departure_rate": 2.5, 
           "walk_prob": 0.33, "vehicle_prob": 0.33, "staionary_prob": 0.34, 
           "embb_prob": 0.33, "urllc_prob": 0.33, "mmtc_prob": 0.34, 
           "qos_demand": 0.332, 
           "dynamic_ratio": 1.0, "dynamic_stress": 0.1655,  
           "m_stress": 0.331, 
           "raw_pressure": 0.4975, "pressure": 0.0002, 
           "raw_jain": 0.7599, "jain": 0.9998, "jain_m": 0.6203, "jain_s": 0.9308}
    prompt = build_prompt_zs(env)
    res = generate_response(prompt)
    print(res)
    print(type(res))