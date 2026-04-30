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
    from prompt_builder import build_prompt_kb
    from semantic_analyzer import _get_env_pref
    from kb_handler import query_similar_entries
    from db_config import connect_db
    from type import EnvVector, EvalVector
    conn = connect_db()
    env = {"arrival_rate": 4.5, "departure_rate": 0.5, 
           "walk_prob": 0.45, "vehicle_prob": 0.3, "stationary_prob": 0.25,
            "embb_prob": 0.05, "urllc_prob": 0.05, "mmtc_prob": 0.9}
    env = _get_env_pref(env)
    examples = query_similar_entries(conn, EnvVector(env), EvalVector(env), top_k=5)
    prompt = build_prompt_kb(env, examples)
    res = generate_response(prompt)
    print(res)
    print(type(res))