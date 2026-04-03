from type import EnvConfig
import math

import sqlite3
conn = sqlite3.connect("kb_log.db")
cursor = conn.cursor()

def mmr(candidates, top_k=3, mmr_lambda=0.7):
    selected = []

    def dot(a,b):
        return sum(x*y for x,y in zip(a,b))
        
    while len(selected) < top_k and candidates:

        best_doc = None
        best_score = -math.inf

        for doc in candidates:
            similarity = doc['sim']
            if not selected:
                diversity = 0
            else:
                diversity = max(dot(doc['vec'], s['vec']) for s in selected)

            mmr_score = mmr_lambda * similarity - (1 - mmr_lambda) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_doc = doc
            
        selected.append(best_doc)
        candidates.remove(best_doc)
    
    return selected

def retrieve_candidates(env_config: EnvConfig, top_k=3):
    pass

def add_to_kb(env_config: EnvConfig, reward: float):
    pass

def _check_duplicated(env_config: EnvConfig):
    pass