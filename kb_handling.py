import math

import sqlite3
conn = sqlite3.connect("kb_log.db")
cursor = conn.cursor()

from type import KBEntry, EnvConfig


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

def retrieve_candidates(env: EnvConfig, top_k=3):
    if _is_mmr_needed():
        candidates = []
        return mmr(candidates, top_k=top_k)
    env_dict = env.__to_dict__()
    # print (f"Retrieving candidates for env: {env_dict}")
    cursor.execute(
        """SELECT 
                id, 
                embb_prob,
                urllc_prob,
                mmtc_prob,
                arrival_rate,
                departure_rate,
                w_qos,
                w_energy,
                w_fairness,
                reward,
                thinking,
                (embb_prob*? 
                + urllc_prob*? 
                + mmtc_prob*? 
                + arrival_rate*? 
                + departure_rate*?) AS sim
           FROM exp
           ORDER BY sim DESC
           LIMIT ?  
        """,
        (
            env_dict['embb_prob'],
            env_dict['urllc_prob'],
            env_dict['mmtc_prob'],
            env_dict['arrival_rate'],
            env_dict['departure_rate'],
            top_k
        )
    )
    selected = cursor.fetchall()
    return selected
   
def add_to_kb(kb_entry: KBEntry):
    kb_dict = kb_entry.__to_dict__()
    if _check_duplicated(kb_dict):
        print("Duplicate entry. Skipping KB update.")
        return
    cursor.execute("""INSERT INTO exp (
                        embb_prob, 
                        urllc_prob, 
                        mmtc_prob, 
                        arrival_rate, 
                        departure_rate, 
                        w_qos, 
                        w_energy, 
                        w_fairness, 
                        reward, 
                        thinking)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                   tuple(kb_dict.values()))
    # conn.commit()
    print (f"Added to KB: {kb_dict}")
    return

def _check_duplicated(env: dict):
    cursor.execute("""SELECT COUNT(*) FROM exp WHERE 
                    embb_prob=? AND 
                    urllc_prob=? AND 
                    mmtc_prob=? AND 
                    arrival_rate=? AND 
                    departure_rate=? AND
                    w_qos=? AND
                    w_energy=? AND
                    w_fairness=?
                   """, 
                   (env['embb_prob'], 
                    env['urllc_prob'], 
                    env['mmtc_prob'], 
                    env['arrival_rate'], 
                    env['departure_rate'],
                    env['w_qos'],
                    env['w_energy'],
                    env['w_fairness']))
    count = cursor.fetchone()[0]
    return count > 0

def _is_mmr_needed():
    kb_size = cursor.execute("SELECT COUNT(*) FROM exp").fetchone()[0]
    return kb_size >= 100