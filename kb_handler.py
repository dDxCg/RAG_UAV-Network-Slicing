from type import EnvVector, EvalVector, KBEntry

def append_entry(conn, entry: KBEntry):
    id = _find_existed_row(conn, entry)
    if id:
        _update_metadata(conn, id[0], entry)
    else:
        _add_entry(conn, entry)

def query_similar_entries(conn, env_vector: EnvVector, eval_vector: EvalVector, top_k=3):
    env_vector = env_vector.__pgvector__()
    eval_vector = eval_vector.__pgvector__()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT metadata,
                   (env <=> %s) AS env_distance,
                   (eval <=> %s) AS eval_distance
            FROM experiences
            ORDER BY eval_distance, env_distance
            LIMIT %s;
            """,
            (env_vector, eval_vector, top_k)
        )
        results = cur.fetchall()
    return [result[0] for result in results]

def _find_existed_row(conn, entry: KBEntry):
    env_vector = entry.env_vector.__pgvector__()
    weight_vector = entry.weight_vector.__pgvector__()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM experiences
            WHERE env = %s AND weights = %s
            LIMIT 1;
            """,
            (env_vector, weight_vector)
        )
        return cur.fetchone() 

def _update_metadata(conn, id, entry: KBEntry):
    metadata = entry.metadata
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE experiences
            SET metadata = %s,
                created_at = NOW()
            WHERE id = %s;
            """,
            (metadata, id)
        )
        conn.commit()

def _add_entry(conn, entry: KBEntry):
    env_vector = entry.env_vector.__pgvector__()
    eval_vector = entry.eval_vector.__pgvector__()
    weight_vector = entry.weight_vector.__pgvector__()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiences (env, eval, weights, metadata)
            VALUES (%s, %s, %s, %s);
            """,
            (env_vector, eval_vector, weight_vector, entry.metadata)
        )
        conn.commit()


if __name__ == "__main__":
    from db_config import connect_db
    conn = connect_db()
    # Example usage
    env = {
        "embb_prob": 0.33, "urllc_prob": 0.33, "mmtc_prob": 0.34,
        "walk_prob": 0.33, "vehicle_prob": 0.33, "stationary_prob": 0.34,
        "arrival_rate": 2.5, "departure_rate": 2.5,
        "pressure": 0.0002, "qos_demand": 0.332, "m_stress": 0.331, "dynamic_stress": 0.1655
    }
    selected_exps = query_similar_entries(conn, EnvVector(env), EvalVector(env), top_k=5)
    for exp in selected_exps:
        print(f"{exp}\n")