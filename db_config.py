import psycopg
from pgvector.psycopg import register_vector
import os
from dotenv import load_dotenv

load_dotenv()  

def connect_db():
    conn = psycopg.connect(os.getenv("DATABASE_URL"))
    # conn = psycopg.connect(
    #     dbname=os.getenv("DB_NAME"),
    #     user=os.getenv("DB_USER"),
    #     password=os.getenv("DB_PASSWORD"),
    #     host=os.getenv("DB_HOST"),
    #     port=os.getenv("DB_PORT"),
    # )
    register_vector(conn)
    return conn

def ping_timeout(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
    except psycopg.OperationalError as e:
        print(f"[ERROR] Database connection lost: {e}")
        return connect_db()
    return conn

def test_db():
    conn = connect_db()
    conn = ping_timeout(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

if __name__ == "__main__":
    test_db()