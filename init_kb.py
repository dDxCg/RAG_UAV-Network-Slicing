import csv
import sqlite3

conn = sqlite3.connect("kb_log.db")
cursor = conn.cursor()

# CREATE TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS exp (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embb_prob REAL,
    urllc_prob REAL,
    mmtc_prob REAL,
    arrival_rate REAL,
    departure_rate REAL,
    w_qos REAL,
    w_energy REAL,
    w_fairness REAL,
    reward REAL,
    thinking TEXT
)
""")

# TODO: Import data from jsonl file

