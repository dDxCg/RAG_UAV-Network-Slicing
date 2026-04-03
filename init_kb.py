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

# Init data from csv file
with open("exp_data.csv", "r") as f:
    reader = csv.reader(f)
    try:
        header = next(reader)  # Skip header
    except StopIteration:
        print("Warning: The CSV file is empty!")
        header = None
    
    for row in reader:
        cursor.execute("""
            INSERT INTO exp (
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)

conn.commit()

