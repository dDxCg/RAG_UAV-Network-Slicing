import json

from kb_handler import append_entry
import os

from type import KBEntry

def merge_jsonl(trials_dir, output_file):
    total = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for file in os.listdir(trials_dir):
            if not file.endswith(".jsonl"):
                continue

            path = os.path.join(trials_dir, file)

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    out.write(line + "\n")
                    total += 1

    print(f"[INFO] Merged {total} lines → {output_file}")

def init_database(conn, trial_file):
    count = 0

    with open(trial_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)
                entry = KBEntry(log)

                append_entry(conn, entry)
                count += 1

            except Exception as e:
                print(f"[WARN] Skipped line: {e}")

    print(f"[INFO] Inserted {count} entries")

if __name__ == "__main__":
    from db_config import connect_db
    conn = connect_db()
    trials_dir = 'trial_logs'
    merged_file = f'{trials_dir}/trials.jsonl'
    merge_jsonl(trials_dir, merged_file)
    init_database(conn, merged_file)