"""
Utility script to quickly inspect the SQLite database (quality.db).
Usage:
    python check_db.py
    python check_db.py inspections
    python check_db.py batches
    python check_db.py actions
"""
import sys
import sqlite3
from pathlib import Path

DB_PATH = Path("quality.db")

def format_row(row, headers):
    return " | ".join(str(row[h] if row[h] is not None else "-")[:25].ljust(len(h) if len(h) > 10 else 10) for h in headers)

def inspect_db():
    if not DB_PATH.exists():
        print(f"[!] Database file '{DB_PATH}' does not exist.")
        return

    print("=" * 70)
    print(f" DATABASE INSPECTION: {DB_PATH.resolve()}")
    print(f" File Size: {DB_PATH.stat().st_size / 1024:.2f} KB")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. Summary of Tables & Row Counts
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
    print("\n--- TABLES & RECORD COUNTS ---")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  * {t:<15}: {count:>5} records")

    # 2. Latest Inspections
    target = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if not target or target == "inspections":
        print("\n--- LATEST 5 INSPECTIONS ---")
        rows = conn.execute("""
            SELECT id, sample_ref, ml_prediction, ROUND(ml_score, 3) AS score, human_result, status, created_by, created_at
            FROM inspections ORDER BY id DESC LIMIT 5
        """).fetchall()
        if rows:
            headers = ["id", "sample_ref", "ml_prediction", "score", "human_result", "status", "created_by"]
            print(" | ".join(h.ljust(12) for h in headers))
            print("-" * 75)
            for r in rows:
                print(" | ".join(str(r[h] if r[h] is not None else "-")[:12].ljust(12) for h in headers))
        else:
            print("  (No inspections found)")

    # 3. Disputed Cases
    disputed = conn.execute("SELECT id, sample_ref, ml_prediction, human_result, status, dispute_reason FROM inspections WHERE status = 'Disputed'").fetchall()
    if disputed:
        print(f"\n--- DISPUTED / CHALLENGED INSPECTIONS ({len(disputed)}) ---")
        for d in disputed:
            print(f"  * ID #{d['id']} [{d['sample_ref']}]: AI={d['ml_prediction']} vs Human={d['human_result']} | Reason: {d['dispute_reason']}")

    # 4. Corrective Actions (CAPA)
    if not target or target == "actions":
        print("\n--- LATEST 5 CAPA ACTIONS ---")
        actions = conn.execute("SELECT id, inspection_id, title, owner, status, due_date FROM actions ORDER BY id DESC LIMIT 5").fetchall()
        if actions:
            headers = ["id", "inspection_id", "title", "owner", "status"]
            print(" | ".join(h.ljust(15) for h in headers))
            print("-" * 75)
            for a in actions:
                print(" | ".join(str(a[h] if a[h] is not None else "-")[:15].ljust(15) for h in headers))
        else:
            print("  (No corrective actions found)")

    # 5. Batches Summary
    if not target or target == "batches":
        print("\n--- BATCH STATUS SUMMARY ---")
        batches = conn.execute("SELECT status, COUNT(*) as count FROM batches GROUP BY status").fetchall()
        for b in batches:
            print(f"  * Status '{b['status']}': {b['count']} batch(es)")

    conn.close()
    print("\n" + "=" * 70)
    print(" Tips:")
    print("   Run full table dump:  python check_db.py <table_name>")
    print("   Or query SQLite CLI:  sqlite3 quality.db")
    print("=" * 70)

if __name__ == "__main__":
    inspect_db()
