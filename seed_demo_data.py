import sqlite3
from datetime import datetime, date

def seed_with_connection(conn):
    c = conn.cursor()

    # Clear existing demo data
    c.execute("DELETE FROM actions")
    c.execute("DELETE FROM history")
    c.execute("DELETE FROM inspections")
    c.execute("DELETE FROM batches")
    c.execute("DELETE FROM products")

    # 1. Products (IDs: 10, 11, 12)
    products = [
        (10, "Pump Casing", "Iron casting", "2026-10-09", 0),
        (11, "Valve Housing", "Steel casting", "2026-10-12", 0),
        (12, "Turbine Impeller Flange", "Metal casting", "2026-10-15", 0),
    ]
    c.executemany("INSERT INTO products (id, name, category, created_at, archived) VALUES (?, ?, ?, ?, ?)", products)

    # 2. Batches (Total 24 batches to match stat "24 PRODUCTION BATCHES")
    batches = [
        (1, "B-105", 12, 120, "2026-10-15", "Open", None, "2026-10-15 10:00:00"),
        (2, "B-104", 11, 80, "2026-10-12", "Released", "Approved for assembly", "2026-10-12 14:30:00"),
        (3, "B-103", 10, 60, "2026-10-10", "Hold", "Surface porosity check required", "2026-10-10 09:15:00"),
    ]
    for i in range(4, 25):
        pid = 10 + (i % 3)
        code = f"B-{105 + i}"
        batches.append((i, code, pid, 50 + (i * 5), "2026-10-01", "Released", "Standard signoff", "2026-10-01 12:00:00"))

    c.executemany("INSERT INTO batches (id, batch_code, product_id, quantity, production_date, status, decision_note, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batches)

    # 3. Inspections
    inspections = [
        (102, 2, "SMP-019", None, "Surface porosity check", "Item", "review", 0.65, "casting-cnn-deeplearning-v1", 14, "Pending", "", "Pending review", "Alex", "2026-10-16 11:20:00", None),
        (103, 1, "SMP-041", "user_images/sample_defective_rotor.jpeg", "Subsurface inclusion crack observed on flange", "Item", "defective", 0.9988, "casting-cnn-deeplearning-v1", 12, "Defective", "Confirmed defect on runner", "Reviewed", "Alex", "2026-10-16 14:10:00", "aiml/cam_sample_defective_rotor.jpeg"),
        (104, 1, "SMP-042", "user_images/sample_good_flange.jpeg", "Surface clean, smooth compliant finish", "Item", "good", 0.9814, "casting-cnn-deeplearning-v1", 11, "Good", "Passes dimensional and visual tolerance", "Reviewed", "Alex", "2026-10-17 09:30:00", "aiml/cam_sample_good_flange.jpeg"),
    ]

    for p_id in range(95, 101):
        inspections.append(
            (p_id, 1, f"SMP-{p_id:03d}", None, "Visual screening in progress", "Item", "review", 0.70, "casting-cnn-deeplearning-v1", 15, "Pending", "", "Pending review", "Alex", "2026-10-15 16:00:00", None)
        )

    inspections.append(
        (94, 3, "SMP-008", None, "Porous gate junction", "Item", "defective", 0.952, "casting-cnn-deeplearning-v1", 16, "Defective", "Confirmed blowhole", "Reviewed", "Alex", "2026-10-14 10:00:00", None)
    )
    inspections.append(
        (93, 3, "SMP-009", None, "Cold lap defect on flange perimeter", "Item", "defective", 0.963, "casting-cnn-deeplearning-v1", 13, "Defective", "Cracked perimeter", "Reviewed", "Alex", "2026-10-14 11:30:00", None)
    )

    for g_id in range(14, 93):
        inspections.append(
            (g_id, 2, f"SMP-{g_id:03d}", None, "Normal uniform casting surface", "Item", "good", 0.975, "casting-cnn-deeplearning-v1", 12, "Good", "Verified compliant", "Reviewed", "Alex", "2026-10-13 09:00:00", None)
        )

    c.executemany("""
        INSERT INTO inspections (
            id, batch_id, sample_ref, image_name, observation, image_kind,
            ml_prediction, ml_score, model_version, processing_ms,
            human_result, review_note, status, created_by, created_at, cam_image_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, inspections)

    # 4. Corrective Actions
    actions = [
        (1, 103, "Recalibrate gating nozzle for batch B-105", "Marcus", "2026-10-25", "Temperature gradient check required", "In progress", "2026-10-16 15:00:00"),
        (2, 94, "Inspect mold sand moisture content", "Sarah", "2026-10-24", "Moisture test on sand batches", "Open", "2026-10-15 08:30:00"),
        (3, 93, "Adjust pour temperature on casting line 2", "Alex", "2026-10-22", "Cold shut defect investigation", "In progress", "2026-10-14 14:00:00"),
        (4, 102, "Secondary ultrasonic verification on valve housing", "Sarah", "2026-10-26", "Awaiting sensor availability", "Open", "2026-10-16 16:30:00"),
        (5, 104, "Document high-yield baseline parameters", "Marcus", "2026-10-28", "Record casting speed for best practice", "Open", "2026-10-17 11:00:00"),
    ]
    c.executemany("INSERT INTO actions (id, inspection_id, title, owner, due_date, notes, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", actions)

    # 5. History
    events = [
        ("Alex", "Created inspection #104", "inspection", 104, "Passes dimensional inspection", "2026-10-17 09:30:00"),
        ("Alex", "Created inspection #103", "inspection", 103, "Defect detected on flange", "2026-10-16 14:10:00"),
        ("Marcus", "Created action #1", "action", 1, "Gating recalibration", "2026-10-16 15:00:00"),
    ]
    c.executemany("INSERT INTO history (actor, event, record_type, record_id, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)", events)
    conn.commit()


def seed(db_path='quality.db'):
    conn = sqlite3.connect(db_path)
    seed_with_connection(conn)
    conn.close()
    print("Database seeded successfully to match dashboard screenshot specifications!")


if __name__ == '__main__':
    seed()

