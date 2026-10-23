"""
Comprehensive Test & Verification Suite for Product Quality Inspection
Tests all routes, authentication, role access, image uploading, CNN inference,
Grad-CAM heatmap generation, database persistence, and API endpoints.
"""
import os
import sys
import io
import json
from pathlib import Path
from PIL import Image

# Import the Flask application
from app import app, connect, USER_UPLOAD_DIR, AIML_UPLOAD_DIR, CNN_MODEL_PATH, CNN_MANIFEST_PATH


def create_test_image():
    """Load a real casting image from dataset if available, otherwise synthetic in memory."""
    real_sample = Path(__file__).resolve().parent / "data" / "casting" / "test" / "defective" / "cast_def_0_1059.jpeg"
    if real_sample.is_file():
        return io.BytesIO(real_sample.read_bytes())
    img = Image.new("RGB", (300, 300), color=(140, 142, 145))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def run_all_tests():
    print("=" * 70)
    print("STARTING FULL-STACK AUTOMATED VERIFICATION SUITE")
    print("=" * 70)

    client = app.test_client()
    passed = 0
    failed = 0

    def assert_test(condition, name, details=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f" [PASS] {name} {details}")
        else:
            failed += 1
            print(f" [FAIL] {name} - FAILED! {details}")

    # 1. Verification of File Structure & CNN Model
    print("\n--- 1. Storage & Model Artifact Checks ---")
    assert_test(USER_UPLOAD_DIR.is_dir(), "User upload directory exists", str(USER_UPLOAD_DIR))
    assert_test(AIML_UPLOAD_DIR.is_dir(), "AIML upload directory exists", str(AIML_UPLOAD_DIR))
    assert_test(CNN_MODEL_PATH.is_file(), "CNN model weights exist", f"{CNN_MODEL_PATH.stat().st_size / 1024:.1f} KB")
    assert_test(CNN_MANIFEST_PATH.is_file(), "CNN manifest exists")

    if CNN_MANIFEST_PATH.is_file():
        manifest = json.loads(CNN_MANIFEST_PATH.read_text(encoding="utf-8"))
        acc = manifest.get("test_accuracy", 0)
        assert_test(acc >= 0.95, "CNN Held-out Test Accuracy >= 95%", f"Actual: {acc * 100:.2f}%")

    # 2. Authentication & Redirects
    print("\n--- 2. Authentication & Access Control ---")
    res = client.get("/")
    assert_test(res.status_code == 302 and "/login" in res.headers.get("Location", ""), "Unauthenticated root / redirects to /login")

    res = client.get("/dashboard")
    assert_test(res.status_code == 302 and "/login" in res.headers.get("Location", ""), "Unauthenticated /dashboard redirects to /login")

    # Login as Manager
    res = client.post("/login", data={"username": "manager", "password": "manager123"}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"manager" in res.data, "Manager login successful")

    # 3. Operational Pages Rendering
    print("\n--- 3. Page Rendering & Design Integration ---")
    pages = [
        ("/dashboard", b"Quality Inspection Command Center"),
        ("/inspections", b"Visual Product Inspections"),
        ("/batches", b"Production Batches"),
        ("/products", b"Product Specifications"),
        ("/actions", b"Corrective & Preventive Actions"),
        ("/history", b"System Audit & Compliance Log"),
        ("/report", b"Evidence-Grounded Quality Report"),
        ("/model-architecture", b"CNN Deep Learning Model & Architecture"),
    ]

    for route, expected_text in pages:
        res = client.get(route)
        assert_test(res.status_code == 200 and expected_text in res.data, f"Route {route} renders correctly")

    # 4. Filter and Search on /inspections
    print("\n--- 4. Search and Filter Mechanisms ---")
    res = client.get("/inspections?status=defective")
    assert_test(res.status_code == 200, "Filter status=defective")

    res = client.get("/inspections?status=good")
    assert_test(res.status_code == 200, "Filter status=good")

    res = client.get("/inspections?q=B-105")
    assert_test(res.status_code == 200, "Search query q=B-105")

    # 5. Product and Batch Registration
    print("\n--- 5. Data Entity Lifecycle (Products & Batches) ---")
    test_prod_name = f"Test Casting {os.getpid()}"
    res = client.post("/products", data={"name": test_prod_name, "category": "Ductile Iron"}, follow_redirects=True)
    assert_test(res.status_code == 200 and test_prod_name.encode() in res.data, f"Created product: {test_prod_name}")

    with connect() as c:
        prod = c.execute("SELECT id FROM products WHERE name=?", (test_prod_name,)).fetchone()
        prod_id = prod["id"] if prod else 1

    test_batch_code = f"LOT-T{os.getpid()}"
    res = client.post("/batches", data={
        "batch_code": test_batch_code,
        "product_id": prod_id,
        "quantity": 100,
        "production_date": "2026-10-22"
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and test_batch_code.encode() in res.data, f"Created batch: {test_batch_code}")

    with connect() as c:
        batch = c.execute("SELECT id FROM batches WHERE batch_code=?", (test_batch_code,)).fetchone()
        batch_id = batch["id"] if batch else 1

    # Manager batch decision
    res = client.post(f"/batches/{batch_id}/decision", data={"status": "Hold", "note": "Pending dimensional verification"}, follow_redirects=True)
    assert_test(res.status_code == 200, f"Manager updated batch #{batch_id} decision to Hold")

    # 6. Real Inspection Workflow with Image Upload, CNN Inference & Grad-CAM
    print("\n--- 6. Inspection Pipeline & CNN Inference ---")
    test_img = create_test_image()
    sample_ref = f"SMPL-TEST-{os.getpid()}"

    res = client.post("/inspections", data={
        "batch_id": batch_id,
        "sample_ref": sample_ref,
        "image_kind": "Item",
        "image": (test_img, "test_casting_sample.jpg"),
        "observation": "Automated test casting image inspection run"
    }, follow_redirects=True)

    assert_test(res.status_code == 200 and sample_ref.encode() in res.data, "Inspection created & redirected to detail")

    # Verify Database Record and Folder Placement
    with connect() as c:
        rec = c.execute("SELECT * FROM inspections WHERE sample_ref=?", (sample_ref,)).fetchone()

    assert_test(rec is not None, "Inspection record saved in quality.db")
    if rec:
        inspection_id = rec["id"]
        assert_test("user_images/" in (rec["image_name"] or ""), "Uploaded user image stored in uploads/user_images/", rec["image_name"])
        assert_test("aiml/" in (rec["cam_image_name"] or ""), "Generated Grad-CAM heatmap stored in uploads/aiml/", rec["cam_image_name"])
        assert_test(rec["ml_prediction"] in ("defective", "good"), "CNN prediction recorded", rec["ml_prediction"])
        assert_test(rec["ml_score"] is not None and rec["ml_score"] > 0, "CNN confidence score recorded", f"{rec['ml_score']:.4f}")

        # Verify uploaded file serving
        res = client.get(f"/uploads/{rec['image_name']}")
        assert_test(res.status_code == 200, "User image served via /uploads/ route")

        if rec["cam_image_name"]:
            res = client.get(f"/uploads/{rec['cam_image_name']}")
            assert_test(res.status_code == 200, "Grad-CAM heatmap served via /uploads/ route")

        # 7. Human Disposition & Corrective Actions
        print("\n--- 7. Human Review Disposition & CAPA ---")
        res = client.post(f"/inspections/{inspection_id}", data={
            "human_result": "Defective",
            "review_note": "Confirmed surface defect during automated QA testing"
        }, follow_redirects=True)
        assert_test(res.status_code == 200 and b"Defective (Rejected)" in res.data, "Human review recorded as Defective")

        # Create CAPA action
        res = client.post(f"/inspections/{inspection_id}/actions", data={
            "title": "Recalibrate mold temperature sensor",
            "owner": "Marcus Test",
            "due_date": "2026-10-30",
            "notes": "Triggered by automated test defect confirmation"
        }, follow_redirects=True)
        assert_test(res.status_code == 200 and b"Recalibrate mold temperature sensor" in res.data, "CAPA action created")

        with connect() as c:
            act = c.execute("SELECT * FROM actions WHERE inspection_id=?", (inspection_id,)).fetchone()
        if act:
            act_id = act["id"]
            res = client.post(f"/actions/{act_id}/status", data={
                "status": "In progress",
                "note": "Work order dispatched to line technician"
            }, follow_redirects=True)
            assert_test(res.status_code == 200, f"CAPA action #{act_id} status updated to 'In progress'")

    # 8. Report Generation
    print("\n--- 8. Quality Report Synthesis ---")
    res = client.post("/report", data={"batch_code": test_batch_code}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"QUALITY AUDIT &amp; INSPECTION REPORT" in res.data or b"QUALITY AUDIT & INSPECTION REPORT" in res.data,
                "Report synthesized deterministically from database")

    # 9. Real-Time API Endpoint /api/predict
    print("\n--- 9. Real-Time API (/api/predict) ---")
    test_img2 = create_test_image()
    res = client.post("/api/predict", data={"image": (test_img2, "api_test.jpg")})
    assert_test(res.status_code == 200, "/api/predict returns HTTP 200")
    api_data = json.loads(res.data.decode("utf-8"))
    assert_test(api_data.get("status") == "success", "API status is 'success'")
    assert_test(api_data.get("prediction") in ("defective", "good"), "API prediction valid", api_data.get("prediction"))
    assert_test(api_data.get("cam_image_url") is not None, "API returns Grad-CAM URL", api_data.get("cam_image_url"))

    # 10. Role-Based Access Control & Target Users Matrix Verification
    print("\n--- 10. Role-Based Access Control & User Requirements Audit ---")
    
    # A. Inspector: create inspection, upload image & observations, confirm or challenge AI
    client.get("/logout")
    res = client.post("/login", data={"username": "inspector", "password": "inspect123"}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"inspector" in res.data, "Inspector login successful")

    # Inspector creates inspection with image and observation
    insp_img = create_test_image()
    res = client.post("/inspections", data={
        "batch_id": batch_id,
        "sample_ref": f"INSP-OP-{os.getpid()}",
        "observation": "Operator visual check: minor surface rough spot",
        "image_kind": "Item",
        "image": (insp_img, "op_sample.jpg"),
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and b"recorded" in res.data, "Inspector created inspection with image & observation")

    with connect() as c:
        op_insp = c.execute("SELECT * FROM inspections WHERE sample_ref=?", (f"INSP-OP-{os.getpid()}",)).fetchone()
    assert_test(op_insp is not None, "Inspector record saved in database")
    op_insp_id = op_insp["id"]

    # Inspector confirms AI suggestion
    res = client.post(f"/inspections/{op_insp_id}", data={"action": "confirm_ai"}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"confirmed" in res.data, "Inspector confirmed AI suggestion")

    # Inspector creates second sample and challenges AI suggestion
    insp_img2 = create_test_image()
    res = client.post("/inspections", data={
        "batch_id": batch_id,
        "sample_ref": f"INSP-CHAL-{os.getpid()}",
        "observation": "Visual check: suspected micro-crack missed by AI",
        "image_kind": "Surface",
        "image": (insp_img2, "chal_sample.jpg"),
    }, follow_redirects=True)
    with connect() as c:
        chal_insp = c.execute("SELECT * FROM inspections WHERE sample_ref=?", (f"INSP-CHAL-{os.getpid()}",)).fetchone()
    chal_insp_id = chal_insp["id"]

    res = client.post(f"/inspections/{chal_insp_id}", data={
        "action": "challenge_ai",
        "challenge_reason": "Visible defect along runner joint, AI false negative"
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and b"DISPUTED" in res.data, "Inspector challenged AI suggestion -> case marked Disputed")

    # Backend enforcement: Inspector CANNOT resolve dispute (403 Forbidden)
    res = client.post(f"/inspections/{chal_insp_id}/resolve-dispute", data={
        "final_decision": "Good", "resolution_note": "Inspector trying to resolve"
    })
    assert_test(res.status_code == 403, "Inspector blocked from resolving dispute (403 Forbidden)")

    # Backend enforcement: Inspector CANNOT update action status / submit evidence (403 Forbidden)
    res = client.post("/actions/1/status", data={"status": "Completed", "note": "Inspector attempt"})
    assert_test(res.status_code == 403, "Inspector blocked from updating action status (403 Forbidden)")

    # Backend enforcement: Inspector CANNOT record batch decisions (403 Forbidden)
    res = client.post(f"/batches/{batch_id}/decision", data={"status": "Released", "note": "Inspector attempt"})
    assert_test(res.status_code == 403, "Inspector blocked from batch decisions (403 Forbidden)")

    # Backend enforcement: Inspector CANNOT access /settings (403 Forbidden)
    res = client.get("/settings")
    assert_test(res.status_code == 403, "Inspector blocked from /settings (403 Forbidden)")

    # B. Production Supervisor: Handle assigned issues, record corrective work, submit follow-up evidence
    client.get("/logout")
    res = client.post("/login", data={"username": "supervisor", "password": "super123"}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"supervisor" in res.data, "Supervisor login successful")

    # Supervisor creates action
    res = client.post(f"/inspections/{op_insp_id}/actions", data={
        "title": "Recalibrate runner gate polishing",
        "owner": "Supervisor Sarah",
        "due_date": "2026-10-31",
        "notes": "Containment order"
    }, follow_redirects=True)
    assert_test(res.status_code == 200, "Supervisor created corrective action")

    with connect() as c:
        sup_act = c.execute("SELECT * FROM actions WHERE inspection_id=?", (op_insp_id,)).fetchone()
    sup_act_id = sup_act["id"]

    # Supervisor records corrective work and submits follow-up evidence image
    evidence_img = create_test_image()
    res = client.post(f"/actions/{sup_act_id}/status", data={
        "status": "Completed",
        "corrective_work": "Mold core dismantled, thermal sensor replaced, test cast passed",
        "note": "Re-inspection verified zero porosity",
        "evidence_file": (evidence_img, "evidence_photo.jpg")
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and b"Corrective work" in res.data, "Supervisor recorded corrective work & submitted follow-up evidence")

    with connect() as c:
        updated_act = c.execute("SELECT * FROM actions WHERE id=?", (sup_act_id,)).fetchone()
    assert_test(updated_act["status"] == "Completed", "Action status is 'Completed'")
    assert_test(bool(updated_act["corrective_work"]), "Action corrective_work recorded in DB")
    assert_test(bool(updated_act["evidence_image"]), "Action evidence_image recorded in DB")

    # Backend enforcement: Supervisor CANNOT resolve dispute (403 Forbidden)
    res = client.post(f"/inspections/{chal_insp_id}/resolve-dispute", data={
        "final_decision": "Good", "resolution_note": "Supervisor attempt"
    })
    assert_test(res.status_code == 403, "Supervisor blocked from resolving dispute (403 Forbidden)")

    # Backend enforcement: Supervisor CANNOT record batch decisions (403 Forbidden)
    res = client.post(f"/batches/{batch_id}/decision", data={"status": "Released", "note": "Supervisor attempt"})
    assert_test(res.status_code == 403, "Supervisor blocked from batch decisions (403 Forbidden)")

    # Backend enforcement: Supervisor CANNOT access /settings (403 Forbidden)
    res = client.get("/settings")
    assert_test(res.status_code == 403, "Supervisor blocked from /settings (403 Forbidden)")

    # C. Quality Manager: Resolve disputed cases, record batch decisions, review trends/settings
    client.get("/logout")
    res = client.post("/login", data={"username": "manager", "password": "manager123"}, follow_redirects=True)
    assert_test(res.status_code == 200 and b"manager" in res.data, "Manager login successful")

    # Manager resolves disputed case
    res = client.post(f"/inspections/{chal_insp_id}/resolve-dispute", data={
        "final_decision": "Defective",
        "resolution_note": "Managerial review confirms subsurface shrinkage crack under optical comparator. Lot quarantined."
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and b"successfully resolved" in res.data, "Quality Manager resolved disputed case")

    with connect() as c:
        resolved_rec = c.execute("SELECT * FROM inspections WHERE id=?", (chal_insp_id,)).fetchone()
    assert_test(resolved_rec["human_result"] == "Defective", "Disputed case resolved as 'Defective'")
    assert_test("Resolved" in resolved_rec["status"], "Disputed case status marked 'Resolved'")
    assert_test(resolved_rec["resolved_by"] == "manager", "Disputed case recorded resolved_by manager")

    # Manager records batch release decision
    res = client.post(f"/batches/{batch_id}/decision", data={
        "status": "Released",
        "note": "Production batch released after containment verification"
    }, follow_redirects=True)
    assert_test(res.status_code == 200, "Quality Manager recorded batch decision")

    # Manager reviews trends and updates settings
    res = client.get("/settings")
    assert_test(res.status_code == 200 and b"Quality Governance Trends" in res.data, "Manager accessed /settings page")

    res = client.post("/settings", data={
        "ai_confidence_threshold": "0.90",
        "dispute_sensitivity": "High (Auto-flag if ML score 0.35 - 0.70)",
        "auto_quarantine_defect_rate": "4.5",
        "quality_standard": "ISO 9001:2015 / ASTM E155 Reference",
        "escalation_policy": "Mandatory CAPA on Defective Castings"
    }, follow_redirects=True)
    assert_test(res.status_code == 200 and b"persisted" in res.data, "Manager updated and persisted quality settings")

    with connect() as c:
        setting_val = c.execute("SELECT value FROM settings WHERE key='ai_confidence_threshold'").fetchone()
    assert_test(setting_val and setting_val["value"] == "0.90", "Settings persisted correctly in database")

    # Clean up test artifacts from database so they do not pollute live UI
    with connect() as c:
        c.execute("DELETE FROM actions WHERE inspection_id IN (SELECT id FROM inspections WHERE sample_ref LIKE 'SMPL-TEST-%' OR sample_ref LIKE 'INSP-%')")
        c.execute("DELETE FROM history WHERE record_type='inspection' AND record_id IN (SELECT id FROM inspections WHERE sample_ref LIKE 'SMPL-TEST-%' OR sample_ref LIKE 'INSP-%')")
        c.execute("DELETE FROM inspections WHERE sample_ref LIKE 'SMPL-TEST-%' OR sample_ref LIKE 'INSP-%'")

    # Summary
    print("\n" + "=" * 70)
    print(f"VERIFICATION RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
