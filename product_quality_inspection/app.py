import os
import sqlite3
import json
import time
import hashlib
from datetime import datetime, date
from pathlib import Path
from functools import wraps
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    abort,
    send_from_directory,
    jsonify,
)
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from skimage.feature import hog
import joblib

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "quality.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "models" / "casting_svm.joblib"
MANIFEST_PATH = BASE_DIR / "models" / "casting_svm.manifest.json"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = 8

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-qualitytrack-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ROLES = {"inspector", "supervisor", "manager"}
DEMO_USERS = {
    "inspector": {"password": "inspect123", "role": "inspector", "name": "Inspector Alex"},
    "supervisor": {"password": "super123", "role": "supervisor", "name": "Supervisor Sarah"},
    "manager": {"password": "manager123", "role": "manager", "name": "Manager Marcus"},
}

# ---------------------------------------------------------
# Database Operations
# ---------------------------------------------------------

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, category TEXT NOT NULL DEFAULT 'Metal casting',
            created_at TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_code TEXT NOT NULL UNIQUE, product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL CHECK(quantity > 0),
            production_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Open',
            decision_note TEXT, updated_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id)
        );
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL, sample_ref TEXT NOT NULL,
            image_name TEXT, observation TEXT NOT NULL DEFAULT '',
            image_kind TEXT NOT NULL DEFAULT 'Item',
            ml_prediction TEXT, ml_score REAL, model_version TEXT,
            processing_ms INTEGER, human_result TEXT NOT NULL DEFAULT 'Pending',
            review_note TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'Pending review',
            created_by TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES batches(id)
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL, title TEXT NOT NULL,
            owner TEXT NOT NULL, due_date TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'Open', created_at TEXT NOT NULL,
            FOREIGN KEY(inspection_id) REFERENCES inspections(id)
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL,
            event TEXT NOT NULL, record_type TEXT NOT NULL, record_id INTEGER NOT NULL,
            reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
        );
        """)
        if c.execute("SELECT COUNT(*) n FROM products").fetchone()["n"] == 0:
            now = datetime.now().isoformat(timespec="seconds")
            c.execute(
                "INSERT INTO products(name,category,created_at) VALUES(?,?,?)",
                ("Precision Cast Rotor Flange", "Metal casting", now),
            )
            pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            c.execute(
                "INSERT INTO batches(batch_code,product_id,quantity,production_date,status,updated_at) VALUES(?,?,?,?,?,?)",
                ("B-104", pid, 50, date.today().isoformat(), "Open", now),
            )


def log_event(actor, event, record_type, record_id, reason=""):
    with connect() as c:
        c.execute(
            "INSERT INTO history(actor,event,record_type,record_id,reason,created_at) VALUES(?,?,?,?,?,?)",
            (
                actor,
                event,
                record_type,
                record_id,
                reason,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def current_user():
    return session.get("user")


def roles_required(*allowed):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not current_user():
                return redirect(url_for("login", next=request.path))
            if session.get("role") not in allowed:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return deco


@app.context_processor
def inject_user():
    return {
        "current_user": current_user(),
        "current_role": session.get("role"),
        "user_display_name": session.get("display_name", current_user()),
        "model_info": get_model_info(),
    }


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------
# High-Performance Cached Model & Manifest Handling
# ---------------------------------------------------------

_MODEL_CACHE = None
_MODEL_MTIME = None


def get_model_info():
    """Read manifest metadata for UI and audit logs."""
    if not MODEL_PATH.is_file():
        return {"status": "unloaded", "version": "None", "accuracy": None, "manifest": None}
    info = {
        "status": "ready",
        "version": "casting-hog-svm-v1",
        "file_size_kb": round(MODEL_PATH.stat().st_size / 1024, 1),
    }
    if MANIFEST_PATH.is_file():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            info["version"] = data.get("model_version", "casting-hog-svm-v1")
            info["accuracy"] = data.get("test_accuracy")
            info["train_count"] = data.get("train_count")
            info["test_count"] = data.get("test_count")
            info["trained_at"] = data.get("trained_at")
            info["classifier_type"] = data.get("classifier_type", "svm")
            info["manifest"] = data
        except Exception:
            pass
    return info


def get_loaded_model():
    """Thread-safe cached loader that auto-reloads when the joblib file updates."""
    global _MODEL_CACHE, _MODEL_MTIME
    if not MODEL_PATH.is_file():
        return None, MODEL_PATH
    try:
        current_mtime = MODEL_PATH.stat().st_mtime
        if _MODEL_CACHE is None or _MODEL_MTIME != current_mtime:
            app.logger.info("Loading model from disk: %s", MODEL_PATH)
            _MODEL_CACHE = joblib.load(MODEL_PATH)
            _MODEL_MTIME = current_mtime
        return _MODEL_CACHE, MODEL_PATH
    except Exception as exc:
        app.logger.exception("Failed to load model file %s: %s", MODEL_PATH, exc)
        return None, MODEL_PATH


def predict_image(path):
    """Predict quality class using standard RGB resize + HOG pipeline.

    Utilizes the cached model for sub-50ms inference.
    """
    model, model_path = get_loaded_model()
    if model is None:
        return None, None, "model-not-trained", 0

    start = time.perf_counter()
    try:
        image_path = Path(path).resolve()
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize((128, 128))
            arr = np.asarray(image)

        # Standard HOG feature extraction
        feature_vector = hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1,
        )
        features = feature_vector.reshape(1, -1)

        predicted_label = str(model.predict(features)[0])
        score = None
        probability_details = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = [str(lbl) for lbl in model.classes_]
            probability_details = {
                lbl: round(float(prob), 6)
                for lbl, prob in zip(classes, probabilities)
            }
            if predicted_label in classes:
                predicted_index = classes.index(predicted_label)
                score = float(probabilities[predicted_index])
            else:
                score = float(max(probabilities))

        ms = max(1, int((time.perf_counter() - start) * 1000))
        model_version = get_model_info().get("version", "casting-hog-svm-v1")

        app.logger.info(
            "PREDICTION: label=%s | score=%.4f | ms=%d | probs=%s",
            predicted_label,
            score or 0.0,
            ms,
            probability_details,
        )
        return predicted_label, score, model_version, ms

    except Exception as exc:
        app.logger.exception("Model prediction failed for %s: %s", path, exc)
        return None, None, "model-error", 0


# Pre-warm model cache in background or import
try:
    if MODEL_PATH.is_file():
        get_loaded_model()
except Exception:
    pass


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
@roles_required("inspector", "supervisor", "manager")
def api_predict():
    """Live interactive image prediction for drag-and-drop UI preview."""
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify({"error": "No image uploaded"}), 400
    if not allowed_file(upload.filename):
        return jsonify({"error": "Unsupported image format"}), 400

    ext = upload.filename.rsplit(".", 1)[-1].lower()
    temp_name = f"temp_preview_{time.time_ns()}.{ext}"
    temp_path = UPLOAD_DIR / temp_name
    try:
        upload.save(temp_path)
        pred, score, version, ms = predict_image(temp_path)
        return jsonify({
            "status": "success",
            "prediction": pred,
            "score": score,
            "confidence_percent": round(score * 100, 1) if score is not None else None,
            "model_version": version,
            "processing_ms": ms,
        })
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.route("/api/model-info")
def api_model_info():
    return jsonify(get_model_info())


# ---------------------------------------------------------
# Core Web Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    if not current_user():
        return redirect(url_for("login"))
    with connect() as c:
        stats = {
            "batches": c.execute("SELECT COUNT(*) n FROM batches").fetchone()["n"],
            "pending": c.execute(
                "SELECT COUNT(*) n FROM inspections WHERE status='Pending review'"
            ).fetchone()["n"],
            "defects": c.execute(
                "SELECT COUNT(*) n FROM inspections WHERE human_result='Defective'"
            ).fetchone()["n"],
            "actions": c.execute(
                "SELECT COUNT(*) n FROM actions WHERE status IN ('Open','In progress')"
            ).fetchone()["n"],
        }
        recent = c.execute("""
            SELECT i.*, b.batch_code, p.name as product_name
            FROM inspections i
            JOIN batches b ON b.id=i.batch_id
            JOIN products p ON p.id=b.product_id
            ORDER BY i.id DESC LIMIT 8
        """).fetchall()

        trend = c.execute("""
            SELECT substr(created_at,1,10) day,
                   SUM(CASE WHEN human_result='Defective' THEN 1 ELSE 0 END) defects,
                   COUNT(*) inspections
            FROM inspections
            GROUP BY day
            ORDER BY day DESC LIMIT 7
        """).fetchall()

    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        trend=list(reversed(trend)),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = DEMO_USERS.get(username)
        if user and user["password"] == password:
            session.clear()
            session["user"] = username
            session["role"] = user["role"]
            session["display_name"] = user.get("name", username.capitalize())
            log_event(username, "login", "user", 0)
            return redirect(url_for("index"))
        flash("Invalid demo username or password. Please select a quick login below.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/products", methods=["GET", "POST"])
@roles_required("inspector", "supervisor", "manager")
def products():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        category = request.form.get("category", "Metal casting").strip()
        if not name:
            flash("Product name is required.", "error")
        else:
            with connect() as c:
                cur = c.execute(
                    "INSERT INTO products(name,category,created_at) VALUES(?,?,?)",
                    (name, category, datetime.now().isoformat(timespec="seconds")),
                )
                pid = cur.lastrowid
            log_event(current_user(), "created product", "product", pid, name)
            flash(f"Product '{name}' registered successfully.", "success")
            return redirect(url_for("products"))

    with connect() as c:
        rows = c.execute("""
            SELECT p.*,
                   (SELECT COUNT(*) FROM batches b WHERE b.product_id=p.id) as batch_count
            FROM products p
            ORDER BY p.id DESC
        """).fetchall()
    return render_template("products.html", products=rows)


@app.route("/batches", methods=["GET", "POST"])
@roles_required("inspector", "supervisor", "manager")
def batches():
    if request.method == "POST":
        code = request.form.get("batch_code", "").strip()
        try:
            product_id = int(request.form["product_id"])
            qty = int(request.form["quantity"])
        except (ValueError, KeyError):
            product_id, qty = 0, 0
        prod_date = request.form.get("production_date") or date.today().isoformat()

        if not code or qty < 1:
            flash("Enter a unique batch code and quantity greater than zero.", "error")
        else:
            try:
                with connect() as c:
                    cur = c.execute(
                        """INSERT INTO batches(batch_code,product_id,quantity,production_date,updated_at)
                           VALUES(?,?,?,?,?)""",
                        (code, product_id, qty, prod_date, datetime.now().isoformat(timespec="seconds")),
                    )
                    bid = cur.lastrowid
                log_event(current_user(), "created batch", "batch", bid, code)
                flash(f"Batch {code} created successfully.", "success")
                return redirect(url_for("batches"))
            except sqlite3.IntegrityError:
                flash("Batch code must be unique and product must exist.", "error")

    with connect() as c:
        prods = c.execute("SELECT * FROM products WHERE archived=0 ORDER BY name").fetchall()
        rows = c.execute("""
            SELECT b.*, p.name product_name,
                   (SELECT COUNT(*) FROM inspections i WHERE i.batch_id=b.id) inspection_count,
                   (SELECT COUNT(*) FROM inspections i WHERE i.batch_id=b.id AND i.human_result='Defective') defect_count
            FROM batches b
            JOIN products p ON p.id=b.product_id
            ORDER BY b.id DESC
        """).fetchall()
    return render_template("batches.html", products=prods, batches=rows)


@app.route("/batches/<int:batch_id>/decision", methods=["POST"])
@roles_required("manager")
def batch_decision(batch_id):
    status = request.form.get("status")
    note = request.form.get("note", "").strip()
    if status not in {"Open", "Hold", "Released"} or not note:
        flash("Choose a valid decision and enter an auditable reason.", "error")
    else:
        with connect() as c:
            c.execute(
                "UPDATE batches SET status=?, decision_note=?, updated_at=? WHERE id=?",
                (status, note, datetime.now().isoformat(timespec="seconds"), batch_id),
            )
        log_event(current_user(), f"batch decision: {status}", "batch", batch_id, note)
        flash(f"Batch #{batch_id} decision updated to '{status}'.", "success")
    return redirect(url_for("batches"))


@app.route("/inspections", methods=["GET", "POST"])
@roles_required("inspector", "supervisor", "manager")
def inspections():
    if request.method == "POST":
        try:
            batch_id = int(request.form["batch_id"])
        except (ValueError, KeyError):
            batch_id = 0
        sample_ref = request.form.get("sample_ref", "").strip()
        observation = request.form.get("observation", "").strip()
        image_kind = request.form.get("image_kind", "Item")
        upload = request.files.get("image")

        if not batch_id or not sample_ref:
            flash("Select a batch and enter a sample reference.", "error")
            return redirect(url_for("inspections"))
        if image_kind not in {"Item", "Surface", "Batch sample"}:
            image_kind = "Item"

        saved_name = None
        pred = score = version = None
        ms = 0

        if upload and upload.filename:
            if not allowed_file(upload.filename):
                flash("Unsupported image format. Please use PNG, JPG, JPEG, or WEBP.", "error")
                return redirect(url_for("inspections"))

            clean_base = secure_filename(upload.filename)
            if not clean_base:
                ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else "jpg"
                clean_base = f"sample.{ext}"

            saved_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{clean_base}"
            path = UPLOAD_DIR / saved_name
            upload.save(path)
            pred, score, version, ms = predict_image(path)

        with connect() as c:
            cur = c.execute(
                """INSERT INTO inspections(batch_id,sample_ref,image_name,observation,image_kind,
                                           ml_prediction,ml_score,model_version,processing_ms,created_by,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    batch_id,
                    sample_ref,
                    saved_name,
                    observation,
                    image_kind,
                    pred,
                    score,
                    version,
                    ms,
                    current_user(),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            iid = cur.lastrowid

        log_event(
            current_user(),
            "created inspection",
            "inspection",
            iid,
            f"prediction={pred or 'model unavailable'}, sample={sample_ref}",
        )
        flash(
            f"Inspection #{iid} recorded. AI classification: {pred.upper() if pred else 'Unavailable (Manual Review Required)'}.",
            "success",
        )
        return redirect(url_for("inspection_detail", inspection_id=iid))

    with connect() as c:
        bs = c.execute("""
            SELECT b.*, p.name product_name
            FROM batches b
            JOIN products p ON p.id=b.product_id
            WHERE b.status != 'Archived'
            ORDER BY b.id DESC
        """).fetchall()
        rows = c.execute("""
            SELECT i.*, b.batch_code, p.name as product_name
            FROM inspections i
            JOIN batches b ON b.id=i.batch_id
            JOIN products p ON p.id=b.product_id
            ORDER BY i.id DESC
        """).fetchall()

    return render_template("inspections.html", batches=bs, inspections=rows)


@app.route("/uploads/<path:filename>")
@roles_required("inspector", "supervisor", "manager")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/inspections/<int:inspection_id>", methods=["GET", "POST"])
@roles_required("inspector", "supervisor", "manager")
def inspection_detail(inspection_id):
    with connect() as c:
        row = c.execute("""
            SELECT i.*, b.batch_code, p.name product_name, b.quantity as batch_quantity
            FROM inspections i
            JOIN batches b ON b.id=i.batch_id
            JOIN products p ON p.id=b.product_id
            WHERE i.id=?
        """, (inspection_id,)).fetchone()
        if not row:
            abort(404)
        actions = c.execute(
            "SELECT * FROM actions WHERE inspection_id=? ORDER BY id DESC",
            (inspection_id,),
        ).fetchall()
        history = c.execute(
            "SELECT * FROM history WHERE record_type='inspection' AND record_id=? ORDER BY id DESC",
            (inspection_id,),
        ).fetchall()

    if request.method == "POST":
        result = request.form.get("human_result")
        note = request.form.get("review_note", "").strip()
        if result not in {"Good", "Defective", "Uncertain", "Pending"}:
            flash("Invalid review result selection.", "error")
        else:
            if not note and result != "Pending":
                note = f"Disposition confirmed as {result} by {current_user()} ({session.get('role', 'user')})."
            new_status = "Pending review" if result in {"Pending", "Uncertain"} else "Reviewed"
            with connect() as c:
                c.execute(
                    "UPDATE inspections SET human_result=?, review_note=?, status=? WHERE id=?",
                    (result, note, new_status, inspection_id),
                )
            log_event(current_user(), f"reviewed as {result}", "inspection", inspection_id, note)
            flash(f"Human disposition successfully recorded as '{result}'.", "success")
            return redirect(url_for("inspection_detail", inspection_id=inspection_id))

    return render_template("inspection_detail.html", inspection=row, actions=actions, history=history)


@app.route("/inspections/<int:inspection_id>/actions", methods=["POST"])
@roles_required("supervisor", "manager")
def create_action(inspection_id):
    title = request.form.get("title", "").strip()
    owner = request.form.get("owner", "").strip()
    due_date = request.form.get("due_date", "").strip()
    notes = request.form.get("notes", "").strip()

    if not title or not owner or not due_date:
        flash("Action title, assigned owner, and due date are mandatory.", "error")
    else:
        with connect() as c:
            cur = c.execute(
                """INSERT INTO actions(inspection_id,title,owner,due_date,notes,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (inspection_id, title, owner, due_date, notes, datetime.now().isoformat(timespec="seconds")),
            )
            aid = cur.lastrowid
        log_event(current_user(), "created corrective action", "action", aid, title)
        flash(f"Corrective action #{aid} assigned to {owner}.", "success")
    return redirect(url_for("inspection_detail", inspection_id=inspection_id))


@app.route("/actions")
@roles_required("inspector", "supervisor", "manager")
def actions():
    with connect() as c:
        rows = c.execute("""
            SELECT a.*, i.sample_ref, i.human_result, i.image_name, b.batch_code
            FROM actions a
            JOIN inspections i ON i.id=a.inspection_id
            JOIN batches b ON b.id=i.batch_id
            ORDER BY a.id DESC
        """).fetchall()
    return render_template("actions.html", actions=rows)


@app.route("/actions/<int:action_id>/status", methods=["POST"])
@roles_required("supervisor", "manager")
def action_status(action_id):
    status = request.form.get("status")
    note = request.form.get("note", "").strip()
    if status not in {"Open", "In progress", "Completed", "Reopened"} or not note:
        flash("Choose a valid status and provide an update reason.", "error")
    else:
        timestamp_entry = f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ({current_user()}): {note}"
        with connect() as c:
            c.execute(
                "UPDATE actions SET status=?, notes=notes || ? WHERE id=?",
                (status, timestamp_entry, action_id),
            )
        log_event(current_user(), f"action status: {status}", "action", action_id, note)
        flash(f"Action #{action_id} status updated to {status}.", "success")
    return redirect(url_for("actions"))


@app.route("/history")
@roles_required("manager")
def history():
    with connect() as c:
        rows = c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 250").fetchall()
    return render_template("history.html", history=rows)


@app.route("/report", methods=["GET", "POST"])
@roles_required("inspector", "supervisor", "manager")
def report():
    batch_code = request.values.get("batch_code", "").strip()
    with connect() as c:
        batches = c.execute("SELECT batch_code FROM batches ORDER BY batch_code").fetchall()
        params = []
        where = ""
        if batch_code:
            where = "WHERE b.batch_code=?"
            params.append(batch_code)
        rows = c.execute(
            f"""SELECT i.id, i.sample_ref, i.human_result, i.ml_prediction, i.ml_score,
                       i.status, i.created_at, i.review_note, b.batch_code
                FROM inspections i
                JOIN batches b ON b.id=i.batch_id {where}
                ORDER BY i.id DESC""",
            params,
        ).fetchall()

        ids = [r["id"] for r in rows]
        actions_rows = []
        if ids:
            q = ",".join("?" for _ in ids)
            actions_rows = c.execute(
                f"""SELECT id, inspection_id, title, owner, due_date, status
                    FROM actions
                    WHERE inspection_id IN ({q})
                    ORDER BY id DESC""",
                ids,
            ).fetchall()

    summary = {
        "inspections": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "Pending review"),
        "confirmed_defects": sum(1 for r in rows if r["human_result"] == "Defective"),
        "open_actions": sum(1 for a in actions_rows if a["status"] in ("Open", "In progress", "Reopened")),
    }

    generated = None
    if request.method == "POST":
        generated = f"""QUALITY AUDIT & INSPECTION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Scope: {batch_code or 'All Production Batches'}
Auditor: {current_user()} ({session.get('role', 'staff')})

1. EXECUTIVE SUMMARY & EVIDENCE
   - Total Inspections Evaluated: {summary['inspections']}
   - Pending Quality Reviews: {summary['pending']}
   - Human-Confirmed Defects: {summary['confirmed_defects']}
   - Open Corrective Actions: {summary['open_actions']}

2. GROUNDED INSPECTION TRACEABILITY
   {', '.join('#'+str(r['id'])+' ['+r['batch_code']+': '+r['sample_ref']+' -> '+r['human_result']+']' for r in rows) or 'No matching inspection records found.'}

3. CORRECTIVE ACTIONS RECORD
   {chr(10).join('   - Action #'+str(a['id'])+': '+a['title']+' (Assigned: '+a['owner']+', Status: '+a['status']+', Ref: Inspection #'+str(a['inspection_id'])+')' for a in actions_rows) or '   - No corrective actions recorded for this scope.'}

4. GOVERNANCE & LIMITATIONS NOTE
   - Machine learning predictions serve as initial diagnostic screening suggestions, NOT autonomous release decisions.
   - All final dispositions require sign-off by qualified inspection personnel.
   - Single-sample observations do not unilaterally characterize upstream metallurgical properties."""
        log_event(current_user(), "generated evidence-based report", "report", 0, batch_code or "all batches")

    return render_template(
        "report.html",
        batches=batches,
        selected=batch_code,
        summary=summary,
        inspections=rows,
        actions=actions_rows,
        generated=generated,
    )


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="Your role does not have permission for this action."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="The requested record was not found."), 404


@app.errorhandler(413)
def too_large(e):
    flash(f"Image too large. Maximum upload size is {MAX_UPLOAD_MB} MB.", "error")
    return redirect(request.referrer or url_for("inspections"))


init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
