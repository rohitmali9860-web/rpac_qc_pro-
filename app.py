from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os, json, fitz, pdfplumber, cv2, numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

USERS = {
    "rohit": {"password": "admin123", "role": "admin", "name": "Rohit Master"},
    "operator": {"password": "user123", "role": "user", "name": "Operator User"}
}


def current_user():
    if "username" in session:
        username = session["username"]
        return {
            "username": username,
            "role": session["role"],
            "name": session["name"]
        }
    return None


def login_required():
    return current_user() is not None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_image(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(6, 6), alpha=False)
    pix.save(output_path)
    doc.close()


def create_overlay(original_img, revised_img, overlay_path):
    original = cv2.imread(original_img)
    revised = cv2.imread(revised_img)
    revised = cv2.resize(revised, (original.shape[1], original.shape[0]))

    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_revised = cv2.cvtColor(revised, cv2.COLOR_BGR2GRAY)

    overlay = np.ones_like(original) * 255
    overlay[gray_original < 245] = [0, 0, 255]
    overlay[gray_revised < 245] = [0, 0, 0]

    diff = cv2.absdiff(gray_original, gray_revised)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((6, 6), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1800 and w > 35 and h > 18:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    issues = []

    for index, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 5)

        label_y1 = max(0, y - 45)
        label_y2 = y if y > 45 else y + 45

        cv2.rectangle(overlay, (x, label_y1), (x + 70, label_y2), (0, 0, 255), -1)
        cv2.putText(overlay, str(index), (x + 18, label_y2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        issues.append({
            "no": index,
            "x": int(x),
            "y": int(y),
            "issue": "Visual difference detected",
            "status": "Changed"
        })

    cv2.imwrite(overlay_path, overlay)
    return len(issues), issues


def extract_text(pdf_path):
    extracted = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for word in page.extract_words():
                extracted.append({
                    "text": word["text"],
                    "x0": word["x0"],
                    "top": word["top"],
                    "page": page_num
                })
    return extracted


def compare_text_details(original_words, revised_words):
    mismatches = []
    original_texts = [w["text"] for w in original_words]
    revised_texts = [w["text"] for w in revised_words]

    max_len = max(len(original_texts), len(revised_texts))

    for i in range(max_len):
        original_value = original_texts[i] if i < len(original_texts) else "EMPTY"
        revised_value = revised_texts[i] if i < len(revised_texts) else "EMPTY"

        if original_value != revised_value:
            mismatches.append({
                "no": len(mismatches) + 1,
                "original": original_value,
                "revised": revised_value,
                "issue": "Text mismatch detected",
                "status": "Changed"
            })

    total_words = len(original_texts)
    text_match = ((total_words - len(mismatches)) / total_words * 100) if total_words else 100
    return round(text_match, 2), mismatches[:80]


def run_qc_analysis(original_path, revised_path, timestamp):
    original_img = f"{app.config['RESULT_FOLDER']}/original_{timestamp}.png"
    revised_img = f"{app.config['RESULT_FOLDER']}/revised_{timestamp}.png"
    overlay_img = f"{app.config['RESULT_FOLDER']}/overlay_{timestamp}.png"

    pdf_to_image(original_path, original_img)
    pdf_to_image(revised_path, revised_img)

    visual_changes, visual_issues = create_overlay(original_img, revised_img, overlay_img)

    original_words = extract_text(original_path)
    revised_words = extract_text(revised_path)

    text_match, text_mismatches = compare_text_details(original_words, revised_words)

    visual_score = max(0, 100 - visual_changes)
    overall_match = round((visual_score + text_match) / 2, 2)

    status = "Approved" if overall_match >= 95 else "Review Required"

    return {
        "overall_match_percentage": overall_match,
        "text_match_percentage": text_match,
        "visual_changes": visual_changes,
        "visual_issues": visual_issues,
        "text_mismatches": text_mismatches,
        "status": status,
        "original_image": "/" + original_img,
        "revised_image": "/" + revised_img,
        "overlay_image": "/" + overlay_img,
    }


def get_reports():
    user = current_user()
    reports = []

    if not os.path.exists(app.config["RESULT_FOLDER"]):
        return reports

    for file in os.listdir(app.config["RESULT_FOLDER"]):
        if file.startswith("result_") and file.endswith(".json"):
            path = os.path.join(app.config["RESULT_FOLDER"], file)
            with open(path, "r") as f:
                data = json.load(f)

            if user["role"] == "admin" or data.get("uploaded_by") == user["username"]:
                reports.append(data)

    return sorted(reports, key=lambda x: x.get("timestamp", ""), reverse=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USERS and USERS[username]["password"] == password:
            session["username"] = username
            session["role"] = USERS[username]["role"]
            session["name"] = USERS[username]["name"]
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    reports = get_reports()
    user = current_user()
    return render_template("dashboard.html", reports=reports, user=user)


@app.route("/compare")
def compare():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("upload.html", user=current_user())


@app.route("/history")
def history():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("history.html", reports=get_reports(), user=current_user())


@app.route("/sop-rules")
def sop_rules():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("sop_rules.html", user=current_user())


@app.route("/upload", methods=["POST"])
def upload_files():
    if not login_required():
        return jsonify({"error": "Login required"}), 401

    if "original" not in request.files or "revised" not in request.files:
        return jsonify({"error": "Please upload both PDF files"}), 400

    original_file = request.files["original"]
    revised_file = request.files["revised"]

    if original_file.filename == "" or revised_file.filename == "":
        return jsonify({"error": "Please select both files"}), 400

    if not allowed_file(original_file.filename) or not allowed_file(revised_file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = secure_filename(f"original_{timestamp}_{original_file.filename}")
    revised_filename = secure_filename(f"revised_{timestamp}_{revised_file.filename}")

    original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
    revised_path = os.path.join(app.config["UPLOAD_FOLDER"], revised_filename)

    original_file.save(original_path)
    revised_file.save(revised_path)

    try:
        results = run_qc_analysis(original_path, revised_path, timestamp)

        user = current_user()

        results["original_filename"] = original_file.filename
        results["revised_filename"] = revised_file.filename
        results["timestamp"] = timestamp
        results["uploaded_by"] = user["username"]
        results["uploaded_by_name"] = user["name"]
        results["user_role"] = user["role"]

        result_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{timestamp}.json")

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        return jsonify({"success": True, "result_id": timestamp})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result/<result_id>")
def show_result(result_id):
    if not login_required():
        return redirect(url_for("login"))

    result_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{result_id}.json")

    if not os.path.exists(result_path):
        return "Result not found", 404

    with open(result_path, "r") as f:
        results = json.load(f)

    user = current_user()

    if user["role"] != "admin" and results.get("uploaded_by") != user["username"]:
        return "Access denied", 403

    return render_template("result.html", results=results, user=user)


if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)