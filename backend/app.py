from flask import Flask, render_template, request, redirect, url_for, flash,session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import re
from backend.individual import individual_bp
from backend.bulk import bulk_bp

app = Flask(__name__, template_folder="../frontend/templates",
            static_folder="../frontend/static",
            static_url_path="/static")
CORS(app)
app.register_blueprint(individual_bp, url_prefix="/individual")
app.register_blueprint(bulk_bp, url_prefix="/clinic")  # Register clinic blueprint if needed


# Needed for flashing messages
app.secret_key = "supersecretkey"  # In production

# demo data
DEMO_USERS = {
    "shivansh": {
        "password": generate_password_hash("shivansh123"),
        "role": "clinic"
    },
    "aman": {
        "password": generate_password_hash("12345678"),
        "role": "individual"
    }
}

# ================= ROUTES =================
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("id", "").strip()
        password = request.form.get("password", "")

        if not user_id or not password:
            flash("Please enter both ID and password.", "error")
            return redirect(url_for("login"))

        # ===== DEMO AUTH =====
        user = DEMO_USERS.get(user_id)

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user_id
            session["role"] = user["role"]

            flash("Login successful!", "success")

            if user["role"] == "clinic":
                return redirect(url_for("bulk.dashboard"))
            elif user["role"] == "individual":
                return redirect(url_for("individual.dashboard"))
        else:
            flash("Invalid ID or password.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user_id = request.form.get("id", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        role = request.form.get("role", "")

        if not user_id or not password or not confirm or not role:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        if user_id in DEMO_USERS:
            flash("User already exists.", "error")
            return redirect(url_for("signup"))

        if role not in ["individual", "clinic"]:
            flash("Invalid role selected.", "error")
            return redirect(url_for("signup"))

        # Add to demo dictionary
        DEMO_USERS[user_id] = {
            "password": generate_password_hash(password),
            "role": role
        }

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")
    
if __name__ == "__main__":
    app.run(debug=True)



