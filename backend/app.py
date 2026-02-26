from flask import Flask, render_template, request, redirect, url_for, flash,session
from flask_cors import CORS
import mysql.connector
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

# ================= MySQL CONFIG =================
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Shivansh@1843",
    "database": "HeartRiskDatabase"
}

# ================= ROUTES =================
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form["id"].strip()
        password = request.form["password"]

        try:
            with mysql.connector.connect(**db_config) as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute("SELECT * FROM userInfo WHERE id = %s", (user_id,))
                    user = cursor.fetchone()

                    if user and check_password_hash(user["password"], password):
                        session["user_id"] = user["id"]
                        session["role"] = user["role"]

                        flash("Login successful!", "success")

                        if user["role"] == "clinic":
                            return redirect(url_for("bulk.dashboard"))
                        elif user["role"] == "individual":
                            return redirect(url_for("individual.dashboard"))

                    flash("Invalid ID or password.", "error")
                    return redirect(url_for("login"))

        except Exception as e:
            print("Database error:", e)
            flash("An unexpected error occurred.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user_id = request.form["id"].strip()
        password = request.form["password"]
        confirm = request.form["confirm"]
        role = request.form["role"]

        # Validation
        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))
        if not re.match("^[A-Za-z0-9_]+$", user_id):
            flash("ID can only contain letters, numbers, and underscores!", "error")
            return redirect(url_for("signup"))
        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return redirect(url_for("signup"))
        if role not in ["individual", "clinic"]:
            flash("Invalid role selected.", "error")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        try:
            with mysql.connector.connect(**db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM userInfo WHERE id = %s", (user_id,))
                    if cursor.fetchone():
                        flash("User ID already exists. Choose another.", "error")
                        return redirect(url_for("signup"))

                    cursor.execute(
                        "INSERT INTO userInfo (id, password, role) VALUES (%s, %s, %s)",
                        (user_id, hashed_password, role)
                    )
                    conn.commit()
                    flash("Account created successfully! Please login.", "success")
                    return redirect(url_for("login"))

        except Exception as e:
            # Log error internally, don’t expose details to user
            print(f"Database error: {e}")
            flash("An unexpected error occurred. Please try again.", "error")
            return redirect(url_for("signup"))

    return render_template("signup.html")

    
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
