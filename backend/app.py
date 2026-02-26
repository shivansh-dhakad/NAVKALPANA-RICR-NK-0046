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
import os

db_config = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
    "port": int(os.environ.get("DB_PORT", 3306))
}

# ================= ROUTES =================
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("id", "").strip()
        password = request.form.get("password", "")

        if not user_id or not password:
            flash("Please enter both ID and password.", "error")
            return redirect(url_for("login"))

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM userInfo WHERE id = %s", (user_id,))
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["role"] = user["role"]

                flash("Login successful!", "success")

                if user["role"] == "clinic":
                    return redirect(url_for("bulk.dashboard"))
                elif user["role"] == "individual":
                    return redirect(url_for("individual.dashboard"))
            else:
                flash("Invalid ID or password.", "error")
                return redirect(url_for("login"))

        except mysql.connector.Error as db_error:
            print("DB ERROR:", db_error)
            return "Database connection failed. Check your DB config.", 500

        except Exception as e:
            print("Unexpected error:", e)
            return "Something went wrong.", 500

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user_id = request.form.get("id", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        role = request.form.get("role", "")

        # ===== Validation =====
        if not user_id or not password or not confirm or not role:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        if not re.match(r"^[A-Za-z0-9_]+$", user_id):
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
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute("SELECT id FROM userInfo WHERE id = %s", (user_id,))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                flash("User ID already exists. Choose another.", "error")
                return redirect(url_for("signup"))

            # Insert new user
            cursor.execute(
                "INSERT INTO userInfo (id, password, role) VALUES (%s, %s, %s)",
                (user_id, hashed_password, role)
            )
            conn.commit()

            cursor.close()
            conn.close()

            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))

        except mysql.connector.Error as db_error:
            print("DB ERROR:", db_error)
            return "Database connection failed. Check DB configuration.", 500

        except Exception as e:
            print("Unexpected error:", e)
            return "Something went wrong.", 500

    return render_template("signup.html")
    
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)

