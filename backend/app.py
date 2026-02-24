from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# ================= INIT =================
app = Flask(__name__, template_folder="../frontend/templates",
            static_folder="../frontend/static",
            static_url_path="/static")
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model")

# ================= LOAD MODELS =================
Xgb = joblib.load(os.path.join(model_path, "xgb.pkl"))
Lgbm = joblib.load(os.path.join(model_path, "lightbgm.pkl"))
tabnet = joblib.load(os.path.join(model_path, "tabnet.pkl"))
stack = joblib.load(os.path.join(model_path, "stack.pkl"))
preprocessor = joblib.load(os.path.join(model_path, "preprocessor.pkl"))
# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= FEATURE ENGINEERING =================

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity1"
    elif bmi < 40:
        return "Obesity2"
    else:
        return "Obesity3"



def pulse_pressure_category(pp):
    if pp < 30:
        return "Low"
    elif pp < 50:
        return "Normal"
    elif pp < 70:
        return "Elevated"
    else:
        return "High"


def compute_simple_risk(df):
    df = df.copy()

    df['cholesterol_score'] = df['cholesterol'].map({1:0, 2:0.5, 3:1})
    df['gluc_score'] = df['gluc'].map({1:0, 2:0.5, 3:1})
    df['smoke_score'] = df['smoke'].astype(float)

    # Use realistic max constants (IMPORTANT — do NOT use df.max() in production)
    df['age_norm'] = df['age'] / 100
    df['ap_hi_norm'] = df['ap_hi'] / 200
    df['Bmi_norm'] = df['Bmi'] / 50

    df['simple_risk_index'] = (
        0.3 * df['age_norm'] +
        0.25 * df['ap_hi_norm'] +
        0.15 * df['Bmi_norm'] +
        0.1 * df['cholesterol_score'] +
        0.1 * df['gluc_score'] +
        0.1 * df['smoke_score']
    )

    return df

def generate_explanation(row):

    factors = []

    if row["pulse_pressure_cat"] in ["Elevated", "High"]:
        factors.append("elevated pulse pressure")

    if row["cholesterol"] == 3:
        factors.append("high cholesterol levels")
    elif row["cholesterol"] == 2:
        factors.append("above-normal cholesterol levels")

    if row["gluc"] == 3:
        factors.append("high glucose levels")
    elif row["gluc"] == 2:
        factors.append("elevated glucose levels")

    if row["Bmi_cat"] in ["Obesity1", "Obesity2", "Obesity3"]:
        factors.append("obesity-range BMI")
    elif row["Bmi_cat"] == "Overweight":
        factors.append("overweight BMI")

    if row["smoke"] == 1:
        factors.append("smoking habit")

    if row["alco"] == 1:
        factors.append("regular alcohol consumption")

    if row["age"] > 55:
        factors.append("advanced age")

    if row["ap_hi"] > 150:
        factors.append("high systolic blood pressure")

    # ===== Build Clinical Statement =====
    if not factors:
        return "The model assessed a low cardiovascular risk as all major clinical indicators fall within normal ranges."

    if len(factors) == 1:
        return f"The predicted cardiovascular risk is influenced primarily by {factors[0]}."

    else:
        main_text = ", ".join(factors[:-1])
        last_factor = factors[-1]
        return f"The predicted cardiovascular risk is influenced by {main_text}, and {last_factor}."
# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        # Raw input
        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        ap_hi = float(data["ap_hi"])
        ap_lo = float(data["ap_lo"])

        cholesterol = int(data["cholesterol"])
        gluc = int(data["gluc"])
        smoke = int(data["smoke"])
        alco = int(data.get("alco", 0))

        # ===== Basic Calculations =====
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo

        df = pd.DataFrame([{
            "age": age,
            "ap_hi": ap_hi,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "Bmi": bmi,
            "pulse_pressure": pulse_pressure
        }])

        # ===== Categories =====
        df["Bmi_cat"] = df["Bmi"].apply(bmi_category)
        
        df["pulse_pressure_cat"] = df["pulse_pressure"].apply(pulse_pressure_category)

        # ===== Interaction Features =====
        df["age_bp_inter"] = df["age"] * df["ap_hi"]
        df["gluc_bmi_inter"] = df["gluc"] * df["Bmi"]

        # ===== Risk Indices =====
        df = compute_simple_risk(df)
        explanations = generate_explanation(df.iloc[0])
        print(explanations)
        # ===== Keep ONLY training features =====
        feature_cols = [
            "gluc",
            "ap_hi",
            'cholesterol',
            'pulse_pressure',
            'Bmi_cat',
            'age_bp_inter',
            'gluc_bmi_inter',
            'simple_risk_index'
        ]

        input = df[feature_cols]
        model_input = preprocessor.transform(input)
        print("Model Input:\n", model_input)
        # ===== Model Predictions =====
        print("Running XGBoost prediction...")
        xgb_prob = float(Xgb.predict_proba(input)[0][1]) * 100
        print("XGBoost Probability:", xgb_prob)
        print("Running lightgbm prediction...")
        lgbm_prob = float(Lgbm.predict_proba(pd.DataFrame(input, columns=Lgbm.feature_names_in_))[0][1]) * 100
        print("Running tabnet prediction...")
        X_test_np = model_input.values if hasattr(model_input, "values") else model_input
        proba = tabnet.predict_proba(X_test_np)
        tabnet_prob = float(proba[0][1]) * 100
        print("Running stack prediction...")
        stack_prob = float(stack.predict_proba(input)[0][1]) * 100

        def shrink_towards_mid(x, factor=0.9):
            return 50 + factor * (x - 50)

        xgb_prob = shrink_towards_mid(xgb_prob)
        lgbm_prob = shrink_towards_mid(lgbm_prob)
        tabnet_prob = shrink_towards_mid(tabnet_prob)
        stack_prob = shrink_towards_mid(stack_prob)
        
        final_risk = round((xgb_prob + lgbm_prob + tabnet_prob + stack_prob) / 4,2)

        if final_risk < 30:
            category = "Low Risk"
        elif final_risk < 60:
            category = "Moderate Risk"
        else:
            category = "High Risk"
            
        return jsonify({
            "final_risk": final_risk,
            "category": category,
            "xgb": round(xgb_prob,2),
            "lgbm": round(lgbm_prob,2),
            "tabnet": round(tabnet_prob,2),
            "stack": round(stack_prob,2),
            "reasons": explanations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================= BULK PREDICT =================
@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # ===== Read File =====
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only CSV and Excel allowed"}), 400

        # ===== Required Columns =====
        required_cols = [
            "id", "age", "gender", "weight", "height",
            "ap_hi", "ap_lo", "cholesterol", "smoke", "alco", "gluc"
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # ===== Type Conversion =====
        numeric_cols = [
            "age","weight","height","ap_hi","ap_lo",
            "cholesterol","smoke","alco","gluc"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)

        # ===== Basic Calculations =====
        df["Bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

        # ===== Categories =====
        df["Bmi_cat"] = df["Bmi"].apply(bmi_category)
        df["pulse_pressure_cat"] = df["pulse_pressure"].apply(pulse_pressure_category)

        # ===== Interaction Features =====
        df["age_bp_inter"] = df["age"] * df["ap_hi"]
        df["gluc_bmi_inter"] = df["gluc"] * df["Bmi"]

        # ===== Risk Indices =====
        df = compute_simple_risk(df)

        # ===== Keep Training Features =====
        feature_cols = [
            "gluc",
            "ap_hi",
            'cholesterol',
            'pulse_pressure',
            'Bmi_cat',
            'age_bp_inter',
            'gluc_bmi_inter',
            'simple_risk_index',
        ]

        model_df = df[feature_cols]

        # ===== Preprocess =====
        model_input = preprocessor.transform(model_df)

        # ===== Predictions =====
        xgb_prob = Xgb.predict_proba(model_df)[:, 1] * 100
        lgbm_prob = Lgbm.predict_proba(model_df)[:, 1] * 100
        stack_prob = stack.predict_proba(model_df)[:, 1] * 100

        # TabNet is NOT pipeline → use transformed numpy
        X_test_np = model_input.values if hasattr(model_input, "values") else model_input
        tabnet_prob = tabnet.predict_proba(X_test_np)[:, 1] * 100
        df["xgb"] = np.round(xgb_prob, 2)
        df["lgbm"] = np.round(lgbm_prob, 2)
        df["tabnet"] = np.round(tabnet_prob, 2)
        df["stack"] = np.round(stack_prob, 2)
        # Final average (same logic as single)
        # ===== Confidence Shrinkage Calibration =====
        def shrink_towards_mid(x, factor=0.9):
            return 50 + factor * (x - 50)

        risk_cols = ["xgb", "lgbm", "tabnet", "stack"]

        for col in risk_cols:
            df[col] = df[col].apply(lambda x: round(shrink_towards_mid(x), 2))
        df["final_risk"] = np.round(
            (df["xgb"] + df["lgbm"] + df["tabnet"] + df["stack"]) / 4,
            2
        )      
        df["reasons"] = df.apply(generate_explanation, axis=1)
        # ===== Risk Category =====
        def classify(r):
            if r < 30:
                return "Low Risk"
            elif r < 60:
                return "Moderate Risk"
            else:
                return "High Risk"

        df["category"] = df["final_risk"].apply(classify)

        # ===== Build JSON Output =====
        predictions = []

        for _, row in df.iterrows():
            predictions.append({
                "id": row["id"],
                "xgb": row["xgb"],
                "lgbm": row["lgbm"],
                "tabnet": row["tabnet"],
                "stack": row["stack"],
                "final_risk": row["final_risk"],
                "category": row["category"],
                "reasons": row["reasons"]
            })

        return jsonify({
            "total_records": len(predictions),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == "__main__":
    app.run(debug=True)