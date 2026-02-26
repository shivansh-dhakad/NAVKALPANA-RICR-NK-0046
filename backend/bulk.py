import os
import joblib
import pandas as pd
import numpy as np
import shap
from flask import Blueprint, request, jsonify,render_template

# ================= CREATE BLUEPRINT =================
bulk_bp = Blueprint("bulk", __name__)

# ================= LOAD MODELS =================
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model")
stack = joblib.load(os.path.join(model_path, "stack.pkl"))
pipeline_model = stack.calibrated_classifiers_[0].estimator
preprocessor = pipeline_model.named_steps['preprocess']
stacking_model = pipeline_model.named_steps['model']
xgb_model = stacking_model.named_estimators_["xgb"]

print("Bulk model loaded successfully ✅")


@bulk_bp.route("/")
def dashboard():
    return render_template("bulk_prediction.html")
    
# ================= FEATURE FUNCTIONS =================

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

def generate_clinical_summary(user_row):

    explanations = []
    # age
    if user_row["age"] > 55:
        explanations.append("Age above 55 (higher cardiac risk).")
    else:
        explanations.append("Age within lower-risk range.")
    #ap_hi
    if user_row['ap_hi'] > 180:
        explanations.append("The patient has severely elevated systolic blood pressure")
    elif user_row['ap_hi'] > 150:
        explanations.append("The patient has high systolic blood pressure")
    elif user_row['ap_hi'] > 120:
        explanations.append("The patient has elevated systolic blood pressure")
    else:
        explanations.append("The patient's systolic blood pressure is within a normal range")
    #ap_lo
    if user_row['ap_lo'] > 100:
        explanations.append("The patient has elevated diastolic blood pressure")
    elif user_row['ap_lo'] > 80:
        explanations.append("The patient has high diastolic blood pressure")
    else:
        explanations.append("The patient's diastolic blood pressure is within a normal range")
    #bmi
    if user_row['Bmi_cat'] in ["Obesity1", "Obesity2", "Obesity3"]:
        explanations.append("The patient falls into the obesity category")
    elif user_row['Bmi_cat'] == "Overweight":
        explanations.append("The patient is classified as overweight ")
    else:
        explanations.append("The patient's BMI is within a normal range")
    # cholesterol
    if user_row['cholesterol'] == 3:
        explanations.append("The patient has high cholesterol levels")
    elif user_row['cholesterol'] == 2:
        explanations.append("The patient has above-normal cholesterol levels")
    else:
        explanations.append("The patient's cholesterol levels are within a normal range")
    # glucose
    if user_row['gluc'] == 3:
        explanations.append("The patient has high glucose levels")
    elif user_row['gluc'] == 2:
        explanations.append("The patient has elevated glucose levels")
    else:
        explanations.append("The patient's glucose levels are within a normal range")
# smoking
    if user_row['smoke'] == 1:
        explanations.append("The patient has a smoking habit")
    else:
        explanations.append("The patient does not have a smoking habit")
# alcohol
    if user_row['alco'] == 1:   
        explanations.append("The patient has regular alcohol consumption")
    else:
        explanations.append("The patient does not have regular alcohol consumption")
        
    return explanations

import pymysql

def save_bulk_to_db(df):
    conn = pymysql.connect(
        host=os.environ.get("DB_HOST"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        database=os.environ.get("DB_NAME"),
        port=int(os.environ.get("DB_PORT", 3306))
    )

    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO bulk_predictions
            (id, age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, bmi, pulse_pressure,
             Bmi_cat, pulse_pressure_cat, age_bp_inter, gluc_bmi_inter, simple_risk_index, cardio)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            for _, row in df.iterrows():
                cardio = 1 if row["final_risk"] > 44 else 0

                cursor.execute(sql, (
                    row["id"],
                    row["age"], row["height"], row["weight"],
                    row["ap_hi"], row["ap_lo"],
                    row["cholesterol"], row["gluc"],
                    row["smoke"], row["alco"],
                    row["Bmi"], row["pulse_pressure"],
                    row["Bmi_cat"], row["pulse_pressure_cat"],
                    row["age_bp_inter"], row["gluc_bmi_inter"],
                    row["simple_risk_index"], cardio
                ))

            conn.commit()

    finally:
        conn.close()

# ================= BULK PREDICTION ROUTE =================

@bulk_bp.route("/predict", methods=["POST"])
def predict():

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

        model_input = df[feature_cols]
               

        # ===== Predictions =====
        stack_prob = stack.predict_proba(model_input)[:, 1] * 100
        df['stack'] = stack_prob
        def shrink_towards_mid(x, factor=0.9):
            return 50 + factor * (x - 50)


        df['final_risk'] = df['stack'].apply(lambda x: round(shrink_towards_mid(x), 2))
        df["clinical_summary"] = df.apply(generate_clinical_summary, axis=1)
        # ===== Risk Category =====
        def classify(r):
            if r < 30:
                return "Low Risk"
            elif r < 60:
                return "Moderate Risk"
            else:
                return "High Risk"
        def suggest(r):
            if r < 30:
                return "Maintain a healthy lifestyle with regular exercise and a balanced diet. Monitor blood pressure and cholesterol levels regularly."
            elif r < 60:
                return "Consider lifestyle modifications such as increased physical activity, dietary changes, and regular health check-ups. Consult with a healthcare provider for personalized advice."
            else:
                return "Seek medical evaluation promptly. Implement aggressive lifestyle changes and discuss potential medical interventions with a healthcare provider."
        
        df["category"] = df["final_risk"].apply(classify)
        df["suggestion"] = df["final_risk"].apply(suggest)

        df_sample =df.sample(min(8, len(df)))
        # ===== SHAP for Bulk =====
        sample_model_input = df[feature_cols]
        X_processed = preprocessor.transform(sample_model_input)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_processed)

        top_features_all = []

        for i in range(len(df)):  
            shap_importance = np.abs(shap_values.values[i])

            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": shap_importance
            }).sort_values(by="importance", ascending=False)

            top_5 = importance_df.head(5)

            top_5_list = []
            for _, row in top_5.iterrows():
                top_5_list.append({
                    "feature": row["feature"],
                    "importance": float(row["importance"])
                })

            top_features_all.append(top_5_list)

        df["top_features"] = top_features_all
        # ===== Build JSON Output =====
        predictions = []
        for idx, row in df.iterrows():
            predictions.append({
            "id": row["id"],
            "final_risk": row["final_risk"],
            "category": row["category"],
            "suggestion": row["suggestion"],
            "clinical_summary": row["clinical_summary"],
            "top_features": row["top_features"],
            "original_data": {
                "Age": row["age"],
                "Systolic BP": row["ap_hi"],
                "Diastolic BP": row["ap_lo"],
                "BMI Category": row["Bmi_cat"],
                "Cholesterol": row["cholesterol"],
                "Glucose": row["gluc"],
                "Smoking": row["smoke"],
                "Alcohol": row["alco"]
            }
        })
        save_bulk_to_db(df)

        return jsonify({
            "total_records": len(predictions),
            "predictions": predictions
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 400
