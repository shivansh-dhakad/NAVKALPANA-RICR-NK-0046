from flask import Blueprint, request, jsonify,render_template
import pandas as pd
import numpy as np
import shap
import joblib
import os

# Create Blueprint
individual_bp = Blueprint("individual", __name__)


# loading model and preprocessor
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model")
stack = joblib.load(os.path.join(model_path, "stack.pkl"))
pipeline_model = stack.calibrated_classifiers_[0].estimator
preprocessor = pipeline_model.named_steps['preprocess']

stacking_model = pipeline_model.named_steps['model']
xgb_model = stacking_model.named_estimators_["xgb"]

@individual_bp.route("/dashboard")
def dashboard():
    return render_template("single_prediction.html")


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


def generate_explanation(row):
    factors = []

    if row["pulse_pressure_cat"] in ["Elevated", "High"]:
        factors.append("elevated pulse pressure indicating increased arterial stiffness")

    if row["cholesterol"] == 3:
        factors.append("high cholesterol levels contributing to plaque formation")
    elif row["cholesterol"] == 2:
        factors.append("above-normal cholesterol levels")

    if row["gluc"] == 3:
        factors.append("high glucose levels affecting vascular health")
    elif row["gluc"] == 2:
        factors.append("elevated glucose levels")

    if row["Bmi_cat"] in ["Obesity1", "Obesity2", "Obesity3"]:
        factors.append("obesity-range BMI associated with metabolic strain")
    elif row["Bmi_cat"] == "Overweight":
        factors.append("overweight BMI increasing cardiovascular workload")

    if row["smoke"] == 1:
        factors.append("smoking habit that damages blood vessels")

    if row["alco"] == 1:
        factors.append("regular alcohol consumption impacting heart function")

    if row["age"] > 55:
        factors.append("advanced age, which naturally increases cardiovascular risk")

    if row["ap_hi"] > 150:
        factors.append("high systolic blood pressure stressing the heart")

    if not factors:
        return "The model indicates low cardiovascular risk, as key clinical and lifestyle indicators remain within healthy ranges."

    if len(factors) == 1:
        return f"The predicted cardiovascular risk is mainly driven by {factors[0]}."

    main_text = ", ".join(factors[:-1])
    last_factor = factors[-1]

    return f"The predicted cardiovascular risk is influenced by {main_text}, and {last_factor}."

import pymysql
import os


# ================= SINGLE PREDICT ROUTE =================

@individual_bp.route("/predict", methods=["POST"])
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
            "ap_lo": ap_lo,
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

        model_input = df[feature_cols]
        # ===== Model Predictions =====
        print("Running  prediction...")
        stack_prob = float(stack.predict_proba(model_input)[0][1]) * 100

        def shrink_towards_mid(x, factor=0.9):
            return 50 + factor * (x - 50)
        
        stack_prob = shrink_towards_mid(stack_prob)
        
        final_risk = round(stack_prob, 2)

        p = stack_prob / 100  # convert to 0–1

        if p < 0.42:
            category = "Low Risk"
            suggestions = "Maintain a healthy lifestyle with regular exercise and a balanced diet. Monitor blood pressure and cholesterol levels regularly."
        elif p < 0.65:
            category = "Moderate Risk"
            suggestions = "Consider lifestyle modifications such as increased physical activity, dietary changes, and regular health check-ups. Consult with a healthcare provider for personalized advice."
        else:
            category = "High Risk"
            suggestions = "Seek medical evaluation promptly. Implement aggressive lifestyle changes and discuss potential medical interventions with a healthcare provider."
                
    
            
        X_processed = preprocessor.transform(model_input)
        feature_names = preprocessor.get_feature_names_out()    
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_processed)
        shap_importance = np.abs(shap_values.values[0])
        
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
        
        return jsonify({
            "final_risk": final_risk,
            "category": category,
            "reasons": explanations,
            "suggestions": suggestions,
            "top_features": top_5_list
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 400

