# NAVKALPANA-RICR-NK-0046
# CardioShield AI
AI-Powered Early Cardiovascular Risk Detection

Hackathon: NavKalpana AI Innovation Challenge
Team: NavKalpana-RICR-NK-0046

# Project Title

CardioShield AI – Early Detection. Smarter Prevention. Healthier Lives.

An AI-powered web application designed to predict early cardiovascular disease risk using machine learning and ensemble modeling techniques.

# Team Members & Roles
1) Shivansh Dhakad	- Team Lead & Backend Development
2) Aman Kumar Pandey - Data Analysis & Preprocessing
3) Sonu Yadav - Model Building & SHAP Analysis
4) Tanishka Sharma - Frontend Development & Real-World Analysis

# Problem Statement
Cardiovascular diseases are the leading cause of global mortality.
Traditional screening methods are:
- Expensive
- Resource-intensive
- Reactive instead of proactive
There is a need for a low-cost, scalable AI-based solution that enables:
- Early detection
- Risk stratification
- Preventive healthcare intervention
CardioShield AI aims to provide a proactive cardiovascular risk prediction system using structured health data and ensemble machine learning models.

# Tech Stack Used
1) Programming & Backend
- Python
- Flask (REST API)
2) Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- Neural Networks (TabNet)
- Stacking / Ensemble Learning
3) Data Processing
- Pandas
- NumPy
- Feature Engineering
- Normalization & Encoding
4) Frontend
- HTML
- CSS
- JavaScript

# Installation Steps
1) Clone the Repository
   git clone https://github.com/your-username/cardioshield-ai.git
   cd cardioshield-ai
2) Create Virtual Environment
   python -m venv venv
   venv\Scripts\activate   # Windows
3) Install Dependencies
   pip install -r requirements.txt
4) Run the Application
   python backend/app.py

# API Endpoints
1) Predict Risk:
  POST /predict
2) Request Body (JSON):
   {
  "age": 52,
  "gender": 1,
  "height": 170,
  "weight": 85,
  "ap_hi": 140,
  "ap_lo": 90,
  "cholesterol": 2,
  "gluc": 2,
  "smoke": 0,
  "alco": 1,
  "active": 0
}
3) Response:
   {
  "risk_probability": 85.03,
  "risk_category": "High Risk",
  "explanation": "Elevated BP, cholesterol and glucose detected."
}

# Screenshots
1) landing page (dark mode,light mode)
 <img width="2880" height="1800" alt="Screenshot 2026-02-24 234357" src="https://github.com/user-attachments/assets/0f80c052-e064-44ae-86b8-39b6b0f18a0b" />
<img width="2880" height="1800" alt="Screenshot 2026-02-24 234413" src="https://github.com/user-attachments/assets/fe4c0367-7f5f-4695-85f2-1e3a07a854ad" />

2) Risk prediction
<img width="2880" height="1800" alt="Screenshot 2026-02-24 235251" src="https://github.com/user-attachments/assets/47d95775-dffc-4b10-b384-c112595fc6eb" />

3) model summary
<img width="2880" height="1800" alt="Screenshot 2026-02-24 235451" src="https://github.com/user-attachments/assets/72cacaef-249b-4471-8b27-438cc56de150" />

# Future Improvements

- Hospital API Integration

- Wearable Device Data Integration

- Mobile App Version

- Cloud Deployment (AWS / Render / Azure)

- Real-time Risk Monitoring

- Improved Model Calibration

- Clinical Dataset Expansion

# Conclusion
CardioShield AI demonstrates how ensemble machine learning can be leveraged for early cardiovascular risk prediction, enabling proactive intervention and potentially saving lives
