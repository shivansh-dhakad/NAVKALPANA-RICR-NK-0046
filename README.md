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
1) login page
   <img width="1913" height="908" alt="login page" src="https://github.com/user-attachments/assets/1e1f539c-9fe3-45e4-adce-0e3fddf26239" />
2) signup page
   
   
4) individual-prediction (dark mode,light mode)
<img width="1896" height="1074" alt="individual-light" src="https://github.com/user-attachments/assets/a29d20d5-5407-414a-99af-e9717609a111" />
<img width="1896" height="1072" alt="individual-dark" src="https://github.com/user-attachments/assets/8b8bf9d2-4cc4-42c0-9a62-9bedab920af4" />
<img width="1897" height="1078" alt="individual-summary-dark" src="https://github.com/user-attachments/assets/8c99159e-158c-497a-8cbe-5a3746dded48" />


4) bulk-prediction
   <img width="1920" height="1080" alt="bulk-light" src="https://github.com/user-attachments/assets/c811e809-2b35-4589-9412-237a1ce53c0b" />
   <img width="1920" height="1080" alt="bulk summary" src="https://github.com/user-attachments/assets/705d958b-ebc4-41e1-8980-1173542f943a" />


# Future Improvements

- Hospital API Integration

- Wearable Device Data Integration

- Cloud Deployment (AWS / Render / Azure)

- Real-time Risk Monitoring

- Improved Model Calibration

- Clinical Dataset Expansion

# Conclusion
CardioShield AI demonstrates how ensemble machine learning can be leveraged for early cardiovascular risk prediction, enabling proactive intervention and potentially saving lives
