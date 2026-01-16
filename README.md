

# 🇮🇳 Bharat Pulse 2.1  
### AI-Powered Agricultural Price Intelligence for India

**Bharat Pulse** is a production-ready machine learning system designed to forecast short-term agricultural commodity prices in India. The platform predicts **7-day future modal prices** for essential crops such as **Tomato, Onion, and Potato (TOP)** at the **district level**, helping anticipate inflationary pressure before it reaches consumers.

🔗 **Live Application:**  
https://bharat-pulse-2-1.streamlit.app/

---

## 🎯 Why Bharat Pulse Exists

Agricultural price volatility in India directly affects:

- Household food security  
- Inflation metrics  
- Government procurement and logistics  
- Farmer and trader decision-making  

Traditional systems are **reactive** — prices are analyzed *after* spikes occur.  
**Bharat Pulse is predictive**, acting as an **early-warning intelligence layer** for agricultural markets.

---

## 🧠 What the System Does

- Predicts **future modal prices (₹/quintal)** up to **7 days ahead**
- Works across **multiple Indian states and districts**
- Handles real-world data issues such as:
  - Mixed date formats  
  - Missing records  
  - District–state inconsistencies  
- Provides **confidence-aware forecasts**, not just raw numbers

---

## 📊 Model & Data Overview

### Dataset
- **600,000+ historical records**
- Source: Indian agricultural market data (Agmarknet-derived)
- Time span: **2023–2025**

### Target Variable
- **Modal Price** (market-representative price)

### Feature Engineering
- Temporal features (day, month, lagged prices)
- Encoded State–District hierarchy
- Commodity-level behavior patterns

---

## 📈 Model Performance (Production Baseline)

- **R² Score:** ~0.68  
  → Explains ~68% of real-world price variance  

- **Mean Absolute Error (MAE):** ~₹561  
  → Acceptable range for high-value, volatile commodities  

- **Upward Trend Recall:** ~74%  
  → Strong reliability in detecting inflation spikes  

> These metrics prioritize **directional accuracy and robustness**, not lab-perfect scores.

---

## ⚙️ Technology Stack

### Machine Learning
- **Model:** Random Forest Regressor  
- **Library:** Scikit-learn  
- **Hyperparameter Tuning:** RandomizedSearchCV  
- **Model Persistence:** Joblib (`.pkl` files)

### Application Layer
- **Frontend & Runtime:** Streamlit  
- **Deployment:** Streamlit Community Cloud  
- **State–District Mapping:** Precomputed JSON for fast inference

---

## 📂 Production Repository Structure

Bharat-Pulse-2.1/
│
├── app.py # Streamlit application (entry point)
├── requirements.txt # Runtime dependencies
├── state_district_map.json # State–district mapping
├── models/
│ ├── price_model.pkl
│ ├── le_state.pkl
│ ├── le_district.pkl
│ └── le_commodity.pkl
└── .gitignore


> Training, evaluation, and experimentation scripts are intentionally excluded to keep the repository **deployment-focused and lightweight**.

---

## 🚀 Running Locally (Optional)

```bash
git clone https://github.com/sudrxn/Bharat-Pulse-2.1.git
cd Bharat-Pulse-2.1
pip install -r requirements.txt
streamlit run app.py

🌍 Deployment

The application is deployed on Streamlit Community Cloud, ensuring:

Zero infrastructure management

Automatic rebuilds on GitHub push

Public access for demonstrations and testing

🔮 Future Scope

Multi-commodity expansion beyond TOP crops

Longer forecast horizons with uncertainty bands

Integration with government dashboards

Farmer-facing simplified insights

Policy simulation tools for inflation control

👤 Author

Developed by Sudarshan Sharma
AI & Data Science Engineer

Focus: Applied ML systems, public-impact analytics, and decision intelligence.
