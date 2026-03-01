# IntelliCharge360  
## Intelligent EV Charging Demand Prediction & Infrastructure Planning


## 1. Project Overview

IntelliCharge360 is a machine learning–driven platform designed to predict hourly EV charging demand using historical charging session data.

The objective of this project is to:

- Clean and process real-world EV charging session data
- Engineer time-series features
- Train and evaluate regression models
- Analyze demand trends
- Enable future integration with agentic infrastructure planning systems

This project is developed as part of a Capstone focused on intelligent EV infrastructure analytics.

---

## 2. Problem Statement

With the rapid adoption of electric vehicles, charging demand is becoming increasingly dynamic and location-dependent. Accurate forecasting of charging demand is critical for:

- Grid stability
- Infrastructure expansion planning
- Peak load management
- Efficient charger allocation

This system predicts hourly station-level charging demand and extracts insights for infrastructure optimization.

---

## 3. System Architecture

### System Design Diagram

The EV Charging Demand Prediction System follows a three-layered architecture:

![System Design Diagram](https://github.com/S-h-u-b-h-1/EV-charging-demand-prediction/assets/system_design.png)

**Architecture Components:**

- **Data Layer:** Handles raw charging data ingestion, preprocessing, cleaning, and hourly aggregation with persisted artifacts (trained models and scalers)
- **ML Pipeline:** Executes feature engineering, creates lag and rolling features using LightGBM and baseline models, and evaluates performance using RMSE, MAE, and R²
- **Serving Layer:** Delivers real-time inference through Streamlit dashboard, processes user input, applies feature transformations, and returns predictions

---

### Phase 1 – Predictive Demand Engine

Workflow:

Raw JSON Data  
→ Cleaning & Validation  
→ Hourly Aggregation  
→ Feature Engineering  
→ Lag & Rolling Features  
→ Chronological Train-Test Split  
→ Model Training & Evaluation  
→ Best Model Selection  
→ Model Serialization  

Technology Stack:
- Python
- Pandas & NumPy
- Scikit-Learn
- LightGBM
- Streamlit

---

### Phase 2 – (Planned) Agentic Infrastructure Planner

Future extension includes:

- High-load station detection
- Retrieval of infrastructure guidelines
- Agentic reasoning using LangGraph
- Structured infrastructure recommendations

---

## 4. Data Pipeline

### Raw Dataset
Original EV charging session data converted from JSON to CSV.

### Cleaning
- Removed null values in critical columns
- Removed invalid timestamps
- Removed negative or zero energy values
- Ensured logical session duration (disconnect > connection)

### Aggregation
Session-level data converted to hourly station-level demand.

### Feature Engineering
- Hour, Day of Week, Month, Week of Year
- Cyclical time encoding (sin/cos)
- Station encoding
- Lag features
- Rolling averages (3-hour and 24-hour)

---

## 5. Machine Learning Models

Models evaluated:

- Linear Regression (with Pipeline)
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM Regressor

Evaluation Metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Model selection is based on lowest RMSE using a strict chronological split to avoid data leakage.

---

## 6. Trend Analysis

The system performs:

- Hourly demand analysis
- Weekday vs weekend comparison
- Monthly demand trends
- High-load station identification
- Actual vs predicted comparison plots

---

## 7. Repository Structure
```
ev-demand-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── trained_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```
## How to Run Locally

streamlit run app/streamlit_app.py

## 8. Installation

Clone the repository:
git clone <repository-link>
cd ev-demand-prediction

Install dependencies:
pip install -r requirements.txt

Train the model:
python src/train.py

Run the Streamlit application:
streamlit run app/streamlit_app.py


---

## 9. Mid-Sem Deliverables Covered

- Data Cleaning and Preprocessing
- Time-Series Feature Engineering
- Multiple Regression Models
- Chronological Validation Strategy
- Model Evaluation and Comparison
- Trend Analysis
- Model Serialization
- Modular Codebase Structure

---

## 10. Future Scope

- Agentic infrastructure planning using LangGraph
- Charger placement optimization
- Grid load simulation
- Automated model retraining pipeline
- Deployment on cloud infrastructure

---

## 11. Disclaimer

This project is developed for academic purposes. The demand predictions are based on historical data and do not represent real-time operational grid decisions.
