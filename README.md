<div align="center">

### Intelligent EV Charging Demand Prediction

[![Milestone](https://img.shields.io/badge/Milestone-1%20%7C%20Mid--Sem-4CAF50?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![ML Only](https://img.shields.io/badge/Scope-Traditional%20ML%20Only-2196F3?style=for-the-badge&logo=scikit-learn&logoColor=white)](.)
[![No LLMs](https://img.shields.io/badge/LLMs-NOT%20USED-FF5722?style=for-the-badge&logo=openai&logoColor=white)](.)
[![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=black)](.)
[![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)](.)
[![Team](https://img.shields.io/badge/Team-RASS-00BCD4?style=for-the-badge&logo=github&logoColor=white)](.)

<br/>

> **Station-level · Hourly Granularity · Chronologically Validated**
>
> Predicting EV charging demand from real session data using engineered temporal features and ensemble tree-based regressors — no black-box LLMs, no shortcuts.

<br/>

| 👤 Rashmi | 👤 Shubhaang | 👤 Samiksha | 👤 Ankit |
|:---------:|:------------:|:-----------:|:--------:|
| Team Member | Team Member | Team Member | Team Member |

</div>

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Mid-Sem Compliance Declaration](#-mid-sem-compliance-declaration)
3. [Technical Architecture](#-technical-architecture)
4. [Data Pipeline](#-data-pipeline)
5. [Feature Engineering](#-feature-engineering)
6. [Models Implemented](#-models-implemented)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [Model Comparison](#-model-comparison)
9. [Visual Analysis](#-visual-analysis)
10. [Project Structure](#-project-structure)
11. [Installation & Setup](#-installation--setup)
12. [Deployment](#-deployment)
13. [Future Roadmap — Milestone 2](#-future-roadmap--milestone-2)
14. [Project Report & Video](#-project-report--video)
15. [Rubric Alignment](#-rubric-alignment)

---

## Project Overview

**IntelliCharge** is a traditional machine learning system designed to forecast EV charging demand at individual station level with **hourly granularity**. The project ingests raw charging session logs, performs rigorous data cleaning and feature engineering, and trains multiple regression models to predict energy demand (`kWhDelivered`) per station per hour.

The system is built for **operational utility** — outputs are exportable, deployment-ready, and structured for integration into a real-world EV infrastructure management layer in Milestone 2.

### Core Objective

> Given historical charging session records (`connectionTime`, `disconnectTime`, `kWhDelivered`, `stationID`), predict the **aggregated hourly energy demand** at each station with high accuracy and temporal consistency.

---

## Mid-Sem Compliance Declaration

> **This submission strictly adheres to Milestone 1 scope constraints.**

| Constraint | Status |
|:-----------|:------:|
| Traditional ML models only | Compliant |
| No Large Language Models (LLMs) | Not Used |
| No Agentic AI frameworks | Not Used |
| No Transformer-based architectures | Not Used |
| Chronological train/test split enforced | Implemented |
| Evaluation metrics reported | Reported |
| Model exported as `.pkl` | Exported |

---

## Technical Architecture

```
Raw JSON Session Logs
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  ← JSON → Pandas DataFrame
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Data Cleaning     │  ← Null removal, time validation,
│                     │    invalid kWh filtering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Hourly Aggregation  │  ← Group by (stationID, hour)
│ + IQR Outlier Cap   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  ← Temporal, cyclical, lag,
│                     │    rolling window features
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│    Chronological 80/20 Train Split   │
└─────────────┬────────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────┐       ┌──────────┐
│  Train   │       │   Test   │
│  Set     │       │   Set    │
└────┬─────┘       └────┬─────┘
     │                  │
     ▼                  ▼
┌─────────────────────────────────┐
│         Model Training          │
│  Linear Regression (Baseline)   │
│  Random Forest Regressor        │
│  Gradient Boosting Regressor    │
│  LightGBM Regressor             │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Evaluation & Comparison     │
│     MAE · RMSE · R² Score       │
└────────────────┬────────────────┘
                 │
                 ▼
     best_ev_demand_model.pkl
```

---

## Data Pipeline

### Stage 1 — Ingestion
- Raw session data loaded from **JSON format** into structured Pandas DataFrames
- Fields extracted: `connectionTime`, `disconnectTime`, `kWhDelivered`, `stationID`

### Stage 2 — Cleaning
- Removal of null/missing records
- Time validation: `disconnectTime > connectionTime` enforced
- Filtering sessions with `kWhDelivered ≤ 0` or physically implausible values

### Stage 3 — Aggregation & Outlier Handling
- Sessions aggregated to **hourly bins** per station
- **IQR-based outlier capping** applied to the target variable to reduce the effect of anomalous demand spikes

### Stage 4 — Train/Test Split
- **Strictly chronological** 80/20 split — no random shuffling
- Prevents data leakage from future time steps into training
- Simulates real deployment conditions faithfully

---

## Feature Engineering

| Feature | Type | Description |
|:--------|:----:|:------------|
| `station_encoded` | Categorical | Label-encoded station identifier |
| `hour` | Temporal | Hour of day (0–23) |
| `dayofweek` | Temporal | Day of week (0=Monday, 6=Sunday) |
| `month` | Temporal | Month of year (1–12) |
| `day` | Temporal | Day of month |
| `weekofyear` | Temporal | ISO week number |
| `hour_sin` / `hour_cos` | Cyclical | Sine/cosine encoding of hour — preserves circular continuity |
| `dow_sin` / `dow_cos` | Cyclical | Sine/cosine encoding of day-of-week |
| `lag_1` | Lag | Demand from the immediately preceding hour |
| `rolling_3h` | Rolling | 3-hour rolling mean of demand |
| `rolling_24h` | Rolling | 24-hour rolling mean of demand |

> **Design Rationale:** Cyclical encoding ensures the model correctly interprets temporal periodicity (e.g., hour 23 is adjacent to hour 0). Lag and rolling features inject short- and medium-term memory into stateless regressors.

---

## Models Implemented

### 1. Linear Regression *(Baseline)*
- Implemented as a **scikit-learn Pipeline** with `StandardScaler` preprocessing
- Establishes a performance lower bound; interpretable coefficients

### 2. Random Forest Regressor
- Ensemble of decision trees trained via bagging
- Robust to non-linear relationships and feature interaction
- Provides native **feature importance** scores

### 3. Gradient Boosting Regressor
- Sequential boosting of weak learners (sklearn implementation)
- Strong regularization via learning rate and tree depth control
- Typically outperforms Random Forest on structured tabular data

### 4. LightGBM Regressor
- Histogram-based gradient boosting; optimized for speed and memory
- Handles large-scale tabular data efficiently
- Expected to yield the strongest performance

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|:-------|:-------:|:---------------|
| **MAE** | `mean(|y - ŷ|)` | Average absolute error in kWh; directly interpretable |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalizes large errors more heavily than MAE |
| **R² Score** | `1 - SS_res/SS_tot` | Proportion of variance explained; 1.0 = perfect fit |

> All metrics computed on the **held-out chronological test set** only. No test-set leakage.

---

## Model Comparison

> Comprehensive evaluation on held-out chronological test set. No data leakage.

| Rank | Model | MAE (kWh) | RMSE (kWh) | R² Score |
|:----:|:------|:---------:|:----------:|:--------:|
| **1st** | **Linear Regression (Baseline)** | **4.1284** | **6.0836** | **0.6897** |
| 2nd | Random Forest Regressor | 4.2238 | 6.3989 | 0.6567 |
| 3rd | Gradient Boosting Regressor | 4.2818 | 6.4563 | 0.6506 |
| 4th | LightGBM Regressor | 4.4232 | 6.8204 | 0.6100 |

> **Best model:** Linear Regression (Baseline) — exported as `models/best_ev_demand_model.pkl`

---

## Visual Analysis

The following plots are generated and saved as part of the analysis pipeline:

| Plot | Description |
|:-----|:------------|
| **Actual vs Predicted** | Scatter/line overlay of ground truth and model predictions on the test set |
| **Feature Importance** | Bar chart of top contributing features (Random Forest / LightGBM) |
| **Hourly Demand Trend** | Average kWh delivered per hour of day across all stations |
| **Weekly Demand Trend** | Demand variation across days of the week |
| **Monthly Demand Trend** | Month-over-month demand trajectory |

![alt text](<Untitled design.png>)
> All visualizations are produced via `matplotlib` / `seaborn` and stored in `notebooks/`.

---

## Project Structure

```
EV-CHARGING-DEMAND-PREDICTION/
│
├── app/                        # Application layer (Streamlit / FastAPI)
│   └── ...
│
├── data/                       # Raw and processed datasets
│   ├── raw/
│   └── processed/
│
├── models/                     # Serialized trained models
│   └── best_ev_demand_model.pkl
│
├── notebooks/                  # Exploratory & training notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training_Evaluation.ipynb
│
├── src/                        # Modular source code
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- Git

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/team-rass/intellicharge.git
cd intellicharge

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the full training pipeline
python src/train.py

# 5. Evaluate the best model
python src/evaluate.py

# 6. Launch the application (optional)
streamlit run app/app.py
```

### Key Dependencies

```
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
joblib
streamlit
```

> Full pinned versions available in `requirements.txt`.

---

## Deployment

### Current Status

> **Deployment in progress.** The Streamlit application interface is functional locally.

| Component | Status | Link |
|:----------|:------:|:-----|
| Streamlit App (Local) | Functional | `localhost:8501` |
| Hosted Demo (Public) | In Progress | https://ev-charging-demand-prediction-2026.streamlit.app/#4fdf7da3 |
| API Endpoint | Planned | *(Milestone 2)* |

### Deployment Target

The application is planned for deployment on **Streamlit Community Cloud** or **Hugging Face Spaces**, serving real-time demand predictions from the exported `.pkl` model.

---

## Future Roadmap — Milestone 2

> Milestone 2 will evolve IntelliCharge from a predictive system into an **Agentic Infrastructure Planning** platform.

| Phase | Feature | Description |
|:------|:--------|:------------|
| **2.1** | LLM Integration | Natural language querying of demand forecasts |
| **2.2** | Agentic Planning Layer | Autonomous EV charging schedule optimization agents |
| **2.3** | Real-Time Data Ingestion | Live session stream integration via APIs |
| **2.4** | Multi-Station Orchestration | Cross-station demand balancing with RL/planning agents |
| **2.5** | Explainability Dashboard | SHAP-based model explanation interface |

> **Note:** All agentic and LLM capabilities are strictly scoped to Milestone 2. This submission contains **none** of the above.

---

## Project Report & Video

| Deliverable | Status | Link |
|:------------|:------:|:-----|
| LaTeX Project Report | In Progress | https://www.overleaf.com/read/gzkncnjwyhdh#29da89 |
| Project Demo Video | In Progress | *(To be added)* |

> The report is authored in **LaTeX** and covers: problem formulation, data pipeline design, feature justification, model selection rationale, experimental results, and limitations.

---

## Rubric Alignment

| Rubric Criterion | Coverage in This Repository |
|:----------------|:---------------------------|
| **Technical Implementation** | Data pipeline, 4 models, feature engineering, metrics — all documented and implemented in `src/` |
| **GitHub Repository & Code Quality** | Modular `src/` layout, clean notebooks, `.gitignore`, `requirements.txt`, structured README |
| **Hosted Link / Live Demo** | Streamlit app ready; public deployment in progress |
| **Project Report (LaTeX)** | Report authored in LaTeX; link to be added upon submission |
| **Project Video** | Demo walkthrough video in production |
| **Viva Voce** | All design choices documented with justification throughout this README and in-code comments |

---

<div align="center">

**Built with precision by Team RASS** · Milestone 1 · Mid-Semester Submission

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-FFD43B?style=flat-square&logo=python&logoColor=black)](.)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](.)
[![LightGBM](https://img.shields.io/badge/LightGBM-189DE0?style=flat-square&logo=lightgbm&logoColor=white)](.)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](.)

*© 2025 Team RASS — IntelliCharge. Academic use only.*

</div>