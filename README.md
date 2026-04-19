<div align="center">

# ChargeSense
### Intelligent EV Charging Demand Prediction & Agentic Infrastructure Planning

*From raw session data to explainable, AI-driven deployment decisions — in one pipeline.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ev-charging-demand-prediction-2026.streamlit.app)
[![Milestone](https://img.shields.io/badge/Milestone-2%20%7C%20End--Sem-8E44AD?style=for-the-badge&logo=lightning&logoColor=white)](https://github.com/S-h-u-b-h-1/EV-charging-demand-prediction)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-000000?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![RAG](https://img.shields.io/badge/RAG-FAISS%20Enabled-2ECC71?style=for-the-badge&logo=semanticweb&logoColor=white)](https://github.com/S-h-u-b-h-1/EV-charging-demand-prediction)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA--3.1-FF6F00?style=for-the-badge)](https://console.groq.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=black)](https://python.org)
[![Team](https://img.shields.io/badge/Team-RASS-0BCD4?style=for-the-badge&logo=github&logoColor=white)](https://github.com/S-h-u-b-h-1)

---

> **ChargeSense** is a full-stack Agentic AI system that predicts EV charging demand,
> detects congestion hotspots, retrieves infrastructure planning guidelines via RAG,
> reasons with a Groq-hosted LLaMA-3.1 LLM, and generates optimised charger
> placement and scheduling recommendations — all deployed publicly on Streamlit Cloud.

[Live Demo](https://ev-charging-demand-prediction-saamcexbagk7gmpmdfotqd.streamlit.app/) &nbsp;·&nbsp;
[LaTeX Report](https://www.overleaf.com/project/69e4ce549676b0b08627ab2d) &nbsp;·&nbsp;
[Demo Video](#demo-video) &nbsp;·&nbsp;
[Requirements](requirements.txt)

</div>

---

## Table of Contents

- [Why ChargeSense?](#-why-chargesense)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Running the App](#-running-the-app)
- [Input / Output Specification](#-input--output-specification)
- [Model Performance](#-model-performance)
- [Optimisation Logic](#-optimisation-logic)
- [Robustness Design](#-robustness-design)
- [Demo Video](#-demo-video)
- [Team](#-team)
- [Future Improvements](#-future-improvements)

---

## Why ChargeSense?

The global EV fleet is projected to exceed **300 million vehicles by 2030**. Infrastructure
planners face a hard problem every day:

| Question | Old Answer | ChargeSense Answer |
|---|---|---|
| How many chargers do I need? | Rule of thumb: peak ÷ 10 | Utilisation-optimised count (70–90% band) |
| Which stations need attention? | Manual inspection | Automated hotspot detection (μ + σ threshold) |
| When will the grid be stressed? | Guesswork | LLM + RAG-grounded load balancing alert |
| What should I do about it? | Consultant report | Structured deployment plan in seconds |

---

## Features

- ** ML Demand Forecasting** — scikit-learn pipeline with cyclical encoding, lag features, and 4-model comparison. Best model: Linear Regression (MAE 4.13 kWh, R² 0.69).
- ** LangGraph Agentic Workflow** — 8-node stateful graph: Input → Preprocessing → Prediction → Hotspot Detection → RAG Retrieval → Reasoning → Planning → Output.
- ** FAISS-based RAG** — planning guidelines from IEEE, NREL, and DOE retrieved and injected into LLM context at inference time.
- ** Groq LLM Reasoning** — LLaMA-3.1-8b-instant generates 4-point explainable planning rationale grounded in retrieved documents.
- ** Layered Robustness** — graceful degradation at every failure point: missing columns, retrieval failure, LLM unavailability. No silent crashes.
- ** Streamlit Dashboard** — professional UI with metrics, forecast charts, hotspot tables, reasoning display, and JSON export.
- ** Free-tier Deployed** — Streamlit Community Cloud. No paid API required for the default flow; Groq free tier handles LLM calls.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ChargeSense Pipeline                            │
│                                                                         │
│  CSV Upload ──► Validation ──► ML Forecast ──► LangGraph Agent          │
│                                                      │                  │
│                              ┌───────────────────────┤                  │
│                              │                       │                  │
│                         FAISS RAG              Groq LLM                 │
│                         (Guidelines)           (LLaMA-3.1)              │
│                              │                       │                  │
│                              └───────── Planning ────┘                  │
│                                             │                           │
│                                      Streamlit UI                       │
│                               (Dashboard + JSON Export)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### LangGraph Agent Nodes

```
[Input] ──► [Preprocessing] ──► [Prediction] ──► [Hotspot Detection]
                                                          │
                                                   [RAG Retrieval]
                                                          │
                                                  [Reasoning + LLM]
                                                          │
                                                    [Planning]
                                                          │
                                                     [Output]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | scikit-learn, pandas, numpy, joblib |
| Agent Framework | LangGraph (StateGraph) |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Vector Store | FAISS (flat L2 index) |
| Embeddings | sentence-transformers |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.10+ |

---

## Project Structure

```
EV-charging-demand-prediction/
│
├── agent/                  # LangGraph workflow
│   ├── workflow.py         # All 8 node definitions + planning logic
│   ├── graph.py            # Graph builder and run_agent_workflow()
│   ├── state.py            # EVAgentState TypedDict
│   └── __init__.py
│
├── ml/                     # ML pipeline
│   ├── inference.py        # Prediction wrapper with failure handling
│   ├── evaluation.py       # MAE, RMSE, R² computation
│   └── config.py           # Feature list, model path
│
├── rag/                    # Retrieval-Augmented Generation
│   ├── ingest.py           # Build and save FAISS index
│   ├── vectorstore.py      # Query FAISS; fallback rules
│   └── faiss_index/        # Pre-built index (committed)
│
├── ui/
│   └── streamlit_app.py    # Main Streamlit dashboard
│
├── utils/
│   ├── validation.py       # CSV validation and imputation
│   ├── contracts.py        # Architecture and spec text constants
│   └── logger.py           # Shared logging setup
│
├── data/                   # Raw and processed datasets
├── models/                 # best_ev_demand_model.pkl
├── notebooks/              # EDA and training notebooks
├── reports/                # LaTeX report assets
├── requirements.txt
├── .env.example            # Environment variable template
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) A Groq API key for LLM reasoning — [get one free here](https://console.groq.com)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/S-h-u-b-h-1/EV-charging-demand-prediction.git
cd EV-charging-demand-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
# Required for LLM-enhanced reasoning (free tier available)
GROQ_API_KEY=your_groq_api_key_here
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Enables Groq LLaMA-3.1 reasoning. Falls back to rule-based if absent. |

**For Streamlit Cloud deployment:** Add `GROQ_API_KEY` under **Settings → Secrets** in your Streamlit app dashboard.

---

## Running the App

### Local

```bash
streamlit run ui/streamlit_app.py
```

The app opens at `http://localhost:8501`.

### Utility commands

```bash
# Rebuild FAISS index from guideline documents
python rag/ingest.py

# Check model evaluation metrics
python -c "from ml.evaluation import load_model_metrics; import json; print(json.dumps(load_model_metrics(), indent=2))"
```

---

## Input / Output Specification

### Input CSV

| Field | Required | Description |
|---|---|---|
| `station_encoded` | ✅ Yes | Numeric station ID |
| `hour` | ✅ Yes | Hour of day (0–23) |
| `dayofweek` | Optional | Calendar weekday (0=Mon) |
| `month`, `day`, `weekofyear` | Optional | Calendar features |
| `hour_sin`, `hour_cos` | Optional | Cyclical hour encoding |
| `dow_sin`, `dow_cos` | Optional | Cyclical weekday encoding |
| `lag_1` | Optional | Previous-hour demand |
| `rolling_3h`, `rolling_24h` | Optional | Rolling demand averages |

Missing optional fields are automatically imputed with domain-appropriate defaults.

### Output

```json
{
  "summary":   { "avg_demand": 18.4, "peak_demand": 28.1, "peak_hour": "18:00", "risk_level": "High" },
  "hotspots":  [{ "hour": "18:00", "demand": 28.1, "severity": "High" }],
  "infrastructure_plan": [{ "chargers": 3, "charger_type": "DC Fast Charger (50kW+)", "utilization_target": "77.8%", "load_balancing": "Required" }],
  "schedule":  ["Peak window (17:00, 18:00, 19:00): prioritize queue control.", "Off-peak (01:00–06:00): incentivize discounted charging."],
  "reasoning": ["Peak demand is 28.10 kWh at 18:00 with average demand 18.40 kWh.", "..."],
  "llm_reasoning": "• 3 DC Fast Chargers maintain utilisation at 77.8%, within the 70–90% efficiency band...",
  "llm_used": true,
  "rag_fallback_used": false
}
```

---

##  Model Performance

Evaluated on **2,722 held-out rows** using a strict chronological 80/20 split.

| Rank | Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|---|
| 1 | Linear Regression Pipeline | **4.13** | **6.08** | **0.690** |
| 2 | Random Forest Regressor | 4.22 | 6.40 | 0.657 |
| 3 | Gradient Boosting Regressor | 4.28 | 6.46 | 0.651 |
| 4 | LightGBM Regressor | 4.42 | 6.82 | 0.610 |

> The Linear Regression pipeline with polynomial feature interactions and standard
> scaling outperformed tree-based ensembles because the hand-crafted temporal features
> already capture the dominant demand patterns.

---

##  Optimisation Logic

The charger count is **not** `peak_demand / 10`. It solves:

```
n* = argmin [ δ(n) + 0.01·n ]
```

where `δ(n)` = distance of utilisation `u(n) = P_peak / (n × C_charger)` from the
70–90% target band, and `0.01n` is a cost penalty discouraging oversizing.

| Peak Demand | Charger Type | Capacity |
|---|---|---|
| > 25 kWh | DC Fast Charger (50kW+) | 12.0 kWh/slot |
| > 15 kWh | Hybrid AC/DC | 8.0 kWh/slot |
| ≤ 15 kWh | Level 2 AC | 6.0 kWh/slot |

---

## Robustness Design

| Failure | System Response |
|---|---|
| Missing required columns | Validation error with field list |
| Missing optional features | Imputed; warning shown |
| Invalid numeric values | Coerced via `pd.to_numeric`; imputed |
| Model prediction fails | Safe error message |
| FAISS retrieval fails | Fallback rules; `rag_fallback_used=True` |
| `GROQ_API_KEY` absent | Rule-based reasoning; no crash |
| Groq API exception | Logged; rule-based continues |
| Agent workflow exception | Safe structured fallback dictionary |

---

## Demo Video

> **[Watch the Demo Video](#)** ← *(update with YouTube link after recording)*

The 5-minute walkthrough covers:
1. System overview and motivation
2. Live CSV upload and station selection
3. Agent workflow execution with progress bar
4. Forecast chart + hotspot detection
5. LLM Insight and planning output
6. JSON export and GitHub tour

---

## Team

| Name | Role |
|---|---|
| **Rashmi** | RAG pipeline, report writing |
| **Shubhaang** | Architecture, LangGraph, deployment |
| **Samiksha** | ML pipeline, model evaluation |
| **Ankit** | UI, data preprocessing |

---

## Future Improvements

1. **Live OCPI API integration** — real-time demand feeds from public charging networks
2. **Multi-station joint optimisation** — jointly plan charger deployment across a grid-connected network
3. **RL-based dynamic scheduling** — train a pricing policy to shift demand in real time
4. **Weather + event features** — reduce the unexplained 31% demand variance
5. **Conditional LangGraph routing** — route low-confidence predictions to a human-review node

---

## License

This repository is intended for academic use — Project 15, AI/ML Course, 2026.

---

<div align="center">
  <sub>Built with by Team RASS · Powered by LangGraph · Groq · FAISS · Streamlit</sub>
</div>
