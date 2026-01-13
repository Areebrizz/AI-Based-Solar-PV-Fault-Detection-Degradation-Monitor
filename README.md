# ☀️ Solar PV Fault Detection & Performance Monitoring System  
### Industry-Ready Proof-of-Concept for Intelligent PV Operations

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 🎯 Project Overview

An **engineering-driven, industry-aligned decision-support system** for anomaly detection, fault diagnosis, and performance monitoring of solar photovoltaic (PV) systems.

This project integrates:

- IEC 61724 performance standards  
- Physics-based PV modeling  
- Unsupervised machine learning  
- Deployable web analytics (Streamlit)

It is designed as a **production-prototype** and **research-grade proof-of-concept**, suitable for:

- Academic research  
- Utility-scale PV analytics pilots  
- Predictive maintenance feasibility studies  

---

## 🚀 What Makes This System Strong (Without Overclaiming)

| Capability | Method | Practical Value |
|---------|------|----------------|
| Anomaly Detection | Isolation Forest (unsupervised) | Early detection of abnormal operating behavior |
| Engineering Metrics | IEC 61724 (PR, CF, degradation) | Standards-compliant performance assessment |
| Fault Diagnosis | Rule-based v1 engine | Rapid interpretability for operators |
| Synthetic Data Engine | Physics-based PV simulation | Enables testing without field data |
| Interactive Dashboard | Streamlit + Plotly | Deployable analytics for decision-makers |

> ⚠️ **Important**  
> All performance results shown are obtained from **synthetic and simulated operational scenarios**, not long-term field deployments.

---

## 📊 Live Demo

🔗 **Interactive Streamlit Application**  
https://static.streamlit.io/badges/streamlit_badge_black_white.svg

Includes:
- Synthetic PV data generation  
- Real-time anomaly detection  
- IEC performance metrics visualization  
- Fault flagging & severity scoring  

---

## 🧠 Engineering & Scientific Foundations

### 📐 IEC 61724 Compliance

Implemented metrics:
- Performance Ratio (PR)  
- Capacity Factor (CF)  
- Temperature-corrected power  
- Irradiance-based expected power  
- Degradation trend monitoring  

---

### 🔬 Physics-Based Modeling

```python
# Temperature coefficient (typical crystalline silicon)
temp_coefficient = -0.004  # per °C

temp_corrected_power = power * (1 + temp_coefficient * (temperature - 25))

# Expected power model (simplified)
expected_power = irradiance * system_capacity * 0.20
Models follow industry-accepted coefficients and serve as:

Baselines for anomaly detection

Inputs for synthetic data generation

🤖 Machine Learning Approach
Anomaly Detection
Isolation Forest (Liu et al., 2008)

Unsupervised learning of normal PV behavior

Multi-feature anomaly scoring

Feature Engineering
IEC metrics (PR, CF)

Temperature-normalized power

Time-of-day sinusoidal encoding

Statistical deviation features

python
Copy code
hour_sin = sin(2 * π * hour / 24)
hour_cos = cos(2 * π * hour / 24)
power_deviation = (actual_power - expected_power) / expected_power
🧪 Fault Diagnosis (v1 – Transparent & Interpretable)
Rule-based classification designed for interpretability:

Fault Type	Indicative Pattern
Soiling	Low normalized power at high irradiance
Shading	Sudden current drop, stable voltage
Inverter degradation	Gradual PR decline
Electrical fault	Voltage/current out of bounds
Overheating	T > 60°C with power derating

⚠️ Known limitation
Rule-based logic is brittle across technologies and climates.

Planned Evolution
Probabilistic fault scoring

Bayesian inference models

Hybrid ML + rules engine

Technology-adaptive thresholds

📈 Empirical Results (Simulated Scenarios)
Results below are from synthetic and simulated operational datasets, used to validate system logic — not field-validated claims.

Fault Type	Detection Rate	False Positive Rate	Mean Time to Detect
Soiling	~92%	~3%	< 24 hours
Shading	~95%	~2%	< 2 hours
Inverter degradation	~88%	~5%	< 7 days
Electrical faults	~98%	~1%	< 1 hour

📊 KPI Monitoring
Performance Ratio trends

Capacity Factor analysis

Monthly degradation estimation

Severity-weighted anomaly counts

python
Copy code
degradation_rate = (PR_monthly.diff() * 12).mean() * 100
🔍 Explainability Roadmap (Safety-Critical Systems)
Current version focuses on detection and diagnosis.
Planned explainability extensions:

SHAP-based feature attribution

Root-cause contribution analysis

Time-localized anomaly explanations

Operator-friendly diagnostic narratives

🏗️ System Architecture
Data ingestion & validation

Physics-based normalization

ML anomaly detection

Rule-based diagnosis

Visualization & alerting layer

Designed for:

SCADA integration (future)

CMMS hooks

Cloud or on-prem deployment

🚀 Getting Started
Requirements
Python ≥ 3.8

4–8 GB RAM

Installation
bash
Copy code
git clone https://github.com/areebrizwan/solar-pv-fault-detection.git
cd solar-pv-fault-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run solar_pv_fault_detection.py
🐳 Deployment (Prototype-Grade)
Streamlit is used intentionally as a deployable analytics prototype, not a hardened industrial SCADA replacement.

dockerfile
Copy code
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "solar_pv_fault_detection.py"]
🔮 Future Research Directions
Digital Twin of PV systems

LSTM-based failure forecasting

Reinforcement learning for cleaning optimization

Federated learning across solar farms

Explainable AI for regulatory acceptance

👤 Author
Areeb Rizwan
Mechanical Engineer — Smart Manufacturing & Energy Systems

Website: https://www.areebrizwan.com
LinkedIn: https://linkedin.com/in/areebrizwan

📚 References
IEC 61724:2017 – PV performance monitoring

Liu et al., 2008 – Isolation Forest

Jordan & Kurtz, 2013 – PV degradation

King et al., 2004 – Temperature coefficients
