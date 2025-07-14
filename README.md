# Anomaly Detection in Network Logs

Detect suspicious or anomalous network activity using machine learning. This project demonstrates how to process raw network logs, extract meaningful features, train a model, and deploy an inference API for real-time detection.

---

## 📌 Overview

- **Goal:** Identify anomalous traffic (e.g., DDoS, port scans) from network logs using supervised machine learning.
- **Dataset:** CICIDS 2017 (Canadian Institute for Cybersecurity)
- **Tech Stack:**
  - Python (Pandas, Scikit-learn, XGBoost)
  - FastAPI (Real-time API)
  - Docker (Containerized deployment)
  - MLFlow (Model tracking)
  - Prometheus + Grafana (Monitoring)

---

## 🚀 Features

✅ Data preprocessing and feature extraction  
✅ Supervised anomaly classification  
✅ FastAPI-based inference endpoint  
✅ Dockerized for easy deployment  
✅ Model metrics logged with MLflow  
✅ Monitoring hooks (inference time, request count)

---

## 📂 Project Structure
anomaly-detector/
├── data/ # pre-processed datasets
├── notebooks/ 
├── src/
│ ├── preprocess.py 
│ ├── train.py 
│ ├── inference.py 
│ └── api.py 
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md

## 🧪 Model Training

1. Download the [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract features:

```bash
python src/preprocess.py --input data/raw/ --output data/processed/
```
3. Train the model:
```bash
python src/train.py --input data/processed/
```

## 🌐 Inference API
Start the API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```


