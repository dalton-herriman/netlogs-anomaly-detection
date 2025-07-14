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
```plaintext
anomaly-detector/
├── data/                 # pre-processed datasets
├── notebooks/            # Jupyter notebooks for EDA and training
├── src/
│   ├── preprocess.py     # Log parsing and feature extraction
│   ├── train.py          # Model training and evaluation
│   ├── inference.py      # Core prediction logic
│   └── api.py            # FastAPI app for serving predictions
├── Dockerfile            # Container build configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # Project documentation
```
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

### Example Request
```json
POST /predict

{
  "src_ip": "192.168.1.100",
  "dst_ip": "192.168.1.1",
  "src_port": 443,
  "dst_port": 51515,
  "protocol": "TCP",
  "packet_count": 12,
  "byte_count": 1024,
  "duration": 3.4
}
```

### Example response
```json
{
  "anomaly_score": 0.91,
  "label": "anomaly"
}
```

## Docker
```bash
docker build -t anomaly-detector .
docker run -p 8000:8000 anomaly-detector
```

## 📈 Monitoring (Optional)
Export metrics from api.py using Prometheus client. Then:
  - Connect Prometheus to the /metrics endpoint
  - Visualize request counts and latencies in Grafana

## 🛡 Security Considerations
  - Input validation using pydantic
  - Rate-limiting and IP whitelisting ready to integrate
  - Ideal for secure SOC toolchains

## 📖 License 
  - MIT License

## 🤝 Acknowledgments 
  - CICIDS 2017 Dataset (Canadian Institute for Cybersecurity)
  - scikit-learn, FastAPI, Docker, MLflow teams
