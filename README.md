# Churn Prediction API

**A FastAPI-based microservice for predicting customer churn using machine learning, integrated with Redis, MLflow, Prometheus, and Grafana for monitoring.**

---

## Project Overview

The **Churn Prediction API** helps businesses predict customer churn using a machine learning model. It supports batch prediction, incremental retraining, and stores predictions in a Redis database for future analysis.  

It also integrates with **MLflow** for model tracking, **Prometheus** for real-time metrics collection, and **Grafana** for visualizing model performance metrics.

---

## Features

- **Batch prediction** via FastAPI endpoints  
- **Redis integration** to store predictions and training data  
- **MLflow integration** for model tracking and experiment logging  
- **Prometheus + Grafana monitoring** for metrics: accuracy, precision, recall, F1-score, and total models  
- **Incremental model retraining** with historical and new data  
- Clean, professional, and maintainable API structure  

---

## Tech Stack

- **Backend:** Python, FastAPI  
- **Machine Learning:** Scikit-learn, Pandas  
- **Database / Cache:** Redis  
- **Monitoring & Dashboard:** Prometheus, Grafana  
- **Model Management:** MLflow  
- **Async Processing:** `redis.asyncio`  
- **Containerization (optional):** Docker  

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/churn-prediction-api.git
cd churn-prediction-api
