# Churn Prediction API

**A FastAPI-based microservice for predicting customer churn using machine learning, integrated with Redis, MLflow, Prometheus, and Grafana, deployed on Azure.**

---

## Project Overview

The **Churn Prediction API** is designed to help businesses proactively identify customers at risk of churn using advanced machine learning models. By leveraging historical and real-time customer data, organizations can make data-driven decisions to retain valuable customers, optimize revenue, and improve overall customer satisfaction.  

Key capabilities include:

- **Batch and real-time predictions** via RESTful API endpoints.  
- **Incremental model retraining** to continuously improve prediction accuracy using new data.  
- **Redis integration** for storing predictions and customer data for fast retrieval and historical analysis.  
- **MLflow integration** for tracking experiments, managing model versions, and logging metrics.  
- **Prometheus and Grafana monitoring** for real-time performance visualization and alerting.  
- **Deployment on Azure** ensures high availability, scalability, and easy integration with enterprise systems.  

This API is modular, maintainable, and production-ready, making it ideal for cloud deployment and integration with other services.

---

## Key Features

- **Batch Prediction:** Accept customer datasets in CSV or JSON format and return churn probability predictions.  
- **Incremental Retraining:** Incorporate new customer data into existing models without retraining from scratch, improving model performance over time.  
- **Redis Integration:** Cache predictions and features for fast access and historical analysis.  
- **MLflow Integration:** Track all model experiments, metrics, and versions for reproducibility and easy rollback.  
- **Prometheus & Grafana Monitoring:** Collect and visualize key metrics including accuracy, precision, recall, F1-score, and the number of deployed models.  
- **Cloud Deployment on Azure:** Leverage Azure App Services or Azure Container Instances for scalable, reliable hosting.  
- **Clean and Scalable API Structure:** Follows best practices for FastAPI applications with asynchronous processing, modularity, and extensibility.  

---

## Tech Stack

- **Backend:** Python, FastAPI  
- **Machine Learning:** Scikit-learn, Pandas  
- **Database / Cache:** Redis (`redis.asyncio`)  
- **Model Management:** MLflow  
- **Monitoring & Dashboard:** Prometheus, Grafana  
- **Cloud Deployment:** Azure App Services / Azure Container Instances  
- **Containerization (Optional):** Docker  

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/churn-prediction-api.git
cd churn-prediction-api
