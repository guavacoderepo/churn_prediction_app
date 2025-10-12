# Customer Retention Using Churn Prediction

<img src="ChatGPT Image Oct 12, 2025, 09_30_49 PM.png">


**A FastAPI-based microservice for predicting customer churn using trained machine learning models, integrated with Redis, MLflow, Prometheus, and Grafana, deployed on Azure.**

---

## Project Overview

The **Customer Retention API** leverages machine learning to help businesses identify customers at risk of churn, enabling proactive retention strategies. Using a **Random Forest Classifier (RFC)**, the system predicts churn probabilities based on historical customer data, with data preprocessing and imbalance handling incorporated for optimal model performance.  

Key capabilities include:

- **Data preprocessing:** Cleaning, feature engineering, and transformation of raw customer data.  
- **Handling imbalanced datasets:** Resampling techniques applied to ensure model fairness and accuracy.  
- **Batch and real-time predictions** via RESTful API endpoints.  
- **Incremental model retraining** to continuously improve prediction accuracy using new data.  
- **Redis integration** for storing predictions and historical data.  
- **MLflow integration** for experiment tracking, model versioning, and metrics logging.  
- **Prometheus and Grafana monitoring** for real-time performance visualization.  
- **Deployment on Azure** ensures high availability and scalability.  

This API is production-ready, modular, and designed for cloud deployment and enterprise integration.

---

## Key Features

- **Batch Prediction:** Submit customer datasets (CSV or JSON) to receive churn probability predictions.  
- **Random Forest Classifier (RFC):** Trained on preprocessed data with resampling to handle class imbalance.  
- **Incremental Retraining:** Update models with new customer data without retraining from scratch.  
- **Redis Integration:** Cache predictions and features for fast access and historical analysis.  
- **MLflow Integration:** Track all experiments, metrics, and model versions for reproducibility.  
- **Prometheus & Grafana Monitoring:** Collect and visualize metrics including accuracy, precision, recall, F1-score, and total deployed models.  
- **Cloud Deployment on Azure:** Leverage Azure App Services or Container Instances for scalable and reliable hosting.  
- **Clean, Modular API Structure:** Asynchronous processing, modular design, and maintainable codebase.

---

## Tech Stack

- **Backend:** Python, FastAPI  
- **Machine Learning:** Scikit-learn (Random Forest Classifier), Pandas  
- **Database / Cache:** Redis (`redis.asyncio`)  
- **Model Management:** MLflow  
- **Monitoring & Dashboard:** Prometheus, Grafana  
- **Cloud Deployment:** Azure App Services / Azure Container Instances  
- **Containerization (Optional):** Docker  

---
