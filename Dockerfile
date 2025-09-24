# FROM python:3.12

# WORKDIR /app

# # Upgrade system packages to reduce vulnerabilities
# RUN apt-get update && apt-get upgrade -y && apt-get clean

# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY src/ .

# # Expose FastAPI default port
# EXPOSE 8000

# CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]



# Use an official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir mlflow

COPY mlruns .

# Expose MLFlow default port
EXPOSE 5000

# Command to run MLFlow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--default-artifact-root", "/app/mlruns"]
