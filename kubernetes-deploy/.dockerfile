
# Use an official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

RUN RUN apt-get update && apt-get upgrade -y && apt-get clean && \
    pip install --no-cache-dir mlflow 

COPY mlruns .

# Expose MLFlow default port
EXPOSE 5000

# Command to run MLFlow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--default-artifact-root", "/app/mlruns"]
