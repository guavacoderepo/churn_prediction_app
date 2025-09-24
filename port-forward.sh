kubectl port-forward svc/fastapi-service 8000:8000 -n churn-stack &
kubectl port-forward svc/prometheus-operated 9090:9090 -n churn-stack &
kubectl port-forward svc/prometheus-grafana 5050:80 -n churn-stack &
 