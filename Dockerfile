# Container for the DOO4730 live Streamlit dashboard (Azure App Service)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# App Service expects the app on the port in $WEBSITES_PORT (set to 8000)
CMD ["streamlit", "run", "live_sensor_monitor.py", "--server.port=8000", "--server.address=0.0.0.0", "--server.headless=true"]
