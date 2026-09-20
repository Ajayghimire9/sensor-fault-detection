FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app
COPY requirements-portfolio.txt .
RUN pip install --no-cache-dir -r requirements-portfolio.txt fastapi 'uvicorn[standard]' prometheus-client
COPY sensorwatch sensorwatch
COPY artifacts artifacts
USER app
EXPOSE 8000
CMD ["uvicorn", "sensorwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
