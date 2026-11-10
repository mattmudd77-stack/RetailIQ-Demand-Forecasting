FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
COPY dashboard.py ./
# Regenerate synthetic data, forecasts, and dashboard assets during build.
# This avoids committing large generated datasets and model binaries.
RUN retailiq --out artifacts
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
