# ==============================================================================
# StockRL Mini — Production RL Simulation & Telemetry Container
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install Python dependencies with extended timeout for network resilience
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy application source code
COPY . /app

# Install project package for direct Python module imports
RUN pip install --no-cache-dir --no-deps -e .

# Expose ports for Streamlit Telemetry Workbench (8501) and FastAPI Backend (8000)
EXPOSE 8501 8000

# Default execution: Run test suite followed by baseline agent benchmark
CMD ["sh", "-c", "pytest tests/ -v && python scripts/run_agent.py --agent all --task all"]
