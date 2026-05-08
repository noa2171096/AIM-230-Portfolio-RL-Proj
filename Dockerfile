# RL Portfolio Advisor — Multi-stage Docker Build
# ===================
# Stage 1: Base
# ===================
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===================
# Stage 2: Development
# ===================
FROM base as development

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-cache FinBERT so container doesn't download at runtime
RUN python -c "\
from transformers import BertTokenizer, BertForSequenceClassification; \
t = BertTokenizer.from_pretrained('ProsusAI/finbert'); \
m = BertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
t.save_pretrained('/app/app/ml/models/finbert'); \
m.save_pretrained('/app/app/ml/models/finbert'); \
print('FinBERT cached')"

COPY app/ ./app/
COPY data/ ./data/

#create non root user
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/uploads /app/saved_models /app/data && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ===================
# Stage 3: Production
# ===================
FROM base as production

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN python -c "\
from transformers import BertTokenizer, BertForSequenceClassification; \
t = BertTokenizer.from_pretrained('ProsusAI/finbert'); \
m = BertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
t.save_pretrained('/app/app/ml/models/finbert'); \
m.save_pretrained('/app/app/ml/models/finbert'); \
print('FinBERT cached')"

COPY app/ ./app/
COPY data/ ./data/

COPY alembic.ini ./
COPY alembic/ ./alembic/

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/uploads /app/saved_models /app/data && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]