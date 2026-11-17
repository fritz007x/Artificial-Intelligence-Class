# syntax=docker/dockerfile:1
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    ORCHIDSHIELD_ENV=prod
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service needs at runtime: code, config, rules, uncertainty models, trained artifacts, playbooks, scenarios
COPY src ./src
COPY configs ./configs
COPY 03_logic ./03_logic
COPY 04_uncertainty ./04_uncertainty
COPY 05_ml/artifacts ./05_ml/artifacts
COPY docs/playbooks ./docs/playbooks
COPY data/synthetic ./data/synthetic
COPY data/scenarios ./data/scenarios

# Non-root user; only the runtime dir is writable
RUN useradd --create-home --uid 10001 orchid && mkdir -p data/runtime && chown -R orchid data/runtime
USER orchid

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
