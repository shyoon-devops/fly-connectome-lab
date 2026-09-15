FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY webapp ./webapp
# Data artifacts are intentionally not tracked. Prepare data/full before
# building an image that serves the full-connectome experience.
COPY data ./data

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=3)"

CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8501", "--workers", "1"]
