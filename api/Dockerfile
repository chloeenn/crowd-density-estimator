# Build from the repo root: docker build -f api/Dockerfile .
# (the API installs the sibling ml/ package, so the context must include both).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU wheels only: the default PyPI torch pulls in ~2 GB of CUDA libraries.
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY ml ./ml
COPY api/requirements.txt ./api/requirements.txt
WORKDIR /app/api
RUN pip install -r requirements.txt
COPY api/app ./app

RUN useradd --create-home --uid 10001 app
USER app

# Checkpoints are mounted, never baked in: -v ./models:/models:ro
ENV MODEL_PATH=/models/csrnet-partA.pth MODEL_PART=A TORCH_THREADS=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "--factory", "app.main:create_app", "--app-dir", "/app/api", "--host", "0.0.0.0", "--port", "8000"]
