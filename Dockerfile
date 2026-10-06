FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN PYTHONPATH=/app/src python -c "\
import torch; \
import numpy; \
import matplotlib; \
import tqdm; \
from huggingface_hub import PyTorchModelHubMixin; \
from model import VAE; \
print('All training dependencies OK')"

ENTRYPOINT ["python", "src/train.py"]
