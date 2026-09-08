FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

RUN python -c "\
import torch; \
import numpy; \
import safetensors; \
from PIL import Image; \
from huggingface_hub import PyTorchModelHubMixin; \
from kikuchi.model import VAE; \
print('All inference dependencies OK')"

ENTRYPOINT ["python", "scripts/infer.py"]