from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from kikuchi.model import VAE


MODEL_ID = "jiayang23423423/test-vae"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()


app = FastAPI(
    title="VAE API",
    description="API for encoding, decoding, and reconstructing images using a VAE.",
    version="0.1.0",
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = image.resize((28, 28))

    array = np.asarray(image, dtype=np.float32) / 255.0

    # training normalization: [0, 1] -> [-1, 1]
    array = array * 2.0 - 1.0

    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


def tensor_to_png(tensor: torch.Tensor) -> bytes:
    image = tensor.squeeze().cpu()

    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)

    image = (image.numpy() * 255).astype(np.uint8)

    pil_image = Image.fromarray(image, mode="L")

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    return buffer.getvalue()


@app.get("/")
def root():
    return {
        "message": "VAE API",
        "model": MODEL_ID,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
    }


@app.post("/encode")
async def encode(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    x = preprocess_image(image)

    with torch.inference_mode():
        z = model.encode(x)

    return {
        "latent_vector": z.squeeze(0).cpu().tolist()
    }


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    x = preprocess_image(image)

    with torch.inference_mode():
        reconstruction = model.reconstruct(x)

    png = tensor_to_png(reconstruction)

    return Response(
        content=png,
        media_type="image/png",
    )


class DecodeRequest(BaseModel):
    latent_vector: list[float]


@app.post("/decode")
def decode(request: DecodeRequest):

    if len(request.latent_vector) != model.latent_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected latent dimension {model.latent_dim}",
        )

    z = torch.tensor(
        request.latent_vector,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.inference_mode():
        image = model.decode(z)

    png = tensor_to_png(image)

    return Response(
        content=png,
        media_type="image/png",
    )