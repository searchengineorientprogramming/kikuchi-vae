from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import base64
import torch
import numpy as np

from kikuchi.model import VAE


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE.from_pretrained(
    "jiayang23423423/test-vae"
).to(device)

model.eval()


def preprocess(image):

    image = image.convert("L")
    image = image.resize((28, 28))

    x = np.array(image).astype(np.float32) / 255.0

    # if your training used [-1, 1]
    x = x * 2 - 1

    x = torch.tensor(x)
    x = x.unsqueeze(0).unsqueeze(0)

    return x.to(device)


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    )

    x = preprocess(image)

    with torch.inference_mode():

        mu, logvar = model.encode(x)

        # use mean as deterministic latent representation
        z = mu

        reconstruction = model.decode(z)


    # ----------------------
    # latent vector
    # ----------------------

    latent_vector = (
        z.squeeze()
        .cpu()
        .numpy()
        .tolist()
    )


    # ----------------------
    # reconstruction → PNG
    # ----------------------

    reconstructed = reconstruction.squeeze().cpu()

    reconstructed = (reconstructed + 1) / 2
    reconstructed = reconstructed.clamp(0, 1)

    reconstructed = (
        reconstructed.numpy() * 255
    ).astype(np.uint8)

    reconstructed_image = Image.fromarray(
        reconstructed
    )

    buffer = BytesIO()

    reconstructed_image.save(
        buffer,
        format="PNG"
    )

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    return {
        "latent_vector": latent_vector,

        "reconstructed_image":
            f"data:image/png;base64,{image_base64}"
    }