import argparse
import json

import numpy as np
import torch
from PIL import Image

from kikuchi.model import VAE


MODEL_ID = "jiayang23423423/test-vae"


def load_image(path):
    image = Image.open(path).convert("L")
    image = image.resize((28, 28))

    x = np.asarray(image, dtype=np.float32) / 255.0

    # Your model was trained with [-1, 1]
    x = x * 2.0 - 1.0

    x = torch.from_numpy(x)
    x = x.unsqueeze(0).unsqueeze(0)

    return x


def save_reconstruction(tensor, path):
    x = tensor.squeeze().cpu()

    # [-1, 1] -> [0, 1]
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)

    x = (x.numpy() * 255).astype(np.uint8)

    Image.fromarray(x).save(path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "--output",
        default="reconstruction.png",
        help="Output reconstructed image"
    )

    parser.add_argument(
        "--latent-output",
        default="latent_vector.json",
        help="Output latent vector JSON"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Download/load model from Hugging Face
    model = VAE.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    x = load_image(args.input).to(device)

    with torch.inference_mode():

        latent = model.encode(x)

        reconstruction = model.decode(latent)

    save_reconstruction(
        reconstruction,
        args.output
    )

    latent_vector = (
        latent.squeeze(0)
        .cpu()
        .tolist()
    )

    with open(args.latent_output, "w") as f:
        json.dump(
            latent_vector,
            f,
            indent=2
        )

    print("Finished.")
    print(f"Reconstruction: {args.output}")
    print(f"Latent vector: {args.latent_output}")


if __name__ == "__main__":
    main()