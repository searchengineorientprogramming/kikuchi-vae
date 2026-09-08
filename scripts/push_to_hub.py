import torch
from kikuchi import VAE


checkpoint = torch.load(
    "checkpoints/vae_best.pt",
    map_location="cpu"
)

model = VAE(
    latent_dim=checkpoint["latent_dim"],
    in_channels=1,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

model.push_to_hub(
    "jiayang23423423/test-vae"
)