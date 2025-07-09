"""script to evaluate a trained image to image denoiser model"""

import argparse

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset
from utils import plot_batch_input_output_target
from models import DENOISER_MODEL_REGISTRY


# input parameters
parser = argparse.ArgumentParser(description="Train image to image denoiser model")
parser.add_argument(
    "ckpt",
    help="Path to the model checkpoint file",
)

args = parser.parse_args()
ckpt_path = Path(args.ckpt)

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

model_dir = ckpt_path.parent

# %%
# setup of data loaders
################################################################################

# read val_batch_size from args.json
with open(model_dir / "args.json", "r", encoding="UTF8") as f:
    args_json = json.load(f)
val_batch_size = args_json.get("val_batch_size")

# read validation directories from the checkpoint
with open(model_dir / "val_dirs.json", "r", encoding="UTF8") as f:
    val_dirs = [Path(d) for d in json.load(f)]

# write the validation directories to a json file
with open(model_dir / "val_dirs.json", "w", encoding="UTF8") as f:
    json.dump([str(d) for d in val_dirs], f, indent=4)

# make sure that skip_raw_data is set to True
# in this case the loader only returns the OSEM input image and the ground truth which is faster
val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=True)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
)

# %%
# setup of (unet) image to image denoiser model
################################################################################

# load the checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")
model_class = checkpoint["model_class"]
model_kwargs = checkpoint["model_kwargs"]

# model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup UNET model
# in principle we can have any NN here
# if we want to use in in combination with data fidelity gradient layers,
# we should make sure that the output is non-negative
model = DENOISER_MODEL_REGISTRY[model_class](**model_kwargs)

# load the model state dict from the checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

psnr = PeakSignalNoiseRatio().to(device)

# %%
# validation loop
################################################################################

model.eval()
with torch.no_grad():
    batch_psnr = torch.zeros(len(val_loader))
    for batch_idx, batch in enumerate(val_loader):
        val_x = batch["input"].to(device)
        val_target = batch["target"].to(device)

        val_output = model(val_x)
        batch_psnr[batch_idx] = psnr(val_output, val_target)

        # plot input, output, and target for the current validation batch
        plot_batch_input_output_target(
            val_x,
            val_output,
            val_target,
            ckpt_path.parent,
            prefix=f"{ckpt_path.stem}_batch_{batch_idx:03}",
        )

    val_psnr = batch_psnr.mean().item()

    print(f"val PSNR: {batch_psnr.mean().item():.2f}")
