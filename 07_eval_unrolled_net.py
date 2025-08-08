import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import plot_batch_input_output_target
from models import DENOISER_MODEL_REGISTRY, LMNet

# ---- Config ----
parser = argparse.ArgumentParser(description="Evaluate LMNet model on validation data.")
parser.add_argument(
    "checkpoint_path",
    type=str,
    help="Path to the model checkpoint (.pth file).",
)
parser.add_argument(
    "--intermediate_plots",
    action="store_true",
    help="Whether to generate intermediate plots during evaluation.",
)
args_cli = parser.parse_args()

checkpoint_path = Path(args_cli.checkpoint_path)
intermediate_plots = args_cli.intermediate_plots
model_dir = checkpoint_path.parent

# ---- Load Args ----
args_path = Path(checkpoint_path).parent / "args.json"
with open(args_path, "r", encoding="UTF8") as f:
    args = json.load(f)

val_batch_size = args.get("val_batch_size", 5)


# read validation directories from the checkpoint
with open(model_dir / "val_dirs.json", "r", encoding="UTF8") as f:
    val_dirs = [Path(d) for d in json.load(f)]


val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    collate_fn=brainweb_collate_fn,
)

# %%
# build the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(checkpoint_path, map_location="cpu")

# build the denoiser model
denoiser_model_class = ckpt["denoiser_model_class"]
denoiser_model_kwargs = ckpt["denoiser_model_kwargs"]

denoiser_model = DENOISER_MODEL_REGISTRY[denoiser_model_class](**denoiser_model_kwargs)

denoiser_model.to(device)

# rebuild the unrolled model
model_kwargs = ckpt["model_kwargs"]
num_blocks = ckpt["num_blocks"]
weight_sharing = model_kwargs.get("weight_sharing", False)

if weight_sharing:
    conv_nets = torch.nn.ModuleList([denoiser_model])
else:
    conv_nets = torch.nn.ModuleList([denoiser_model for _ in range(num_blocks)])

# setup the LMNet model
model = LMNet(conv_nets, num_blocks, **model_kwargs)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)

# -------------------------------------------------------------------------------

# %%

psnr = PeakSignalNoiseRatio(data_range=(0, 4)).to(device)

# ---- Evaluation Loop ----
val_losses = []
val_psnrs = []

criterion = torch.nn.MSELoss()

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        x = batch["input"].to(device)
        target = batch["target"].to(device)
        lm_pet_lin_ops = batch["lm_pet_lin_ops"]
        contamination_lists = batch["contamination_lists"]
        adjoint_ones = batch["adjoint_ones"]
        diag_preconds = batch["diag_preconds"]

        output = model(
            x,
            lm_pet_lin_ops,
            contamination_lists,
            adjoint_ones,
            diag_preconds,
            intermediate_plots=intermediate_plots,
        )
        loss = criterion(output, target)
        val_losses.append(loss.item())
        val_psnrs.append(psnr(output, target).item())

        print(
            f"Batch [{batch_idx+1}/{len(val_loader)}] - val. loss: {loss.item():.2E}, PSNR: {val_psnrs[-1]:.2f}"
        )
        # plot input, output, and target for the current validation batch
        plot_batch_input_output_target(
            x,
            output,
            target,
            Path(checkpoint_path).parent,
            prefix=f"{Path(checkpoint_path).stem}_val_sample_batch_{batch_idx:03}",
        )

print(f"\nAverage validation loss: {sum(val_losses)/len(val_losses):.2E}")
print(f"Average validation PSNR: {sum(val_psnrs)/len(val_psnrs):.2f}")
