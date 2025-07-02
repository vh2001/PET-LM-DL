import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import LMNet, MiniConvNet, plot_batch_input_output_target
import argparse

# ---- Config ----
parser = argparse.ArgumentParser(description="Evaluate LMNet model on validation data.")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the model checkpoint (.pth file).",
)
parser.add_argument(
    "--intermediate_plots",
    action="store_true",
    help="Whether to generate intermediate plots during evaluation.",
)
args_cli = parser.parse_args()

checkpoint_path = args_cli.checkpoint_path
intermediate_plots = args_cli.intermediate_plots

# ---- Load Args ----
args_path = Path(checkpoint_path).parent / "args.json"
with open(args_path, "r", encoding="UTF8") as f:
    args = json.load(f)

num_training_samples = args["num_training_samples"]
num_validation_samples = args["num_validation_samples"]
num_blocks = args["num_blocks"]
num_features = args["num_features"]
num_hidden_layers = args.get("num_hidden_layers", 1)
weight_sharing = args.get("weight_sharing", False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = args.get(
    "alpha", -0.1
)  # the old default value was -0.01 (not present in args.json)
# a small positive value makes more sense for alpha (slope of non-lin for neg. values)

# ---- Setup Validation DataLoader ----
val_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[
    num_training_samples : num_training_samples + num_validation_samples
]
val_batch_size = 5


val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    collate_fn=brainweb_collate_fn,
)

# ---- Rebuild Model ----
if weight_sharing:
    conv_nets = MiniConvNet(
        num_features=num_features, num_hidden_layers=num_hidden_layers, alpha=alpha
    )
else:
    conv_nets = torch.nn.ModuleList(
        [
            MiniConvNet(
                num_features=num_features,
                num_hidden_layers=num_hidden_layers,
                alpha=alpha,
            )
            for _ in range(num_blocks)
        ]
    )

model = LMNet(conv_nets=conv_nets, num_blocks=num_blocks)


state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model_state_dict"]

######### HACK NEEDED BECAUSE VARIABLE IN LMNET WAS RENAMED
## Remap keys from 'convs.' to 'conv_net_list.'
# new_state_dict = {}
# for k, v in state_dict.items():
#    if k.startswith("convs."):
#        new_k = k.replace("convs.", "conv_net_list.", 1)
#    else:
#        new_k = k
#    new_state_dict[new_k] = v
#########


model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

psnr = PeakSignalNoiseRatio().to(device)

# ---- Evaluation Loop ----
val_losses = []
val_psnrs = []

criterion = torch.nn.MSELoss()

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
