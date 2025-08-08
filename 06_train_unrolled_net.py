import argparse
import matplotlib.pyplot as plt

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import plot_batch_input_output_target
from models import DENOISER_MODEL_REGISTRY, LMNet


def parse_json_model_kwargs(arg: str):
    try:
        obj = json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --model-kwargs: {e}")
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError(
            "--model-kwargs must be a JSON object (i.e. dict)"
        )
    return obj


# input parameters
parser = argparse.ArgumentParser(description="Train LMNet for PET image reconstruction")
parser.add_argument(
    "denoiser_model_path",
    type=str,
    help="checkpoint path to pretraind denoiser model",
)
parser.add_argument(
    "--num_blocks", type=int, default=2, help="Number of unrolled blocks"
)

parser.add_argument(
    "--count_level",
    type=float,
    default=1.0,
    help="Count level of simulated PET data",
)
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

parser.add_argument(
    "--model_kwargs",
    type=parse_json_model_kwargs,
    default={},
    help="JSON-encoded dict of init args for the unrolled model, e.g. "
    "'{\"weight_sharing\":True}'",
)

parser.add_argument(
    "--num_training_samples",
    type=int,
    default=30,
    help="Number of training samples",
)
parser.add_argument("--tr_batch_size", type=int, default=5, help="Training batch size")
parser.add_argument(
    "--num_validation_samples",
    type=int,
    default=5,
    help="Number of validation samples",
)
parser.add_argument(
    "--val_batch_size", type=int, default=5, help="Validation batch size"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--weight_sharing", action="store_true", help="Use weight sharing")
parser.add_argument(
    "--print_gradient_norms",
    action="store_true",
    help="Print gradient norms during training (useful for debugging)",
)
parser.add_argument(
    "--custom_tag",
    type=str,
    default="",
    help="Custom tag for the experiment",
)

args = parser.parse_args()

seed = args.seed
num_epochs = args.num_epochs
lr = args.lr
num_training_samples = args.num_training_samples
tr_batch_size = args.tr_batch_size
num_validation_samples = args.num_validation_samples
val_batch_size = args.val_batch_size
weight_sharing = args.weight_sharing
print_gradient_norms: bool = args.print_gradient_norms
denoiser_model_path: Path = Path(args.denoiser_model_path)
num_blocks: int = args.num_blocks
count_level: float = args.count_level
model_kwargs: dict = args.model_kwargs
custom_tag: str = args.custom_tag

# create a directory for the model checkpoints
if custom_tag != "":
    run_id = f"{custom_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

model_dir = Path(f"checkpoints_unrolled/lmnet_run_{run_id}")
model_dir.mkdir(parents=True, exist_ok=True)

# auto convert the Namespace args to a dictionary
args_dict = vars(args)
# save the arguments to a json file
with open(model_dir / "args.json", "w", encoding="UTF8") as f:
    json.dump(args_dict, f, indent=4)
print(f"Arguments saved to {model_dir / 'args.json'}")

# %%
# setup of data loaders
################################################################################

torch.manual_seed(seed)

all_data_dirs = sorted(
    list(Path("data/sim_pet_data").glob(f"subject*_countlevel_{count_level:.1f}_*"))
)

# setup the training data loader
train_dirs = all_data_dirs[:num_training_samples]

# write the training directories to a json file
with open(model_dir / "train_dirs.json", "w", encoding="UTF8") as f:
    json.dump([str(d) for d in train_dirs], f, indent=4)


train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True, skip_raw_data=False)
train_loader = DataLoader(
    train_dataset,
    batch_size=tr_batch_size,
    collate_fn=brainweb_collate_fn,
    drop_last=True,
)

# setup the validation data loader
val_dirs = all_data_dirs[
    num_training_samples : num_training_samples + num_validation_samples
]

# write the validation directories to a json file
with open(model_dir / "val_dirs.json", "w", encoding="UTF8") as f:
    json.dump([str(d) for d in val_dirs], f, indent=4)

val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    collate_fn=brainweb_collate_fn,
)

# %%
# load the pre-trained denoiser model
################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% load the denoiser model
# load "num_features" from args.json in the same directory as the denoiser model

# load the checkpoint
denoiser_checkpoint = torch.load(denoiser_model_path, map_location="cpu")
denoiser_model_class = denoiser_checkpoint["model_class"]
denoiser_model_kwargs = denoiser_checkpoint["model_kwargs"]

# model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# in principle we can have any NN here
# if we want to use in in combination with data fidelity gradient layers,
# we should make sure that the output is non-negative
denoiser_model = DENOISER_MODEL_REGISTRY[denoiser_model_class](**denoiser_model_kwargs)

# load the model state dict from the checkpoint
denoiser_model.load_state_dict(denoiser_checkpoint["model_state_dict"])
denoiser_model.to(device)


# %%
# setup of LMNet model (combination of data fidelity gradient layers and NNs)
################################################################################

weight_sharing = model_kwargs.get("weight_sharing", False)

if weight_sharing:
    conv_nets = torch.nn.ModuleList([denoiser_model])
else:
    conv_nets = torch.nn.ModuleList([denoiser_model for _ in range(num_blocks)])

# setup the LMNet model
model = LMNet(conv_nets, num_blocks, **model_kwargs).to(device)

# setup the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

psnr = PeakSignalNoiseRatio(data_range=(0, 4)).to(device)

# save model architecture
with open(model_dir / "lmnet_architecture.txt", "w", encoding="UTF8") as f:
    f.write(str(model))
print(f"Model architecture saved to {model_dir / 'lmnet_architecture.txt'}")

val_psnr = torch.zeros(num_epochs)
val_loss_avg = torch.zeros(num_epochs)
train_loss_avg = torch.zeros(num_epochs)

# %%
# training + validation loop
################################################################################

# training loop
model.train()
for epoch in range(1, num_epochs + 1):
    batch_losses = torch.zeros(len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        x = batch["input"].to(device)
        target = batch["target"].to(device)
        lm_pet_lin_ops = batch["lm_pet_lin_ops"]
        contamination_lists = batch["contamination_lists"]
        adjoint_ones = batch["adjoint_ones"]
        diag_preconds = batch["diag_preconds"]

        optimizer.zero_grad()
        output = model(
            x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
        )
        loss = criterion(output, target)
        loss.backward()

        # print gradient norms if requested - useful to see if gradients are exploding or vanishing
        if print_gradient_norms:
            # print gradient norms
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"{name:40s} | grad is None")
                else:
                    print(f"{name:40s} | grad norm: {param.grad.norm():.6f}")
            print()

        optimizer.step()
        batch_losses[batch_idx] = loss.item()

        print(
            f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(train_loader):03}] - tr. loss: {loss.item():.2E}",
            end="\r",
        )

    loss_avg = batch_losses.mean().item()
    train_loss_avg[epoch - 1] = loss_avg
    loss_std = batch_losses.std().item()
    print(
        f"\nEpoch [{epoch:04}/{num_epochs:04}] tr.  loss: {loss_avg:.2E} +- {loss_std:.2E}"
    )

    # validation loop
    model.eval()
    with torch.no_grad():
        val_losses = torch.zeros(len(val_loader))
        batch_psnr = torch.zeros(len(val_loader))
        for batch_idx, batch in enumerate(val_loader):
            val_x = batch["input"].to(device)
            val_target = batch["target"].to(device)
            val_lm_pet_lin_ops = batch["lm_pet_lin_ops"]
            val_contamination_lists = batch["contamination_lists"]
            val_adjoint_ones = batch["adjoint_ones"]
            val_diag_preconds = batch["diag_preconds"]

            val_output = model(
                val_x,
                val_lm_pet_lin_ops,
                val_contamination_lists,
                val_adjoint_ones,
                val_diag_preconds,
            )
            val_loss = criterion(val_output, val_target)
            val_losses[batch_idx] = val_loss.item()
            batch_psnr[batch_idx] = psnr(val_output, val_target)

            # plot input, output, and target for the current validation batch
            plot_batch_input_output_target(
                val_x,
                val_output,
                val_target,
                model_dir,
                prefix=f"val_sample_batch_{batch_idx:03}",
            )

        val_loss_avg[epoch - 1] = val_losses.mean().item()
        val_psnr[epoch - 1] = batch_psnr.mean().item()

        print(
            f"Epoch [{epoch:04}/{num_epochs:04}] val. loss: {val_loss_avg[epoch - 1]:.2E}"
        )
        print(
            f"Epoch [{epoch:04}/{num_epochs:04}] val. PSNR: {val_psnr[epoch - 1]:.2E}"
        )

    # save model checkpoint
    torch.save(
        {
            "model_kwargs": model_kwargs,
            "num_blocks": num_blocks,
            "denoiser_model_path": str(denoiser_model_path),
            "denoiser_model_class": denoiser_model_class,
            "denoiser_model_kwargs": denoiser_model_kwargs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_pnsr": val_psnr[epoch - 1],
            "val_loss": val_loss_avg,
            "train_loss": train_loss_avg[epoch - 1],
        },
        model_dir / f"epoch_{epoch:04}.pth",
    )

    # if the current val_psnr is the best so far, save the model
    if epoch == 1 or val_psnr[epoch - 1] > val_psnr[: epoch - 1].max():
        best_model_path = model_dir / "best.pth"
        epoch_model_path = model_dir / f"epoch_{epoch:04}.pth"
        # Remove existing symlink if it exists
        if best_model_path.exists() or best_model_path.is_symlink():
            best_model_path.unlink()
        # Create a symlink pointing to the current epoch checkpoint
        best_model_path.symlink_to(epoch_model_path.name)
        print(f"Best model symlinked to {best_model_path} -> {epoch_model_path.name}")

    # plot the validation PSNR
    fig, ax = plt.subplots(3, 2, layout="constrained", figsize=(12, 6), sharex="col")
    epochs = range(1, epoch + 1)
    ax[0, 0].semilogy(epochs, train_loss_avg[:epoch].cpu().numpy())
    ax[1, 0].semilogy(epochs, val_loss_avg[:epoch].cpu().numpy())
    ax[2, 0].plot(epochs, val_psnr[:epoch].cpu().numpy())

    ax[0, 1].loglog(epochs, train_loss_avg[:epoch].cpu().numpy())
    ax[1, 1].loglog(epochs, val_loss_avg[:epoch].cpu().numpy())
    ax[2, 1].semilogx(epochs, val_psnr[:epoch].cpu().numpy())

    ax[0, 0].set_ylabel("training loss")
    ax[1, 0].set_ylabel("validation loss")
    ax[2, 0].set_ylabel("validation PSNR (dB)")

    ax[-1, 0].set_xlabel("Epoch")
    ax[-1, 1].set_xlabel("Epoch")

    for axx in ax.ravel():
        axx.grid(ls=":")
    fig.savefig(model_dir / "current_metrics.pdf")
    plt.close(fig)
