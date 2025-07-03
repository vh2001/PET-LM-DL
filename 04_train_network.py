import argparse
import matplotlib.pyplot as plt

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import plot_batch_input_output_target, MiniConvNet, LMNet


# input parameters
parser = argparse.ArgumentParser(description="Train LMNet for PET image reconstruction")
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument(
    "--num_blocks",
    type=int,
    default=4,
    help="Number of MiniConvNet blocks in LMNet",
)
parser.add_argument(
    "--num_features",
    type=int,
    default=16,
    help="Number of features in each MiniConvNet (hidden) layer",
)
parser.add_argument(
    "--num_hidden_layers",
    type=int,
    default=1,
    help="Number of hidden layers in each MiniConvNet",
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
    "--skip_data_fidelity",
    action="store_true",
    help="skip gradient descent data fidelity steps",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.02,
    help="slope parameter for smooth non-linearity (negative part) for MiniConvNets",
)
parser.add_argument(
    "--beta",
    type=float,
    default=4.0,
    help="curvature parameter for smooth non-linearity (at 0) for MiniConvNets",
)

args = parser.parse_args()

seed = args.seed
num_epochs = args.num_epochs
lr = args.lr
# number of blocks in the LMNet
num_blocks = args.num_blocks
# number of features in each MiniConvNet layer
num_features = args.num_features
# number of hidden layers in each MiniConvNet
num_hidden_layers = args.num_hidden_layers
num_training_samples = args.num_training_samples
tr_batch_size = args.tr_batch_size
num_validation_samples = args.num_validation_samples
val_batch_size = args.val_batch_size
weight_sharing = args.weight_sharing
skip_data_fidelity: bool = args.skip_data_fidelity
alpha: float = args.alpha
beta: float = args.beta

# create a directory for the model checkpoints
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = Path(f"checkpoints/lmnet_run_{run_id}")
model_dir.mkdir(parents=True, exist_ok=True)

# auto convert the Namespace args to a dictionary
args_dict = vars(args)
# save the arguments to a json file
with open(model_dir / "args.json", "w", encoding="UTF8") as f:
    json.dump(args_dict, f, indent=4)
print(f"Arguments saved to {model_dir / 'args.json'}")

# %%

torch.manual_seed(seed)

# setup the training data loader
train_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[
    :num_training_samples
]
train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=tr_batch_size,
    collate_fn=brainweb_collate_fn,
    drop_last=True,
)

# setup the validation data loader
val_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[
    num_training_samples : num_training_samples + num_validation_samples
]
val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    collate_fn=brainweb_collate_fn,
)

# model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if weight_sharing:
    # if weight sharing is used, create a single MiniConvNet and use it for all blocks
    conv_nets = MiniConvNet(
        num_features=num_features,
        num_hidden_layers=num_hidden_layers,
        alpha=alpha,
        beta=beta,
    )
else:
    # setup a list of mini conv nets - defined in utils.py
    conv_nets = torch.nn.ModuleList(
        [
            MiniConvNet(
                num_features=num_features,
                num_hidden_layers=num_hidden_layers,
                alpha=alpha,
                beta=beta,
            )
            for _ in range(num_blocks)
        ]
    )

# setup the LMNet model - defined in utils.py
model = LMNet(
    conv_nets=conv_nets, num_blocks=num_blocks, use_data_fidelity=not skip_data_fidelity
).to(device)

# setup the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

psnr = PeakSignalNoiseRatio().to(device)

# save model architecture
with open(model_dir / "lmnet_architecture.txt", "w", encoding="UTF8") as f:
    f.write(str(model))
print(f"Model architecture saved to {model_dir / 'lmnet_architecture.txt'}")

val_psnr = torch.zeros(num_epochs)
val_loss_avg = torch.zeros(num_epochs)
train_loss_avg = torch.zeros(num_epochs)

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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_pnsr": val_psnr[epoch - 1],
            "val_loss": val_loss_avg,
        },
        model_dir / f"lmnet_epoch_{epoch:04}.pth",
    )

    # if the current val_psnr is the best so far, save the model
    if epoch == 1 or val_psnr[epoch - 1] > val_psnr[: epoch - 1].max():
        best_model_path = model_dir / "lmnet_best.pth"
        epoch_model_path = model_dir / f"lmnet_epoch_{epoch:04}.pth"
        # Remove existing symlink if it exists
        if best_model_path.exists() or best_model_path.is_symlink():
            best_model_path.unlink()
        # Create a symlink pointing to the current epoch checkpoint
        best_model_path.symlink_to(epoch_model_path.name)
        print(f"Best model symlinked to {best_model_path} -> {epoch_model_path.name}")

    # plot the validation PSNR
    fig, ax = plt.subplots(3, 1, layout="constrained", figsize=(6, 6), sharex=True)
    epochs = range(1, epoch + 1)
    ax[0].semilogy(epochs, train_loss_avg[:epoch].cpu().numpy())
    ax[1].semilogy(epochs, val_loss_avg[:epoch].cpu().numpy())
    ax[2].plot(epochs, val_psnr[:epoch].cpu().numpy())

    ax[0].set_ylabel("training loss")
    ax[1].set_ylabel("validation loss")
    ax[2].set_ylabel("validation PSNR (dB)")

    ax[2].set_xlabel("Epoch")
    for axx in ax.ravel():
        axx.grid(ls=":")
    fig.savefig(model_dir / "current_metrics.pdf")
    plt.close(fig)
