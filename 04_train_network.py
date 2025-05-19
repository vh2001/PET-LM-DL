import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import LMNegPoissonLogLGradientLayer, plot_batch_input_output_target


# 2. Define a mini 3D conv net block
class MiniConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=3, padding="same"),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding="same"),
            nn.PReLU(),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding="same"),
            nn.Conv3d(num_features, out_channels, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        return self.conv(x)


# 3. Define the full model
class LMNet(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__()
        self.n_blocks = n_blocks
        self.convs = nn.ModuleList([MiniConvNet() for _ in range(n_blocks)])
        self.lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply

    def forward(
        self, x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
    ):

        # PET images can have arbitrary global scales, but we don't want to
        # normalize before calculating the log-likelihood gradient
        # instead we calculate scales of all images in the batch and apply
        # them only before using the neural network

        # as scale we use the mean of the input images
        # if we are using early stopped OSEM images, the mean is well defined
        # and stable

        sample_scales = x.mean(dim=(2, 3, 4), keepdim=True)

        for i in range(self.n_blocks):
            x = x - self.lm_logL_grad_layer(
                x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
            )
            x = nn.ReLU()(x - self.convs[i](x / sample_scales) * sample_scales)
        return x


if __name__ == "__main__":
    # input parameters
    parser = argparse.ArgumentParser(
        description="Train LMNet for PET image reconstruction"
    )
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
        "--num_training_samples",
        type=int,
        default=30,
        help="Number of training samples",
    )
    parser.add_argument(
        "--tr_batch_size", type=int, default=5, help="Training batch size"
    )
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

    args = parser.parse_args()

    seed = args.seed
    num_epochs = args.num_epochs
    lr = args.lr
    num_blocks = args.num_blocks
    num_training_samples = args.num_training_samples
    tr_batch_size = args.tr_batch_size
    num_validation_samples = args.num_validation_samples
    val_batch_size = args.val_batch_size

    # create a directory for the model checkpoints
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"checkpoints/lmnet_run_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # save args to a json file
    with open(model_dir / "args.json", "w", encoding="UTF8") as f:
        f.write(str(args))
    print(f"Args saved to {model_dir / 'args.json'}")

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
    model = LMNet(n_blocks=num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    psnr = PeakSignalNoiseRatio().to(device)

    # save model architecture
    with open(model_dir / "lmnet_architecture.txt", "w", encoding="UTF8") as f:
        f.write(str(model))
    print(f"Model architecture saved to {model_dir / 'lmnet_architecture.txt'}")

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
        loss_std = batch_losses.std().item()
        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] tr.  loss: {loss_avg:.2E} +- {loss_std:.2E}"
        )

        # validation loop
        model.eval()
        with torch.no_grad():
            val_losses = torch.zeros(len(val_loader))
            val_psnr = torch.zeros(len(val_loader))
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
                val_psnr[batch_idx] = psnr(val_output, val_target)

                # plot input, output, and target for the current validation batch
                plot_batch_input_output_target(
                    val_x,
                    val_output,
                    val_target,
                    model_dir,
                    prefix=f"val_sample_batch_{batch_idx:03}",
                )

            val_loss_avg = val_losses.mean().item()
            val_loss_std = val_losses.std().item()

            val_psnr_avg = val_psnr.mean().item()
            val_psnr_std = val_psnr.std().item()

            print(
                f"Epoch [{epoch:04}/{num_epochs:04}] val. loss: {val_loss_avg:.2E} +- {val_loss_std:.2E}"
            )
            print(
                f"Epoch [{epoch:04}/{num_epochs:04}] val. PSNR: {val_psnr_avg:.2E} +- {val_psnr_std:.2E}"
            )

        # save model checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            model_dir / f"lmnet_epoch_{epoch:04}.pth",
        )
