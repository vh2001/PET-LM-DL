import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import LMNegPoissonLogLGradientLayer


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
    seed = 42
    num_epochs = 2  # 28
    batch_size = 6
    lr = 3e-4
    num_training_samples = 30
    num_blocks = 6  # number of unrolled blocks where each block does a data fidelity gradient step and a network step

    torch.manual_seed(seed)

    train_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[
        :num_training_samples
    ]
    train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=brainweb_collate_fn,
        drop_last=True,
    )

    # 4. Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LMNet(n_blocks=num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # create a unique output directory for the model checkpoints, use the current datetime string in the directory name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"checkpoints/lmnet_run_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)

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
                f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] - tr. loss: {loss.item():.3E}",
                end="\r",
            )

        loss_avg = batch_losses.mean().item()
        loss_std = batch_losses.std().item()
        print(
            f"\nEpoch [{epoch}/{num_epochs}] tr. loss: {loss_avg:.3E} +- {loss_std:.3E}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            model_dir / f"lmnet_epoch_{epoch:04}.pth",
        )
