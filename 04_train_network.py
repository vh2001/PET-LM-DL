import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import LMNegPoissonLogLGradientLayer


# 2. Define a mini 3D conv net block
class MiniConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, num_features, 1, padding="same"),
            nn.Conv3d(num_features, num_features, 1, padding="same"),
            nn.PReLU(),
            nn.Conv3d(num_features, num_features, 1, padding="same"),
            nn.Conv3d(num_features, out_channels, 1, padding="same"),
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
        for i in range(self.n_blocks):
            x = x - self.lm_logL_grad_layer(
                x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
            )
            x = nn.ReLU()(x - self.convs[i](x))
        return x


if __name__ == "__main__":
    # input parameters
    seed = 42
    num_epochs = 100
    batch_size = 3
    lr = 3e-4
    num_training_samples = 30
    num_blocks = 4

    torch.manual_seed(seed)

    # use first 20 data sets for training
    train_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[
        :num_training_samples
    ]
    train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=brainweb_collate_fn
    )

    # 4. Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LMNet(n_blocks=num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Training loop (1 epoch for demo)

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, 1):
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
            running_loss += loss.item()

            print(
                f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}]",
                end="\r",
            )

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch}/{num_epochs}] Average Training Loss: {avg_loss:.4f}")
