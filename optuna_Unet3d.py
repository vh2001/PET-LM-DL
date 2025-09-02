import optuna

import argparse
import matplotlib.pyplot as plt

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchmetrics.image import PeakSignalNoiseRatio

from data_utils import BrainwebLMPETDataset
from utils import plot_batch_input_output_target
from models import UNet3D


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
parser = argparse.ArgumentParser(description="Train image to image denoiser model")

parser.add_argument(
    "--model_kwargs",
    type=parse_json_model_kwargs,
    default={},
    help="JSON-encoded dict of init args for the model, e.g. "
    '\'{"features":[8,16,32]}\' or \'{"num_layers":3,"out_dim":256}\'',
)

parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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
parser.add_argument(
    "--print_gradient_norms",
    action="store_true",
    help="Print gradient norms during training (useful for debugging)",
)
parser.add_argument(
    "--count_level", type=float, default=1.0, help="count level of input data"
)
parser.add_argument(
    "--custom_tag",
    type=str,
    default="",
    help="Custom tag for the experiment",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training (e.g., 'cuda' or 'cpu')",
)

args = parser.parse_args()

model_class= "Unet3D"
model_kwargs = args.model_kwargs

seed = args.seed
count_level = args.count_level
num_epochs = args.num_epochs
lr = args.lr
num_training_samples = args.num_training_samples
tr_batch_size = args.tr_batch_size
num_validation_samples = args.num_validation_samples
val_batch_size = args.val_batch_size
print_gradient_norms: bool = args.print_gradient_norms
custom_tag: str = args.custom_tag
device = torch.device(args.device)






# %%
# setup of data loaders
################################################################################

torch.manual_seed(seed)

all_data_dirs = sorted(
    list(Path("data/sim_pet_data").glob(f"subject*_countlevel_{count_level:.1f}_*"))
)

# setup the training data loader
train_dirs = all_data_dirs[:num_training_samples]



# make sure that skip_raw_data is set to True
# in this case the loader only returns the OSEM input image and the ground truth which is faster
train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True, skip_raw_data=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=tr_batch_size,
    drop_last=True,
)

# setup the validation data loader
val_dirs = all_data_dirs[
    num_training_samples : num_training_samples + num_validation_samples
]



# make sure that skip_raw_data is set to True
# in this case the loader only returns the OSEM input image and the ground truth which is faster
val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=True)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
)


def objective(trial):
    num_levels = trial.suggest_int('num_levels', 2, 4)
    base_features = trial.suggest_categorical('base_features', [4,8, 16, 32,64])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Generate features list
    features = [base_features * (2**i) for i in range(num_levels)]

    # create a directory for the model checkpoints
    if custom_tag != "":
        run_id = f"{custom_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"optuna_models_UNet3D/features_{features}_run_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # auto convert the Namespace args to a dictionary
    args_dict = vars(args)
    # save the arguments to a json file
    with open(model_dir / "args.json", "w", encoding="UTF8") as f:
        json.dump(args_dict, f, indent=4)
    print(f"Arguments saved to {model_dir / 'args.json'}")


    # write the training directories to a json file
    with open(model_dir / "train_dirs.json", "w", encoding="UTF8") as f:
        json.dump([str(d) for d in train_dirs], f, indent=4)

    # write the validation directories to a json file
    with open(model_dir / "val_dirs.json", "w", encoding="UTF8") as f:
        json.dump([str(d) for d in val_dirs], f, indent=4)

    model = UNet3D(
        features=features,
    ).to(device)

    # setup the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    psnr = PeakSignalNoiseRatio().to(device)

    # save model architecture
    val_psnr = torch.zeros(num_epochs)
    val_loss_avg = torch.zeros(num_epochs)
    train_loss_avg = torch.zeros(num_epochs)

    model.train()
    for epoch in range(1, num_epochs + 1):
        batch_losses = torch.zeros(len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(x)
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

                val_output = model(val_x)
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
                "model_class": model_class,
                "model_kwargs": {"features": features},
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_pnsr": val_psnr[epoch - 1],
                "val_loss": val_loss_avg,
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

        # plot and save training and validation loss and val. PSNR
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

    # clear gpu memory
    torch.cuda.empty_cache()
    return val_psnr[epoch - 1]

if __name__ == "__main__":
    # create an Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"Unet3D_{model_class}",
    )
    # optimize the objective function
    study.optimize(objective, n_trials=2)

    # print the best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    # save the study 

    
    # save study
    import joblib
    joblib.dump(study, f"optuna_models_UNet3D/study_optuna_models_Unet3D.pkl")


