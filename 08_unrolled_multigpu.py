import os
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import shutil  # Added for copying the best model checkpoint

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image import PeakSignalNoiseRatio
import parallelproj

# --- Helper imports from original separate files ---
from utils import plot_batch_input_output_target
from models import DENOISER_MODEL_REGISTRY, LMNet

# ==============================================================================
# SECTION 1: DATA UTILITY FUNCTIONS AND CLASSES
# ==============================================================================

def load_lm_pet_data(odir: Path, skip_raw_data: bool = False, device: str | torch.device = "cpu"):
    """
    Loads PET listmode data and returns the linear operator, contamination list, and adjoint ones.
    This version creates all tensors and operators on the specified device.
    """
    if skip_raw_data:
        return torch.load(odir / "ground_truth.pt").to(device)
    else:
        data_tensors = torch.load(odir / "data_tensors.pt")
        ground_truth = torch.load(odir / "ground_truth.pt")

        x_true = ground_truth.to(device)
        event_start_coords = data_tensors["event_start_coords"].to(device)
        event_end_coords = data_tensors["event_end_coords"].to(device)
        event_tofbins = data_tensors["event_tofbins"].to(device)
        att_list = data_tensors["att_list"].to(device)
        contamination_list = data_tensors["contamination_list"].to(device)
        adjoint_ones = data_tensors["adjoint_ones"].to(device)

        with open(odir / "projector_parameters.json", "r", encoding="UTF8") as f:
            projector_parameters = json.load(f)

        in_shape = tuple(projector_parameters["in_shape"])
        voxel_size = tuple(projector_parameters["voxel_size"])
        img_origin = tuple(projector_parameters["img_origin"])
        fwhm_data_mm = projector_parameters["fwhm_data_mm"]
        tof_parameters = parallelproj.TOFParameters(
            **projector_parameters["tof_parameters"]
        )

        lm_proj = parallelproj.ListmodePETProjector(
            event_start_coords,
            event_end_coords,
            in_shape,
            voxel_size,
            img_origin,
        )

        lm_proj.tof_parameters = tof_parameters
        lm_proj.event_tofbins = event_tofbins
        lm_proj.tof = True

        lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

        res_model = parallelproj.GaussianFilterOperator(
            in_shape,
            sigma=fwhm_data_mm
            / (2.35 * torch.tensor(voxel_size, dtype=torch.float32, device=device)),
        )

        lm_pet_lin_op = parallelproj.CompositeLinearOperator(
            (lm_att_op, lm_proj, res_model)
        )

        return lm_pet_lin_op, contamination_list, adjoint_ones, x_true


class BrainwebLMPETDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dirs: list[Path], shuffle: bool = True, skip_raw_data: bool = False, device: str | torch.device = "cpu"
    ):
        self._data_dirs = data_dirs.copy()
        if shuffle:
            idx = torch.randperm(len(self._data_dirs))
            self._data_dirs = [self._data_dirs[i] for i in idx]
        self._skip_raw_data = skip_raw_data
        self._device = device

    @property
    def data_dirs(self):
        return self._data_dirs

    def __len__(self):
        return len(self._data_dirs)

    def __getitem__(self, idx):
        odir = self._data_dirs[idx]
        input_img = torch.load(odir / "mlem_reconstructions.pt")[
            "x_mlem_early_filtered"
        ].to(self._device)

        if self._skip_raw_data:
            target_img = load_lm_pet_data(odir, skip_raw_data=True, device=self._device)
            return { "input": input_img.unsqueeze(0), "target": target_img.unsqueeze(0) }
        else:
            lm_pet_lin_op, contamination_list, adjoint_ones, x_true = load_lm_pet_data(
                odir, device=self._device
            )
            return {
                "input": input_img, "target": x_true, "lm_pet_lin_op": lm_pet_lin_op,
                "contamination_list": contamination_list, "adjoint_ones": adjoint_ones,
            }


def brainweb_collate_fn(batch):
    return {
        "input": torch.stack([item["input"].unsqueeze(0) for item in batch]),
        "target": torch.stack([item["target"].unsqueeze(0) for item in batch]),
        "lm_pet_lin_ops": [item["lm_pet_lin_op"] for item in batch],
        "contamination_lists": [item["contamination_list"] for item in batch],
        "adjoint_ones": torch.stack([item["adjoint_ones"] for item in batch]),
        "diag_preconds": torch.stack([
            item["input"] / item["adjoint_ones"] + 1e-6 for item in batch
        ]),
    }

# ==============================================================================
# SECTION 2: MAIN SCRIPT LOGIC
# ==============================================================================

def setup_distributed_environment():
    if not dist.is_available() or not torch.cuda.is_available():
        print("Distributed training not available or CUDA not found.", file=sys.stderr)
        sys.exit(1)
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"Initialized process {rank}/{world_size} on GPU {device_id}")
    return rank, world_size, device_id


def parse_json_model_kwargs(arg: str):
    try:
        obj = json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --model-kwargs: {e}")
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("--model-kwargs must be a JSON object (i.e. dict)")
    return obj


def main():
    rank, world_size, device_id = setup_distributed_environment()
    device = torch.device(f"cuda:{device_id}")

    parser = argparse.ArgumentParser(description="Train LMNet for PET image reconstruction using DDP")
    parser.add_argument("denoiser_model_path", type=str, help="checkpoint path to pretraind denoiser model")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of unrolled blocks")
    parser.add_argument("--count_level", type=float, default=1.0, help="Count level of simulated PET data")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model_kwargs", type=parse_json_model_kwargs, default={}, help="JSON-encoded dict of init args for the unrolled model")
    parser.add_argument("--num_training_samples", type=int, default=30, help="Number of training samples")
    parser.add_argument("--tr_batch_size", type=int, default=5, help="Per-GPU training batch size")
    parser.add_argument("--num_validation_samples", type=int, default=5, help="Number of validation samples")
    parser.add_argument("--val_batch_size", type=int, default=5, help="Per-GPU validation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weight_sharing", action="store_true", help="Use weight sharing")
    parser.add_argument("--print_gradient_norms", action="store_true", help="Print gradient norms during training")
    parser.add_argument("--custom_tag", type=str, default="", help="Custom tag for the experiment")
    parser.add_argument("--adaptive_lr", action="store_true", help="Use adaptive learning rate")
    args = parser.parse_args()

    # --- Configuration ---
    seed, num_epochs, lr = args.seed, args.num_epochs, args.lr
    num_training_samples, tr_batch_size = args.num_training_samples, args.tr_batch_size
    num_validation_samples, val_batch_size = args.num_validation_samples, args.val_batch_size
    denoiser_model_path, num_blocks, count_level = Path(args.denoiser_model_path), args.num_blocks, args.count_level
    model_kwargs, custom_tag, adaptive_lr = args.model_kwargs, args.custom_tag, args.adaptive_lr

    model_dir = None
    if rank == 0:
        run_id = f"{custom_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if custom_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"checkpoints_unrolled/lmnet_run_{run_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        args_dict = vars(args)
        args_dict['world_size'] = world_size
        with open(model_dir / "args.json", "w", encoding="UTF8") as f:
            json.dump(args_dict, f, indent=4)
        print(f"Arguments saved to {model_dir / 'args.json'}")

    torch.manual_seed(seed)
    all_data_dirs = sorted(list(Path("data/sim_pet_data").glob(f"subject*_countlevel_{count_level:.1f}_*")))
    train_dirs = all_data_dirs[:num_training_samples]
    val_dirs = all_data_dirs[num_training_samples : num_training_samples + num_validation_samples]

    if rank == 0:
        with open(model_dir / "train_dirs.json", "w", encoding="UTF8") as f:
            json.dump([str(d) for d in train_dirs], f, indent=4)
        with open(model_dir / "val_dirs.json", "w", encoding="UTF8") as f:
            json.dump([str(d) for d in val_dirs], f, indent=4)

    train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=False, skip_raw_data=False, device=device)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=tr_batch_size, sampler=train_sampler, collate_fn=brainweb_collate_fn, drop_last=True)

    val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=False, device=device)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, collate_fn=brainweb_collate_fn)

    denoiser_checkpoint = torch.load(denoiser_model_path, map_location="cpu")
    if denoiser_checkpoint["model_class"] == "Unet3D":
        denoiser_checkpoint["model_class"] = "UNet3D"
    denoiser_model_class, denoiser_model_kwargs = denoiser_checkpoint["model_class"], denoiser_checkpoint["model_kwargs"]
    denoiser_model = DENOISER_MODEL_REGISTRY[denoiser_model_class](**denoiser_model_kwargs)
    denoiser_model.load_state_dict(denoiser_checkpoint["model_state_dict"])

    # weight_sharing = model_kwargs.get("weight_sharing", False)
    weight_sharing = args.weight_sharing
    model_kwargs['weight_sharing'] = weight_sharing
    conv_nets = torch.nn.ModuleList([denoiser_model] if weight_sharing else [denoiser_model for _ in range(num_blocks)])

    model = LMNet(conv_nets, num_blocks, **model_kwargs).to(device)
    model = DDP(model, device_ids=[device_id])

    if rank == 0:
        with open(model_dir / "lmnet_architecture.txt", "w", encoding="UTF8") as f:
            f.write(str(model.module))
        print(f"Model architecture saved to {model_dir / 'lmnet_architecture.txt'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    psnr = PeakSignalNoiseRatio(data_range=(0, 4)).to(device)

    if rank == 0:
        val_psnr = torch.zeros(num_epochs)
        val_loss_avg = torch.zeros(num_epochs)
        train_loss_avg = torch.zeros(num_epochs)
        learning_rates = torch.zeros(num_epochs)

    scheduler = None
    if adaptive_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)
        if rank == 0: print("Using adaptive learning rate with CosineAnnealingWarmRestarts scheduler.")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        if rank == 0: learning_rates[epoch - 1] = optimizer.param_groups[0]['lr']
        batch_losses = torch.zeros(len(train_loader), device=device)

        for batch_idx, batch in enumerate(train_loader):
            x, target = batch["input"], batch["target"]
            lm_pet_lin_ops, contamination_lists = batch["lm_pet_lin_ops"], batch["contamination_lists"]
            adjoint_ones, diag_preconds = batch["adjoint_ones"], batch["diag_preconds"]

            optimizer.zero_grad()
            output = model(x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_losses[batch_idx] = loss.item()
            if rank == 0:
                print(f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(train_loader):03}] - tr. loss: {loss.item():.2E}", end="\r")

        dist.all_reduce(batch_losses, op=dist.ReduceOp.AVG)
        if rank == 0:
            loss_avg = batch_losses.mean().item()
            train_loss_avg[epoch - 1] = loss_avg
            print(f"\nEpoch [{epoch:04}/{num_epochs:04}] tr. loss: {loss_avg:.2E}")

        if scheduler: scheduler.step()

        model.eval()
        val_losses_proc = torch.zeros(len(val_loader), device=device)
        batch_psnr_proc = torch.zeros(len(val_loader), device=device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_x, val_target = batch["input"], batch["target"]
                val_lm_pet_lin_ops, val_contamination_lists = batch["lm_pet_lin_ops"], batch["contamination_lists"]
                val_adjoint_ones, val_diag_preconds = batch["adjoint_ones"], batch["diag_preconds"]

                val_output = model(val_x, val_lm_pet_lin_ops, val_contamination_lists, val_adjoint_ones, val_diag_preconds)
                val_loss = criterion(val_output, val_target)

                val_losses_proc[batch_idx] = val_loss.item()
                batch_psnr_proc[batch_idx] = psnr(val_output, val_target)

                # --- MODIFICATION: Only plot the first validation batch and overwrite the file ---
                if rank == 0 and batch_idx == 0:
                    plot_batch_input_output_target(
                        val_x, val_output, val_target, model_dir,
                        prefix="current_validation_batch" # Static filename
                    )

        dist.all_reduce(val_losses_proc, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_psnr_proc, op=dist.ReduceOp.AVG)

        # --- MODIFICATION: Overhaul of the saving logic ---
        if rank == 0:
            current_val_loss = val_losses_proc.mean().item()
            current_val_psnr = batch_psnr_proc.mean().item()
            val_loss_avg[epoch - 1] = current_val_loss
            val_psnr[epoch - 1] = current_val_psnr

            print(f"Epoch [{epoch:04}/{num_epochs:04}] val. loss: {current_val_loss:.2E}")
            print(f"Epoch [{epoch:04}/{num_epochs:04}] val. PSNR: {current_val_psnr:.2F} dB")

            # Define paths for the two checkpoints
            current_checkpoint_path = model_dir / "current.pth"
            best_model_path = model_dir / "best.pth"

            # Prepare checkpoint dictionary
            checkpoint_data = {
                "model_kwargs": model_kwargs, "num_blocks": num_blocks,
                "denoiser_model_path": str(denoiser_model_path),
                "denoiser_model_class": denoiser_model_class,
                "denoiser_model_kwargs": denoiser_model_kwargs,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "val_pnsr": current_val_psnr,
                "val_loss": current_val_loss,
                "train_loss": train_loss_avg[epoch - 1].item(),
            }

            # Save the current model state, overwriting the previous one
            torch.save(checkpoint_data, current_checkpoint_path)

            # If the current model is the best, copy 'current.pth' to 'best.pth'
            if epoch == 1 or current_val_psnr > val_psnr[: epoch - 1].max():
                print(f"New best model found! Saving to {best_model_path}")
                shutil.copy(current_checkpoint_path, best_model_path)

            # Plotting metrics (this part remains the same)
            fig, ax = plt.subplots(4, 2, layout="constrained", figsize=(12, 8), sharex="col")
            epochs_range = range(1, epoch + 1)
            ax[0, 0].semilogy(epochs_range, train_loss_avg[:epoch].cpu().numpy())
            ax[1, 0].semilogy(epochs_range, val_loss_avg[:epoch].cpu().numpy())
            ax[2, 0].plot(epochs_range, val_psnr[:epoch].cpu().numpy())
            ax[3, 0].semilogy(epochs_range, learning_rates[:epoch].cpu().numpy())
            ax[0, 1].loglog(epochs_range, train_loss_avg[:epoch].cpu().numpy())
            ax[1, 1].loglog(epochs_range, val_loss_avg[:epoch].cpu().numpy())
            ax[2, 1].semilogx(epochs_range, val_psnr[:epoch].cpu().numpy())
            ax[3, 1].loglog(epochs_range, learning_rates[:epoch].cpu().numpy())
            ax[0, 0].set_ylabel("training loss"); ax[1, 0].set_ylabel("validation loss")
            ax[2, 0].set_ylabel("validation PSNR (dB)"); ax[3, 0].set_ylabel("learning rate")
            ax[-1, 0].set_xlabel("Epoch"); ax[-1, 1].set_xlabel("Epoch")
            for axx in ax.ravel(): axx.grid(ls=":")
            fig.savefig(model_dir / "current_metrics.pdf")
            plt.close(fig)

    if rank == 0:
        print("Training finished.")


if __name__ == "__main__":
    main()

# ==============================================================================
# HOW TO RUN THIS SCRIPT
# ==============================================================================
# You must use the `torchrun` launcher.
#
# --nproc_per_node: Set this to the number of GPUs you want to use.
#
# Your arguments (like --tr_batch_size) are passed after the script name.
# The --tr_batch_size is the PER-GPU batch size.
#
# Example to run on 4 GPUs:
# torchrun --nproc_per_node=4 your_script_name.py path/to/denoiser.pth --tr_batch_size 1
# ==============================================================================