import os
import argparse
import sys
def setup_gpu_environment():
    """
    Parse GPU selection from command line and set up environment before any PyTorch imports.
    This must be called before importing torch or any other CUDA libraries.
    """
    # Create a simple parser just for GPU selection
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID to use (will be remapped to cuda:0)"
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to make visible (e.g., '0,1,2')"
    )
    
    # Parse only known args to avoid conflicts with your main parser
    args, _ = parser.parse_known_args()
    
    if args.visible_gpus is not None:
        # User specified custom GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        print(f"Set CUDA_VISIBLE_DEVICES to: {args.visible_gpus}")
    else:
        # Single GPU mode - map specified GPU to cuda:0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Remapping GPU {args.gpu_id} to cuda:0")
    
    return args.gpu_id


# Call this BEFORE any other imports
if __name__ == "__main__":
    selected_gpu = setup_gpu_environment()
import optuna
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchmetrics.image import PeakSignalNoiseRatio
from time import sleep
from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import plot_batch_input_output_target
from models import DENOISER_MODEL_REGISTRY, LMNet
from custom_loss_fn import LOSS_REGISTRY


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
# GPU selection arguments
parser.add_argument(
    "--gpu_id", 
    type=int, 
    default=0, 
    help="GPU ID to use (will be remapped to cuda:0)"
)
parser.add_argument(
    "--visible_gpus",
    type=str,
    default=None,
    help="Comma-separated list of GPU IDs to make visible (e.g., '0,1,2')"
)

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

parser.add_argument("--pre_train_epochs", type=int, default=20, help="Number of pre-training epochs")

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
    "--model_class",
    choices=DENOISER_MODEL_REGISTRY.keys(),
    help="Which model to train (e.g. UNet3D)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device to use for training (e.g., 'cuda' or 'cpu')",
)
parser.add_argument(
    "--adaptive_lr",
    action="store_true",
    help="Use adaptive learning rate",
)
parser.add_argument(
    "--loss_fn",
    type=str,
    default="mse",
    choices=list(LOSS_REGISTRY.keys()),
    help="Loss function to use (e.g., 'mse', 'l1', 'huber', etc.)", 
)
parser.add_argument("--weight_sharing", action="store_true", help="Use weight sharing")

args = parser.parse_args()

model_class: str = args.model_class
model_kwargs = args.model_kwargs

seed = args.seed
count_level = args.count_level
pre_train_epochs = args.pre_train_epochs
num_epochs = args.num_epochs
lr = args.lr
num_training_samples = args.num_training_samples
tr_batch_size = args.tr_batch_size
num_validation_samples = args.num_validation_samples
val_batch_size = args.val_batch_size
print_gradient_norms: bool = args.print_gradient_norms
custom_tag: str = args.custom_tag
adaptive_lr: bool = args.adaptive_lr
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
train_dataset_denoiser = BrainwebLMPETDataset(train_dirs, shuffle=True, skip_raw_data=True)
train_loader_denoiser = DataLoader(
    train_dataset_denoiser,
    batch_size=tr_batch_size,
    drop_last=True,
)

# setup the validation data loader
val_dirs = all_data_dirs[
    num_training_samples : num_training_samples + num_validation_samples
]



# make sure that skip_raw_data is set to True
# in this case the loader only returns the OSEM input image and the ground truth which is faster
val_dataset_denoiser = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=True)
val_loader_denoiser = DataLoader(
    val_dataset_denoiser,
    batch_size=val_batch_size,
)


train_dirs = all_data_dirs[:num_training_samples]


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



val_dataset = BrainwebLMPETDataset(val_dirs, shuffle=False, skip_raw_data=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    collate_fn=brainweb_collate_fn,
)

def objective(trial):

    num_blocks = trial.suggest_int("num_blocks", 1, 8)
    weight_sharing = trial.suggest_categorical("weight_sharing", [True, False])
    use_data_fidelity = trial.suggest_categorical("use_data_fidelity", [True, False])
    lr = trial.suggest_float('lr', 1e-6, 5e-4, log=True)
    # select features based on model class
    if model_class == "MiniConvNet":
        num_features = trial.suggest_categorical("num_features", [8,16, 32, 64])
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)
        
        model_kwargs = {
            "num_features": num_features,
            "num_hidden_layers": num_hidden_layers,
        }
    elif model_class == "UNet3D":
        num_levels = trial.suggest_int('num_levels', 1, 4)
        base_features = trial.suggest_categorical('base_features', [4,8, 16, 32])
        features = [base_features * 2 ** i for i in range(num_levels)]


        model_kwargs = {
            "features" : features
        }
    
    # create a directory for the model checkpoints
    if custom_tag != "":
        run_id = f"{custom_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"optuna_models_unrolled/features_{model_class}_layers_{num_blocks}_run_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    # auto convert the Namespace args to a dictionary
    args_dict = vars(args)
    # update args dict with hyperparameters
    args_dict.update({
        "num_blocks": num_blocks,
        "weight_sharing": weight_sharing,
        "use_data_fidelity": use_data_fidelity,
        **model_kwargs
    })

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

    denoiser_model = DENOISER_MODEL_REGISTRY[model_class](**model_kwargs).to(device)

    # setup the optimizer and loss function
    denoiser_optimizer = torch.optim.Adam(denoiser_model.parameters(), lr=lr)
    # criterion = torch.nn.MSELoss()
    criterion = LOSS_REGISTRY[args.loss_fn]()

    psnr = PeakSignalNoiseRatio().to(device)

    # save model architecture
    val_psnr = torch.zeros(num_epochs)
    val_loss_avg = torch.zeros(num_epochs)
    train_loss_avg = torch.zeros(num_epochs)
    learning_rates = torch.zeros(num_epochs)
    try:
        for epoch in range(1, pre_train_epochs + 1):

            
            denoiser_model.train()
            batch_losses = torch.zeros(len(train_loader_denoiser))
            for batch_idx, batch in enumerate(train_loader_denoiser):
                x = batch["input"].to(device)
                target = batch["target"].to(device)

                denoiser_optimizer.zero_grad()
                output = denoiser_model(x)
                loss = criterion(output, target)
                loss.backward()


                denoiser_optimizer.step()
                batch_losses[batch_idx] = loss.item()

                print(
                    f"Epoch(pre) [{epoch:04}/{pre_train_epochs:04}] Batch [{(batch_idx+1):03}/{len(train_loader_denoiser):03}] - tr. loss: {loss.item():.2E}",
                    end="\r",
                )

            loss_avg = batch_losses.mean().item()
            train_loss_avg[epoch - 1] = loss_avg
            loss_std = batch_losses.std().item()
            print(
                f"\nEpoch(pre) [{epoch:04}/{pre_train_epochs:04}] tr.  loss: {loss_avg:.2E} +- {loss_std:.2E}"
            )


            # validation loop
            denoiser_model.eval()
            with torch.no_grad():
                val_losses = torch.zeros(len(val_loader_denoiser))
                batch_psnr = torch.zeros(len(val_loader_denoiser))
                for batch_idx, batch in enumerate(val_loader_denoiser):
                    val_x = batch["input"].to(device)
                    val_target = batch["target"].to(device)

                    val_output = denoiser_model(val_x)
                    val_loss = criterion(val_output, val_target)
                    val_losses[batch_idx] = val_loss.item()
                    batch_psnr[batch_idx] = psnr(val_output, val_target)

                    # plot input, output, and target for the current validation batch
                    plot_batch_input_output_target(
                        val_x,
                        val_output,
                        val_target,
                        model_dir,
                        prefix=f"val_sample_batch_{batch_idx:03}_pretrain",
                    )

                val_loss_avg[epoch - 1] = val_losses.mean().item()
                val_psnr[epoch - 1] = batch_psnr.mean().item()

                print(
                    f"Epoch [{epoch:04}/{num_epochs:04}] val. loss: {val_loss_avg[epoch - 1]:.2E}"
                )
                print(
                    f"Epoch [{epoch:04}/{num_epochs:04}] val. PSNR: {val_psnr[epoch - 1]:.2E}"
                )

            # # save model checkpoint
            # torch.save(
            #     {
            #         "model_class": model_class,
            #         "model_kwargs": model_kwargs,
            #         "model_state_dict": denoiser_model.state_dict(),
            #         "optimizer_state_dict": denoiser_optimizer.state_dict(),
            #         "epoch": epoch,
            #         "val_pnsr": val_psnr[epoch - 1],
            #         "val_loss": val_loss_avg,
            #     },
            #     model_dir / f"epoch_{epoch:04}.pth",
            # )

            # if the current val_psnr is the best so far, save the model
            if epoch == 1 or val_psnr[epoch - 1] > val_psnr[: epoch - 1].max():
                torch.save(
                    {
                        "model_class": model_class,
                        "model_kwargs": model_kwargs,
                        "model_state_dict": denoiser_model.state_dict(),
                        "optimizer_state_dict": denoiser_optimizer.state_dict(),
                        "epoch": epoch,
                        "val_pnsr": val_psnr[epoch - 1],
                        "val_loss": val_loss_avg,
                    },
                    model_dir / "best_denoiser.pth",
                )
            best_model_path = model_dir / "best_denoiser.pth"
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
            fig.savefig(model_dir / "current_metrics_pretrain.pdf")
            plt.close(fig)



        # ########################### Train unrolled network ###########################
        # Add explicit cleanup here:
        print("Clearing pretrained denoiser from memory...")
        denoiser_model.cpu()  # Move to CPU first
        del denoiser_model
        del denoiser_optimizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

        # load best denoiser checkpoint
        denoiser_checkpoint = torch.load(best_model_path, map_location="cpu")
        denoiser_model = DENOISER_MODEL_REGISTRY[model_class](**model_kwargs)
        denoiser_model.load_state_dict(denoiser_checkpoint["model_state_dict"])
        denoiser_model.to(device)


        if weight_sharing:
            conv_nets = torch.nn.ModuleList([denoiser_model])
        else:
            conv_nets = torch.nn.ModuleList([denoiser_model for _ in range(num_blocks)])


        model = LMNet(conv_nets, num_blocks, use_data_fidelity, weight_sharing).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        val_psnr = torch.zeros(num_epochs)
        val_loss_avg = torch.zeros(num_epochs)
        train_loss_avg = torch.zeros(num_epochs)
        learning_rates = torch.zeros(num_epochs)
        # set lr scheduler if adaptive learning rate is enabled
        if adaptive_lr:
            # use a simple step LR scheduler that reduces the learning rate by a factor of 10 every 30 epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10
                    )
            print("Using adaptive learning rate with StepLR scheduler.")
            # training loop
        model.train()
        for epoch in range(1, num_epochs + 1):
            learning_rates[epoch - 1] = optimizer.param_groups[0]['lr']
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

            if adaptive_lr:
                # step the scheduler with the validation loss
                scheduler.step()
                print(f"Learning rate adjusted by scheduler: {optimizer.param_groups[0]['lr']:.2E}")
            # save model checkpoint
            # if the current val_psnr is the best so far, save the model
        #     torch.save(
        #     {
        #         "model_kwargs": model_kwargs,
        #         "num_blocks": num_blocks,
        #         "denoiser_model_class": model_class,
        #         "denoiser_model_kwargs": model_kwargs,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "epoch": epoch,
        #         "val_pnsr": val_psnr[epoch - 1],
        #         "val_loss": val_loss_avg,
        #         "train_loss": train_loss_avg[epoch - 1],
        #     },
        #     model_dir / f"epoch_curr.pth",
        # )
            if epoch == 1 or val_psnr[epoch - 1] > val_psnr[: epoch - 1].max():
                torch.save(
                    {
                        "model_kwargs": model_kwargs,
                        "num_blocks": num_blocks,
                        "denoiser_model_class": model_class,
                        "denoiser_model_kwargs": model_kwargs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_pnsr": val_psnr[epoch - 1],
                        "val_loss": val_loss_avg,
                        "train_loss": train_loss_avg[epoch - 1],
                    },
                    model_dir / "best.pth",
                )
            best_model_path = model_dir / "best.pth"
            # plot the validation PSNR
            fig, ax = plt.subplots(4, 2, layout="constrained", figsize=(12, 8), sharex="col")
            epochs = range(1, epoch + 1)
            ax[0, 0].semilogy(epochs, train_loss_avg[:epoch].cpu().numpy())
            ax[1, 0].semilogy(epochs, val_loss_avg[:epoch].cpu().numpy())
            ax[2, 0].plot(epochs, val_psnr[:epoch].cpu().numpy())
            ax[3, 0].semilogy(epochs, learning_rates[:epoch].cpu().numpy())

            ax[0, 1].loglog(epochs, train_loss_avg[:epoch].cpu().numpy())
            ax[1, 1].loglog(epochs, val_loss_avg[:epoch].cpu().numpy())
            ax[2, 1].semilogx(epochs, val_psnr[:epoch].cpu().numpy())
            ax[3, 1].loglog(epochs, learning_rates[:epoch].cpu().numpy())

            ax[0, 0].set_ylabel("training loss")
            ax[1, 0].set_ylabel("validation loss")
            ax[2, 0].set_ylabel("validation PSNR (dB)")
            ax[3, 0].set_ylabel("learning rate")

            ax[-1, 0].set_xlabel("Epoch")
            ax[-1, 1].set_xlabel("Epoch")
            for axx in ax.ravel():
                axx.grid(ls=":")
            fig.savefig(model_dir / "current_metrics.pdf")
            plt.close(fig)
                    

            # clear torch memory
            torch.cuda.empty_cache()
            
        return val_psnr[epoch - 1]
   
    except Exception as e:
            print(f"Error during training or evaluation: {e}")
            # print full error
            print(sys.exc_info()[0])

            # Save the error to a file
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"{model_dir}/error_log_{timestamp_str}.txt", "a") as f:
                f.write(f"Error during training or evaluation: {e}\n")
                f.write(f"Hyperparameters: Model: {model_kwargs} Num_blocks: {num_blocks} \n")

            torch.cuda.empty_cache()  # Clear GPU memory
            sleep(2)
           
            return None

if __name__ == "__main__":
    # create an Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"MiniConvNet_{model_class}",
    )
    # optimize the objective function
    study.optimize(objective, n_trials=100)

    # print the best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    # save the study 
    # get time
    time= datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # save study
    import joblib
    joblib.dump(study, f"optuna_models_unrolled/optuna_study_{model_class}_{args.loss_fn}_{time}.pkl")


