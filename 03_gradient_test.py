# this script tests whether the backward pass of the LMNegPoissonLogLGradientLayer is correct.
# Unfortunately, this tests takes very long - on data with the usual problem size

import torch
from torch.utils.data import DataLoader

# import multiprocessing as mp

from data_utils import BrainwebLMPETDataset, brainweb_collate_fn
from utils import LMNegPoissonLogLGradientLayer
from pathlib import Path

# mp.set_start_method('spawn', force=True)

batch_size = 2

train_dirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))[:5]
train_dataset = BrainwebLMPETDataset(train_dirs, shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=brainweb_collate_fn,
)
# %%

batch = next(iter(train_loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = batch["input"].to(device)
target = batch["target"].to(device)
lm_pet_lin_ops = batch["lm_pet_lin_ops"]
contamination_lists = batch["contamination_lists"]
adjoint_ones = batch["adjoint_ones"]
diag_preconds = batch["diag_preconds"]

x.requires_grad = True

lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply
logL_grads = lm_logL_grad_layer(
    x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
)

# %%
loss = (logL_grads * logL_grads).sum()
loss.backward()

# %%
from torch.autograd import gradcheck

print("Running gradient check")

test = gradcheck(
    lm_logL_grad_layer,
    (x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds),
    eps=1e-1,
    atol=1e-3,
    rtol=1e-3,
    fast_mode=False,
    nondet_tol=1e-7,
)
print(test)
