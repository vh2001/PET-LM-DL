import torch

from data_utils import load_lm_pet_data
from utils import LMNegPoissonLogLGradientLayer
from pathlib import Path

odirs = sorted(list(Path("data/sim_pet_data").glob("subject*")))

lm_pet_lin_op1, contamination_list1, adjoint_ones1, x_true1 = load_lm_pet_data(odirs[3])
lm_pet_lin_op2, contamination_list2, adjoint_ones2, x_true2 = load_lm_pet_data(
    odirs[11]
)

# stack x_true1 and x_true2
x = torch.stack((x_true1, x_true2), dim=0).unsqueeze(1)
x.requires_grad = True

contamination_lists = [contamination_list1, contamination_list2]
adjoint_ones = [adjoint_ones1, adjoint_ones2]
lm_pet_lin_ops = [lm_pet_lin_op1, lm_pet_lin_op2]
diag_preconds = [x_true1 / adjoint_ones1 + 1e-6, x_true2 / adjoint_ones2 + 1e-6]

# %%

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
