import json
import argparse
import matplotlib.pyplot as plt
import parallelproj
import torch

from pathlib import Path
from data_utils import load_lm_pet_data

# using torch valid choices are 'cpu' or 'cuda'
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"

# %%
parser = argparse.ArgumentParser(
    description="Listmode MLEM reconstruction of simulated PET data"
)
parser.add_argument("odir", type=str, default=1, help="Seed for random noise generator")

parser.add_argument(
    "--fwhm_postfilter_mm",
    type=float,
    default=4.5,
    help="FWHM of Gaussian post-filter in mm",
)

args = parser.parse_args()

odir = Path(args.odir)

# fixed input parameters
num_iter: int = 100
num_iter_early: int = 20
fwhm_postfilter_mm: float = args.fwhm_postfilter_mm

lm_pet_lin_op, contamination_list, adjoint_ones, x_true = load_lm_pet_data(odir)


def neg_poisson_logL_grad(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the gradient of the negative Poisson log-likelihood
    for a random image x using the LM projector and contamination list.
    """
    # affine forward model evaluated at x
    z_lm = lm_pet_lin_op(x) + contamination_list

    # calculate the gradient
    lm_grad = adjoint_ones - lm_pet_lin_op.adjoint(1 / z_lm)

    return lm_grad


# %%
# run quick LM MLEM

print("Running reference listmode MLEM reconstruction")

x_mlem = torch.ones_like(x_true)
x_mlem_early = torch.zeros_like(x_true)

for i in range(num_iter):
    print(f"LM iteration {(i+1):03}/{num_iter}", end="\r")
    if i == num_iter_early:
        x_mlem_early = x_mlem.clone()

    neg_pl_grad = neg_poisson_logL_grad(x_mlem)
    step = x_mlem / adjoint_ones

    # mathematically the clipping / clamping of negative values
    # should not be needed using this step size.
    # however, due to small numerical errors, we need to clamp the negative values
    x_mlem = torch.clamp(x_mlem - step * neg_pl_grad, min=0.0)

print()

# setup a Gaussian post-filter operator
post_filter = parallelproj.GaussianFilterOperator(
    x_mlem.shape, sigma=fwhm_postfilter_mm / (2.35 * lm_pet_lin_op[1].voxel_size)
)

x_mlem_early_filtered = post_filter(x_mlem_early)
x_mlem_filtered = post_filter(x_mlem)

torch.save(
    {
        "x_mlem": x_mlem,
        "x_mlem_early": x_mlem_early,
        "x_mlem_filtered": x_mlem_filtered,
        "x_mlem_early_filtered": x_mlem_early_filtered,
    },
    odir / "mlem_reconstructions.pt",
)

with open(odir / "mlem_parameters.json", "w", encoding="UTF8") as f:
    json.dump(
        {
            "voxel_size": tuple(lm_pet_lin_op[1].voxel_size.tolist()),
            "img_origin": tuple(lm_pet_lin_op[1]._img_origin.tolist()),
            "fwhm_postfilter_mm": fwhm_postfilter_mm,
            "num_iter": num_iter,
            "num_iter_early": num_iter_early,
        },
        f,
        indent=4,
    )


# %%
# show a few images

sl0 = lm_pet_lin_op.in_shape[0] // 2
sl1 = lm_pet_lin_op.in_shape[1] // 2
sl2 = lm_pet_lin_op.in_shape[2] // 3


kws = dict(cmap="Greys", vmin=0, vmax=1.5 * float(torch.max(x_true)), origin="lower")

fig, ax = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
ax[0, 0].imshow(x_true[:, :, sl2].cpu().numpy().T, **kws)
ax[0, 1].imshow(x_true[sl0, :, :].cpu().numpy().T, **kws)
ax[0, 2].imshow(x_true[:, sl1, :].cpu().numpy().T, **kws)
ax[1, 0].imshow(x_mlem_early_filtered[:, :, sl2].cpu().numpy().T, **kws)
ax[1, 1].imshow(x_mlem_early_filtered[sl0, :, :].cpu().numpy().T, **kws)
ax[1, 2].imshow(x_mlem_early_filtered[:, sl1, :].cpu().numpy().T, **kws)
ax[2, 0].imshow(x_mlem_filtered[:, :, sl2].cpu().numpy().T, **kws)
ax[2, 1].imshow(x_mlem_filtered[sl0, :, :].cpu().numpy().T, **kws)
ax[2, 2].imshow(x_mlem_filtered[:, sl1, :].cpu().numpy().T, **kws)

for axx in ax.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

ax[0, 1].set_title("ground truth", fontsize="medium")
ax[1, 1].set_title(f"LM-MLEM 20 it. + {fwhm_postfilter_mm}mm filter", fontsize="medium")
ax[2, 1].set_title(
    f"LM-MLEM 100 it. + {fwhm_postfilter_mm}mm filter", fontsize="medium"
)

fig.show()

fig.savefig(odir / "mlem_recons.png", dpi=300)


# %%


## %%
## calculate what is needed for a pytorch negative Poisson logL gradient descent layer
## note that pet_lin_op in the current form, because of attenuation, is object dependent
## the same holds for adjoint_ones (should be precomputed and saved to disk)
#
## calculate the gradient of the negative Poisson log-likelihood for a random image x
#
# batch_size = 1
#
# x = torch.rand(
#    (batch_size, 1) + lm_pet_lin_op.in_shape,
#    device=dev,
#    dtype=torch.float32,
#    requires_grad=True,
# )
#
## affine forward model evaluated at x
# z_sino = pet_lin_op(x[0,...].squeeze().detach()) + contamination
#
# sino_grad = adjoint_ones - pet_lin_op.adjoint(y/z_sino)
#
## %%
## calculate the gradient of the negative Poisson log-likelihood using LM data and projector
#
# z_lm = lm_pet_lin_op(x[0,...].squeeze().detach()) + contamination_list
# lm_grad = adjoint_ones - lm_pet_lin_op.adjoint(1/z_lm)
#
## now sino_grad and lm_grad should be numerically very close demonstrating that
## both approaches are equivalent
## but for 3D real world low count PET data, the 2nd approach is much faster and
## more memory efficient
#
# assert torch.allclose(lm_grad,sino_grad, atol = 1e-2) # lower limit to the abs tolerance is needed
#
## to minimize computation time, we should keep z_sino / z_lm in memory (using the ctx object) after the fwd pass
## however, if memory is limited, we could also recompute it in the backward pass
#
#
## %%
## calculate the Hessian(x) applied to another random image w
## if the forwad pass computes the gradient of the negative Poisson log-likelihood
## the backward pass needs to compute the Hessian(x) applied to an image w
#
## grad_output next to ctx is usually the input passed to the backward pass
# grad_output = torch.rand(
#    (batch_size,) + lm_pet_lin_op.in_shape,
#    device=dev,
#    dtype=torch.float32,
#    requires_grad=False,
# )
#
#
# hess_app_grad_output =  pet_lin_op.adjoint(y * pet_lin_op(grad_output[0,...].squeeze()) / (z_sino**2))
# hess_app_grad_output_lm = lm_pet_lin_op.adjoint(lm_pet_lin_op(grad_output[0,...].squeeze()) / z_lm**2)
#
## again both ways of computing the Hessian to grad_output should be numerically very close
## the 2nd way should be faster and more memory efficient for real world low count PET data
#
# assert torch.allclose(hess_app_grad_output, hess_app_grad_output_lm, atol = 1e-2) # lower limit to the abs tolerance is needed
#
## the only thing that is now left is to properly wrap everything in a pytorch autograd layer
## as done in ../examples/07_torch/01_run_projection_layer.py
#
## %%
## calculate the gradient of the negative Poisson log-likelihood in listmode using a dedicated pytorch autograd layer
#
# from utils import LMPoissonLogLDescent
# lm_grad_layer = LMPoissonLogLDescent.apply
#
# lm_grad2 = lm_grad_layer(x, lm_pet_lin_op, contamination_list.unsqueeze(0), adjoint_ones.unsqueeze(0))
#
# assert torch.allclose(lm_grad, lm_grad2, atol = 1e-3) # lower limit to the abs tolerance is needed
#
# loss = 0.5*(lm_grad2*lm_grad2).sum()
# loss.backward()
#
