"""
Neg Poisson logL gradient layer (in listmode)
=============================================
"""

# %%
from __future__ import annotations

import array_api_compat.torch as torch  # we use the array_api_compat version of torch which is need for compatibility with parallelproj
import torch.nn.functional as F

import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import parallelproj

import json
from pathlib import Path
from dataclasses import asdict

# using torch valid choices are 'cpu' or 'cuda'
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"


# input parameters
parser = argparse.ArgumentParser(description="Simulate PET data parameters")
parser.add_argument(
    "--seed", type=int, default=1, help="Seed for random noise generator"
)
parser.add_argument(
    "--fwhm_data_mm",
    type=float,
    default=4.5,
    help="Simulated resolution of PET scanner (FWHM in mm)",
)
parser.add_argument("--subject_num", type=int, default=4, help="Brainweb subject id")
parser.add_argument(
    "--contrast_num",
    type=int,
    default=0,
    help="Simulated contrast number (0, 1, or 2)",
    choices=[0, 1, 2],
)
parser.add_argument(
    "--countlevel",
    type=float,
    default=1.0,
    help="Relative count level (higher means more counts and less noise)",
)

args = parser.parse_args()

seed: int = args.seed
fwhm_data_mm: float = args.fwhm_data_mm
subject_num: int = args.subject_num
contrast_num: int = args.contrast_num
countlevel: float = args.countlevel

# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Sinogram forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_rings = 32
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    torch,
    dev,
    radius=300.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=0.5 * 5.0 * num_rings * torch.linspace(-1, 1, num_rings),
    symmetry_axis=2,
)

# load the activity and attenuation images

act_nii = nib.as_closest_canonical(
    nib.load(
        Path("data")
        / "brainweb_petmr_v2"
        / f"subject{subject_num:02}"
        / f"image_{contrast_num}.nii.gz"
    )
)

# Find the last slice along the z-axis where the sum is > 0
act_data = act_nii.get_fdata()
z_indices = [z for z in range(act_data.shape[2]) if act_data[:, :, z].sum() > 0]
last_nonzero_z = z_indices[-1] if z_indices else act_data.shape[2] - 1

end_sl = max(last_nonzero_z + 1, act_data.shape[2])
start_sl = end_sl - 158

x_true = torch.as_tensor(act_data, dtype=torch.float32, device=dev)

att_nii = nib.as_closest_canonical(
    nib.load(
        Path("data")
        / "brainweb_petmr_v2"
        / f"subject{subject_num:02}"
        / f"attenuation_image.nii.gz"
    )
)

x_att = torch.as_tensor(att_nii.get_fdata(), dtype=torch.float32, device=dev)

# %%
# skip first and last 15 slices to reduce the problem size

x_true = x_true[:, :, start_sl:end_sl]
x_att = x_att[:, :, start_sl:end_sl]

# %%
# reduce the image size by a factor of 2 to speed up the simulation

x_true = (
    F.avg_pool3d(x_true.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)
    .squeeze(0)
    .squeeze(0)
)
x_att = (
    F.avg_pool3d(x_att.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)
    .squeeze(0)
    .squeeze(0)
)

img_shape = tuple(x_true.shape)
voxel_size = tuple((2 * act_nii.header["pixdim"][1:4]).tolist())

# setup the sinogram projector

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=10,  # we use "smaller" max rings difference to speed up the simulation
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# calculate the attenuation sinogram
att_sino = torch.exp(-proj(x_att))

# we artficially lowe the attenuation (sensitivity) sinogram to get by default a lower numbrt of counts
att_sino *= countlevel * 0.005

# %%
# Complete PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=17, tofbin_width=12.0, sigma_tof=12.0
)

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino
)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=fwhm_data_mm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

# %%
# calculate the adjoint of the forward model applied to a ones sinogram
# needed to calculate the gradient of the negative Poisson log-likelihood (in listmode)

adjoint_ones = pet_lin_op.adjoint(
    torch.ones(pet_lin_op.out_shape, device=dev, dtype=torch.float32)
)

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = torch.full(
    noise_free_data.shape,
    0.5 * float(torch.mean(noise_free_data)),
    device=dev,
    dtype=torch.float32,
)

noise_free_data += contamination

# add Poisson noise
y = torch.poisson(noise_free_data).to(torch.uint16)

# %%
# Conversion of the emission sinogram to listmode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we can convert an
# integer non-TOF or TOF sinogram to an event list for listmode processing.
#
# **Note:** The create event list is sorted and should be shuffled running LM-MLEM.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y
)

# shuffle the event list using torch
perm = torch.randperm(event_start_coords.shape[0])
event_start_coords = event_start_coords[perm]
event_end_coords = event_end_coords[perm]
event_tofbins = event_tofbins[perm]

# %%
# Setup of the LM projector and LM forward model
# ----------------------------------------------

lm_proj = parallelproj.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    proj.in_shape,
    proj.voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events
# this needs to be a non-TOF projection
att_list = torch.exp(-lm_proj(x_att))
lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

# enable TOF in the LM projector
lm_proj.tof_parameters = proj.tof_parameters
lm_proj.event_tofbins = event_tofbins
lm_proj.tof = proj.tof

# create the contamination list
contamination_list = torch.full(
    event_start_coords.shape[0],
    float(contamination.ravel()[0]),
    device=dev,
    dtype=torch.float32,
)

lm_pet_lin_op = parallelproj.CompositeLinearOperator((lm_att_op, lm_proj, res_model))


# %%
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

x_mlem = torch.ones_like(x_true)
num_iter = 100

for i in range(num_iter):
    print(f"LM iteration {(i+1):03}/{num_iter}", end="\r")
    if i == 20:
        x_mlem_20 = x_mlem.clone()

    neg_pl_grad = neg_poisson_logL_grad(x_mlem)
    step = x_mlem / adjoint_ones

    x_mlem -= step * neg_pl_grad

print()

# setup a Gaussian post-filter operator
post_filter = parallelproj.GaussianFilterOperator(
    x_mlem.shape, sigma=4.5 / (2.35 * lm_proj.voxel_size)
)

x_mlem_20_filtered = post_filter(x_mlem_20)
x_mlem_filtered = post_filter(x_mlem)

# %%
# save all data to disk such that we can re-use it later (e.g. using a torch data loader)

odir = Path(
    f"data/sim_pet_data/subject{subject_num:02}_contrast_{contrast_num}_countlevel_{countlevel}_seed_{seed}"
)
odir.mkdir(exist_ok=True, parents=True)

torch.save(
    {
        "event_start_coords": event_start_coords,
        "event_end_coords": event_end_coords,
        "event_tofbins": event_tofbins,
        "att_list": att_list,
        "contamination_list": contamination_list,
        "adjoint_ones": adjoint_ones,
        "x_true": x_true,
        "x_att": x_att,
        "x_mlem": x_mlem,
        "x_mlem_20": x_mlem_20,
        "x_mlem_filtered": x_mlem_filtered,
        "x_mlem_20_filtered": x_mlem_20_filtered,
    },
    odir / "input_tensors.pt",
)

with open(odir / "projector_parameters.json", "w", encoding="UTF8") as f:
    json.dump(
        {
            "tof_parameters": asdict(lm_proj.tof_parameters),
            "in_shape": proj.in_shape,
            "voxel_size": voxel_size,
            "img_origin": tuple(proj.img_origin.tolist()),
            "fwhm_data_mm": fwhm_data_mm,
            "seed": seed,
        },
        f,
        indent=4,
    )

# %%
# show a few images

sl0 = img_shape[0] // 2
sl1 = img_shape[1] // 2
sl2 = img_shape[2] // 3


kws = dict(cmap="Greys", vmin=0, vmax=1.5 * float(torch.max(x_true)), origin="lower")

fig, ax = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
ax[0, 0].imshow(x_true[:, :, sl2].cpu().numpy().T, **kws)
ax[0, 1].imshow(x_true[sl0, :, :].cpu().numpy().T, **kws)
ax[0, 2].imshow(x_true[:, sl1, :].cpu().numpy().T, **kws)
ax[1, 0].imshow(x_mlem_20_filtered[:, :, sl2].cpu().numpy().T, **kws)
ax[1, 1].imshow(x_mlem_20_filtered[sl0, :, :].cpu().numpy().T, **kws)
ax[1, 2].imshow(x_mlem_20_filtered[:, sl1, :].cpu().numpy().T, **kws)
ax[2, 0].imshow(x_mlem_filtered[:, :, sl2].cpu().numpy().T, **kws)
ax[2, 1].imshow(x_mlem_filtered[sl0, :, :].cpu().numpy().T, **kws)
ax[2, 2].imshow(x_mlem_filtered[:, sl1, :].cpu().numpy().T, **kws)

for axx in ax.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

ax[0, 1].set_title("ground truth", fontsize="medium")
ax[1, 1].set_title("MLEM 20 it. + post-filter", fontsize="medium")
ax[2, 1].set_title("MLEM 100 it. + post-filter", fontsize="medium")

fig.show()

fig.savefig(odir / "mlem_recons.png", dpi=300)

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
