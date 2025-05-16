import array_api_compat.torch as torch
from array_api_compat import device
import parallelproj

import json
from pathlib import Path


def load_lm_pet_data(odir: Path):
    """
    Loads PET listmode data and returns the linear operator, contamination list, and adjoint ones.
    """
    data_tensors = torch.load(odir / "data_tensors.pt")

    event_start_coords = data_tensors["event_start_coords"]
    event_end_coords = data_tensors["event_end_coords"]
    event_tofbins = data_tensors["event_tofbins"]
    att_list = data_tensors["att_list"]
    contamination_list = data_tensors["contamination_list"]
    adjoint_ones = data_tensors["adjoint_ones"]
    x_true = data_tensors["x_true"]

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

    # enable TOF in the LM projector
    lm_proj.tof_parameters = tof_parameters
    lm_proj.event_tofbins = event_tofbins
    lm_proj.tof = True

    lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

    res_model = parallelproj.GaussianFilterOperator(
        in_shape, sigma=fwhm_data_mm / (2.35 * torch.tensor(voxel_size))
    )

    lm_pet_lin_op = parallelproj.CompositeLinearOperator(
        (lm_att_op, lm_proj, res_model)
    )

    return lm_pet_lin_op, contamination_list, adjoint_ones, x_true


class LMPoissonLogLDescent(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        lm_fwd_operators: list[parallelproj.LinearOperator],
        contam_lists: list[torch.Tensor],
        adjoint_ones: list[torch.Tensor],
    ) -> [torch.Tensor, torch.Tensor]:
        """forward pass of listmode negative Poisson logL gradient descent layer

        Parameters
        ----------
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        lm_fwd_operators : list of parallelproj.LinearOperator in listmode
            each linear operator that can act on a single 3D image
        contam_lists : list of torch.Tensors
            list of listmode contamination (scatter and randoms) with dimension (num_events)
        adjoint_ones : torch.Tensor
            list of 3D images of adjoint of the (full not listmode) linear operator applied to ones
            (num_voxels_x, num_voxels_y, num_voxels_z)

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, lm_fwd_operator.out_shape)
        """

        batch_size = x.shape[0]

        g = torch.zeros_like(x)
        z_lm = torch.zeros_like(x)

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            z_lm[i, ...] = lm_fwd_operators[i](x[i, 0, ...].detach()) + contam_lists[i]
            g[i, ...] = adjoint_ones[i] - lm_fwd_operators[i].adjoint(1 / z_lm[i, ...])

        return g, z_lm

    @staticmethod
    def setup_context(ctx, inputs, output):
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        _, lm_fwd_operators, _, _ = inputs
        g, z_lm = output
        ctx.set_materialize_grads(False)
        ctx.lm_fwd_operators = lm_fwd_operators
        ctx.save_for_backward(z_lm)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size = 1, 1, lm_fwd_operator.in_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes four input arguments (x, lm_fwd_operator, contam_list, adjoint_ones)
        # we have to return four arguments (the last three are None)
        if grad_output is None:
            return None, None, None, None
        else:
            lm_fwd_operators = ctx.lm_fwd_operators
            (z_lm,) = ctx.saved_tensors

            batch_size = grad_output.shape[0]

            x = torch.zeros(
                (batch_size, 1) + lm_fwd_operators[0].in_shape,
                dtype=grad_output.dtype,
                device=device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = lm_fwd_operators[i].adjoint(
                    lm_fwd_operators[i](grad_output[i, ...].detach())
                    / z_lm[i, ...] ** 2
                )

            return x, None, None, None
