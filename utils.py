import array_api_compat.torch as torch
import parallelproj

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class LMNegPoissonLogLGradientLayer(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lm_fwd_operators: list[parallelproj.LinearOperator],
        contam_lists: list[torch.Tensor],
        adjoint_ones: list[torch.Tensor],
        diag_precond_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        diag_precond_list : list of torch.Tensors
            list of diagonal preconditioners with dimension (num_voxels_x, num_voxels_y, num_voxels_z)

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, lm_fwd_operator.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)

        batch_size = x.shape[0]

        g = torch.zeros_like(x)
        z_lists = []

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            z_lists.append(lm_fwd_operators[i](x[i, 0, ...].detach()) + contam_lists[i])
            g[i, 0, ...] = (
                adjoint_ones[i] - lm_fwd_operators[i].adjoint(1 / z_lists[i])
            ) * diag_precond_list[i]

        ctx.lm_fwd_operators = lm_fwd_operators
        ctx.z_lists = z_lists
        ctx.diag_precond_list = diag_precond_list

        return g

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
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
            return None, None, None, None, None
        else:
            lm_fwd_operators = ctx.lm_fwd_operators
            z_lists = ctx.z_lists
            diag_precond_list = ctx.diag_precond_list

            batch_size = grad_output.shape[0]

            x = torch.zeros_like(grad_output)

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = lm_fwd_operators[i].adjoint(
                    lm_fwd_operators[i](
                        grad_output[i, 0, ...].detach() * diag_precond_list[i]
                    )
                    / z_lists[i] ** 2
                )

            return x, None, None, None, None


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def plot_batch_input_output_target(
    input_batch, output_batch, target_batch, output_dir, prefix="val_samples"
):
    input_batch = to_np(input_batch)
    output_batch = to_np(output_batch)
    target_batch = to_np(target_batch)
    num_samples = input_batch.shape[0]

    pdf_path = output_dir / f"{prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for sample_idx in range(num_samples):
            inp = input_batch[sample_idx, 0]
            out = output_batch[sample_idx, 0]
            tgt = target_batch[sample_idx, 0]
            nx, ny, nz = inp.shape

            vmin, vmax = tgt.min(), 1.2 * tgt.max()

            def get_slices(img):
                return [
                    img[nx // 2, :, :],  # sagittal
                    img[:, ny // 2, :],  # coronal
                    img[:, :, nz // 2],  # axial
                ]

            diff = out - tgt
            diff_vmax = 0.3 * tgt.max()
            diff_vmin = -0.3 * tgt.max()

            slices = [
                get_slices(inp),
                get_slices(out),
                get_slices(tgt),
                get_slices(diff),
            ]
            row_titles = ["Input", "Output", "Target", "Output - Target"]
            col_titles = ["Sagittal", "Coronal", "Axial"]

            fig, axes = plt.subplots(4, 3, figsize=(12, 13), layout="constrained")
            im_greys = None
            im_diff = None
            for row in range(4):
                for col in range(3):
                    if row < 3:
                        im_greys = axes[row, col].imshow(
                            slices[row][col].T,
                            origin="lower",
                            cmap="Greys",
                            vmin=vmin,
                            vmax=vmax,
                        )
                    else:
                        im_diff = axes[row, col].imshow(
                            slices[row][col].T,
                            origin="lower",
                            cmap="seismic",
                            vmin=diff_vmin,
                            vmax=diff_vmax,
                        )
                    if row == 0:
                        axes[row, col].set_title(col_titles[col])
                    if col == 0:
                        axes[row, col].set_ylabel(row_titles[row])
                    axes[row, col].axis("off")
            # Add a colorbar for the top 3 rows (Greys)
            fig.colorbar(
                im_greys,
                ax=axes[:3, :],
                orientation="vertical",
                fraction=0.025,
                pad=0.04,
            )
            # Add a colorbar for the last row (difference)
            fig.colorbar(
                im_diff, ax=axes[3, :], orientation="vertical", fraction=0.025, pad=0.04
            )
            fig.suptitle(f"Sample {sample_idx}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def plot_batch_intermediate_images(x_intermed):

    num_samples = x_intermed.shape[1]
    num_intermed_imgs = x_intermed.shape[0]
    nx = x_intermed.shape[2]
    ny = x_intermed.shape[3]
    nz = x_intermed.shape[4]

    for sample_idx in range(num_samples):
        fig, ax = plt.subplots(
            3,
            num_intermed_imgs,
            figsize=(2 * num_intermed_imgs, 2 * 3),
            layout="constrained",
        )

        for i in range(num_intermed_imgs):
            vol = x_intermed[i, sample_idx, ...]
            vmin = vol.min()
            vmax = vol.max()
            im0 = ax[0, i].imshow(
                vol[nx // 2, :, :].T, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax
            )
            im1 = ax[1, i].imshow(
                vol[ny // 2, :, :].T, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax
            )
            im2 = ax[2, i].imshow(
                vol[:, :, nz // 2].T, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax
            )
            fig.colorbar(
                im2,
                ax=ax[2, i],
                orientation="horizontal",
                location="bottom",
                fraction=0.04,
                pad=0.01,
            )

            if i % 2 == 0:
                ax[0, i].set_title(f"x_d {i // 2 + 1}", fontsize="medium")
            else:
                ax[0, i].set_title(f"x_o {i // 2 + 1}", fontsize="medium")

        for axx in ax.ravel():
            axx.set_xticks([])
            axx.set_yticks([])

        fig.savefig(f"intermediate_images_sample_{sample_idx}.png")


if __name__ == "__main__":
    ####################################################################
    ### GRADCHECK THE IMPLEMENTATION OF THE LMPOISSONGRADLAYER #########
    ####################################################################
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    tof = True

    img_shape = (5, 5, 5)
    voxel_size = (1, 1, 1)
    img_origin = None

    # %%
    # setup a few pseudo random events

    event_start = []
    event_end = []

    for i in range(img_shape[1]):
        for j in range(img_shape[2]):
            event_start.append([-20, i - img_shape[1] // 2, j - img_shape[2] // 2])
            event_end.append([20, i - img_shape[1] // 2, j - img_shape[2] // 2])

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            event_start.append(
                [i - img_shape[0] // 2 + 0.5, j - img_shape[1] // 2 + 0.5, -20]
            )
            event_end.append(
                [i - img_shape[0] // 2 - 0.5, j - img_shape[1] // 2 - 0.5, 20]
            )

    event_start.append([-20, -20, 0])
    event_start.append([-20, -20, -20])

    event_end.append([20, 20, 0])
    event_end.append([20, 20, 20])

    event_start = torch.tensor(event_start, device=dev, dtype=torch.float32)
    event_end = torch.tensor(event_end, device=dev, dtype=torch.float32)

    num_events = event_start.shape[0]

    lm_proj = parallelproj.ListmodePETProjector(
        event_start,
        event_end,
        img_shape=img_shape,
        voxel_size=voxel_size,
        img_origin=img_origin,
    )

    if tof:
        lm_proj.tof_parameters = parallelproj.TOFParameters(
            num_tofbins=5, tofbin_width=1.2, sigma_tof=1.2
        )
        lm_proj.event_tofbins = (
            torch.randint(lm_proj.tof_parameters.num_tofbins, (num_events,))
            - lm_proj.tof_parameters.num_tofbins // 2
        )
        lm_proj.tof = True

    # %%

    # create a fake sens. image (lm back proj instead of full backproj)
    adjoint_ones = [
        lm_proj.adjoint(torch.ones(num_events, device=dev, dtype=torch.float32)),
        lm_proj.adjoint(torch.ones(num_events, device=dev, dtype=torch.float32)),
    ]

    contam_lists = [
        0.05 * torch.rand(num_events, device=dev, dtype=torch.float32),
        0.1 * torch.rand(num_events, device=dev, dtype=torch.float32),
    ]

    lm_projs = [lm_proj, lm_proj]

    diag_precond_list = [
        torch.rand(*img_shape, device=dev, dtype=torch.float32),
        torch.rand(*img_shape, device=dev, dtype=torch.float32),
    ]

    x = torch.rand(2, 1, *img_shape, device=dev, dtype=torch.float32)

    # %%
    lm_grad_layer = LMNegPoissonLogLGradientLayer.apply

    f1 = lm_grad_layer(x, lm_projs, contam_lists, adjoint_ones, diag_precond_list)

    if True:
        print("starting gradcheck")
        x.requires_grad = True
        grad_test = torch.autograd.gradcheck(
            lm_grad_layer,
            (x, lm_projs, contam_lists, adjoint_ones, diag_precond_list),
            eps=1e-3,
            atol=1e-3,
            rtol=1e-3,
            nondet_tol=1e-5,  # we don't expect to get the exact same result every time due the parallel sums and float issues
        )
