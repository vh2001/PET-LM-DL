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
                x[i, 0, ...] = (
                    lm_fwd_operators[i].adjoint(
                        lm_fwd_operators[i](grad_output[i, 0, ...].detach())
                        / z_lists[i] ** 2
                    )
                    * diag_precond_list[i]
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


# class SmoothLeakyClippedReLU(torch.nn.Module):
#    def __init__(self, alpha=-0.1):
#        super().__init__()
#        self.alpha = alpha
#        self.a = (1 - alpha) / 2
#        self.b = alpha
#        self.d = (alpha - 1) / 2
#
#    def forward(self, x):
#        a, b, d = self.a, self.b, self.d
#        return torch.where(
#            x <= 0, self.alpha * x, torch.where(x < 1, a * x**2 + b * x, x + d)
#        )
#
#
# class ParamSmoothLeakyReLU(torch.nn.Module):
#    def __init__(self, alpha: float, beta: float):
#        """
#        f(x) = alpha*x + (1-alpha) * Softplus(beta*x) / beta
#
#        - alpha controls the negative slope, reasoabke value 0.02
#        - beta > 0 controls how quickly f(x) -> x for x > 0.
#            small beta: slow, gentle transition
#            large beta: sharp, nearly ReLU-like transition at x≈0
#            reasonable value 4.0
#
#        Requires: 0 <= alpha < 1, beta > 0.
#        """
#        super().__init__()
#        if not (0.0 <= alpha < 1.0):
#            raise ValueError(f"alpha must be in [0,1), got {alpha}")
#        if beta <= 0:
#            raise ValueError(f"beta must be positive, got {beta}")
#        self.alpha = alpha
#        self.beta = beta
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        # scaled softplus: (1/beta)*log(1 + exp(beta*x))
#        sp = F.softplus(self.beta * x) / self.beta
#        return self.alpha * x + (1.0 - self.alpha) * sp


# 2. Define a mini 3D conv net block


# 3. Define the full model
