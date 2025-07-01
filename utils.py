import array_api_compat.torch as torch
import parallelproj

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


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


def plot_batch_intermediate_images(
    x_batch, xd_batch, xo_batch, output_dir, prefix="intermediate", vmax=None
):
    """
    Plots central slices for x, xd, and xo for each sample in the batch.
    Each sample gets a 3x3 grid:
      - Row 1: x (sagittal, coronal, axial)
      - Row 2: xd (sagittal, coronal, axial)
      - Row 3: xo (sagittal, coronal, axial)
    vmax can be set, otherwise defaults to 1.2*max(xd).
    """
    x_batch = to_np(x_batch)
    xd_batch = to_np(xd_batch)
    xo_batch = to_np(xo_batch)
    num_samples = x_batch.shape[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for sample_idx in range(num_samples):
            x = x_batch[sample_idx, 0]
            xd = xd_batch[sample_idx, 0]
            xo = xo_batch[sample_idx, 0]
            nx, ny, nz = x.shape

            vmin = 0
            vmax_plot = vmax if vmax is not None else 1.2 * xd.max()

            def get_slices(img):
                return [
                    img[nx // 2, :, :],  # sagittal
                    img[:, ny // 2, :],  # coronal
                    img[:, :, nz // 2],  # axial
                ]

            slices = [
                get_slices(x),
                get_slices(xd),
                get_slices(xo),
            ]
            row_titles = ["x (input)", "xd (after df step)", "xo (output)"]
            col_titles = ["Sagittal", "Coronal", "Axial"]

            fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
            im = None
            for row in range(3):
                for col in range(3):
                    im = axes[row, col].imshow(
                        slices[row][col].T,
                        origin="lower",
                        cmap="Greys",
                        vmin=vmin,
                        vmax=vmax_plot,
                    )
                    if row == 0:
                        axes[row, col].set_title(col_titles[col])
                    if col == 0:
                        axes[row, col].set_ylabel(row_titles[row])
                    axes[row, col].axis("off")
            # Add a single colorbar to the right of the grid
            fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015, pad=0.04)
            fig.suptitle(f"Sample {sample_idx}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


class SmoothLeakyClippedReLU(torch.nn.Module):
    def __init__(self, alpha=-0.1):
        super().__init__()
        self.alpha = alpha
        self.a = (1 - alpha) / 2
        self.b = alpha
        self.d = (alpha - 1) / 2

    def forward(self, x):
        a, b, d = self.a, self.b, self.d
        return torch.where(
            x <= 0, self.alpha * x, torch.where(x < 1, a * x**2 + b * x, x + d)
        )


# 2. Define a mini 3D conv net block
class MiniConvNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=8,
        num_hidden_layers=1,
        alpha=-0.1,
    ):
        super().__init__()
        self.non_lin_func = SmoothLeakyClippedReLU(alpha=alpha)

        layers = [
            torch.nn.Conv3d(in_channels, num_features, kernel_size=3, padding="same")
        ]
        for _ in range(num_hidden_layers):
            layers.append(
                torch.nn.Conv3d(
                    num_features, num_features, kernel_size=3, padding="same"
                )
            )
            layers.append(self.non_lin_func)
        layers.append(
            torch.nn.Conv3d(num_features, out_channels, kernel_size=3, padding="same")
        )

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# 3. Define the full model
class LMNet(torch.nn.Module):
    def __init__(
        self,
        conv_nets: torch.nn.ModuleList | torch.nn.Module,
        num_blocks: int | None = None,
        use_data_fidelity: bool = True,
        renormalize: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        conv_nets : torch.nn.ModuleList | torch.nn.Module
            (list of) convolutional networks. If a single module is passed,
            the same network is used for all blocks.
        num_blocks : int | None, optional
            number of unrolled blocks. only needed for weight-shared case, by default None
        use_data_fidelity : bool, optional
            whether to do a preconditioned data fidelity gradient step
            before the neural network step in every block, by default True
        renormalize : bool, optional
            whether to renormalize the output after every neural network step,
            ensuring that the mean of the images is kept constant.
        """
        super().__init__()

        if isinstance(conv_nets, torch.nn.ModuleList):
            self.weight_sharing = False
            self.conv_net_list = conv_nets
            self.num_blocks: int = len(conv_nets)
            # dummy init for step sizes - not used in forward pass
            self.step_sizes_raw: torch.Tensor = torch.ones(self.num_blocks)

        elif isinstance(conv_nets, torch.nn.Module):
            self.weight_sharing = True
            self.conv_net_list = torch.nn.ModuleList([conv_nets])

            if num_blocks is None:
                raise ValueError(
                    "num_blocks must be specified if conv_nets is a single module"
                )

            self.num_blocks: int = num_blocks

            # if weights are shared, it is better to use trainable step sizes (by default)
            self.trainable_step_sizes = True
            self.step_sizes_raw: torch.nn.Parameter = torch.nn.Parameter(
                torch.ones(self.num_blocks)
            )
        else:
            raise ValueError("conv_nets must be a list of modules or a single module")

        self.lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply
        self.nonneg_layer = SmoothLeakyClippedReLU(alpha=0.0)
        self.use_data_fidelity = use_data_fidelity
        self.renormalize = renormalize

    def forward(
        self,
        x,
        lm_pet_lin_ops,
        contamination_lists,
        adjoint_ones,
        diag_preconds,
        intermediate_plots=False,
    ):

        # PET images can have arbitrary global scales, but we don't want to
        # normalize before calculating the log-likelihood gradient
        # instead we calculate scales of all images in the batch and apply
        # them only before using the neural network

        # as scale we use the mean of the input images
        # if we are using early stopped OSEM images, the mean is well defined
        # and stable

        sample_scales = x.mean(dim=(2, 3, 4), keepdim=True)

        for i in range(self.num_blocks):
            # (preconditioned) gradient step on the data fidelity term
            if self.use_data_fidelity:
                xd = x - self.lm_logL_grad_layer(
                    x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
                )
            else:
                xd = x

            # neueral network step
            if self.weight_sharing:
                # use the same network for all blocks
                xn = self.conv_net_list[0](xd / sample_scales)
                # we have to make sure that the output of the network non negative
                # we use a smoothed ReLU with seems to work much better than a simple ReLU (for the optimization)
                # the softplus around the step size is needed to make sure that the step size is positive
                xo = self.nonneg_layer(
                    xd
                    - torch.nn.functional.softplus(self.step_sizes_raw[i])
                    * xn
                    * sample_scales
                )
            else:
                xn = self.conv_net_list[i](xd / sample_scales)
                # we have to make sure that the output of the network non negative
                # we use a smoothed ReLU with seems to work much better than a simple ReLU (for the optimization)
                xo = self.nonneg_layer(xd - xn * sample_scales)

            if self.renormalize:
                output_scales = xo.mean(dim=(2, 3, 4), keepdim=True)
                # renormalize the output to have the same mean as the input
                xo = xo * (sample_scales / output_scales)

            if intermediate_plots:
                plot_batch_intermediate_images(
                    x, xd, xo, Path("."), prefix=f"block_{i}"
                )

            x = xo

        return x
