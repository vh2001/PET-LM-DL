import array_api_compat.torch as torch
from array_api_compat import device
import parallelproj


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
