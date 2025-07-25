import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils import LMNegPoissonLogLGradientLayer, to_np, plot_batch_intermediate_images


class ScaledSoftplus(nn.Module):
    """
    Differentiable alternative to ReLU for ensuring non-negative outputs.
    
    f(x) = beta * softplus(x/beta)
    
    As beta -> 0: approaches ReLU (sharp)
    As beta -> inf: approaches linear function
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return self.beta * F.softplus(x / self.beta)


class MiniConvNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=8,
        num_hidden_layers=3,
        renorm=True,
        beta=0.0,
    ):
        super().__init__()
        if beta == 0:
            self.non_lin_func = nn.ReLU(inplace=True)
        else:
            self.non_lin_func = nn.Softplus(beta=beta)

        self.renorm = renorm

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
        if self.renorm:
            # renormalize the input by the mean scale
            # important since PET images can have an "arbitrary" global scale
            sample_scales = x.mean(dim=(2, 3, 4), keepdim=True)
            return self.non_lin_func(x - sample_scales * self.conv(x / sample_scales))
        else:
            return self.non_lin_func(x - self.conv(x))


class DoubleConv(nn.Module):
    """(Conv3d → BatchNorm3d → ReLU) × 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpSampleConv(nn.Module):
    """Upsample by 2× with interpolate, then DoubleConv on concatenated features."""

    def __init__(self, in_ch, out_ch, mode="trilinear", align_corners=False):
        super().__init__()
        # in_ch = channels from lower-res + skip channels
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, skip):
        # 1) Upsample
        x = F.interpolate(
            x, scale_factor=2, mode=self.mode, align_corners=self.align_corners
        )
        # 2) If needed, pad to match skip size (odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(
                x,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )
        # 3) Concatenate and convolve
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class UNet3D(nn.Module):
    """3D U-Net architecture for image to image mappings"""

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64], renorm=True, use_scaled_softplus=False, softplus_beta=1.0):
        super().__init__()
        self.renorm = renorm
        self.use_scaled_softplus = use_scaled_softplus
        self.softplus_beta = softplus_beta

        if use_scaled_softplus:
            self.activation = ScaledSoftplus(beta=softplus_beta)
        else:
            self.activation = F.ReLU()


        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (using interpolate + conv)
        self.ups = nn.ModuleList()
        for feat in reversed(features):
            # in channels = feat*2 (from bottleneck or prev up) + feat (skip)
            self.ups.append(UpSampleConv(feat * 2 + feat, feat))

        # Final 1×1×1 conv
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        if beta == 0:
            self.non_lin_layer = nn.ReLU(inplace=True)
        else:
            self.non_lin_layer = nn.Softplus(beta=beta)

    def forward(self, x):
        # PET images can have arbitrary global scales, but we don't want to
        # normalize before calculating the log-likelihood gradient
        # instead we calculate scales of all images in the batch and apply
        # them only before using the neural network

        # as scale we use the mean of the input images
        # if we are using early stopped OSEM images, the mean is well defined

        input_x = x  # Save the original input

        if self.renorm:
            # renormalize the input by the mean scale
            # important since PET images can have an "arbitrary" global scale
            sample_scales = x.mean(dim=(2, 3, 4), keepdim=True)
            x = x / sample_scales

        skips = []
        # Encoder path
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        unet_out = self.final_conv(x)

        # Renormalize the output by multiplying with the sample scales
        if self.renorm:
            unet_out = unet_out * sample_scales

        return self.activation(input_x - unet_out)


class LMNet(torch.nn.Module):
    def __init__(
        self,
        conv_nets: torch.nn.ModuleList,
        num_blocks: int,
        use_data_fidelity: bool = True,
        weight_sharing: bool = False,
    ):
        """Unrolled neural network for PET image reconstruction using listmode (LM) data.

        Parameters
        ----------
        conv_nets : torch.nn.ModuleList | torch.nn.Module
            (list of) convolutional networks. If a single module is passed,
            the same network is used for all blocks.
            The output of these models must be non-negative.
        num_blocks : int | None, optional
            number of unrolled blocks. each block consists of a data fidelity gradient step
            and a neural network step.
        use_data_fidelity : bool, optional
            whether to do a preconditioned data fidelity gradient step
            before the neural network step in every block, by default True
            (useful to test how much data fidelity helps)
        """
        super().__init__()

        self.weight_sharing = weight_sharing
        self.num_blocks = num_blocks
        self.conv_net_list = conv_nets

        if self.weight_sharing and len(conv_nets) != 1:
            raise ValueError(
                "If weight sharing is used, conv_nets must be a ModuleList with a single element."
            )

        if not self.weight_sharing and (len(conv_nets) != self.num_blocks):
            raise ValueError(
                "If weight sharing is not used, conv_nets must be a list of length num_blocks."
            )

        self.lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply
        self.use_data_fidelity = use_data_fidelity

    def forward(
        self,
        x,
        lm_pet_lin_ops,
        contamination_lists,
        adjoint_ones,
        diag_preconds,
        intermediate_plots=False,
    ):

        # list for intermediate images for debugging plots
        x_intermed = []

        for i in range(self.num_blocks):
            if self.use_data_fidelity:
                # (preconditioned) gradient step on the data fidelity term
                # xd = x - diag_preconditioner * \nabla_x PET_data_fidelity(x)
                xd = x - self.lm_logL_grad_layer(
                    x, lm_pet_lin_ops, contamination_lists, adjoint_ones, diag_preconds
                )
            else:
                xd = x

            # neueral network step
            if self.weight_sharing:
                # use the same network for all blocks
                xo = self.conv_net_list[0](xd)
            else:
                xo = self.conv_net_list[i](xd)

            if intermediate_plots:
                x_intermed.append(to_np(xd))
                x_intermed.append(to_np(xo))

            x = xo

        if intermediate_plots:
            plot_batch_intermediate_images(np.array(x_intermed).squeeze())

        return x


def detailed_param_count(model):
    counts = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            counts[name] = p.numel()
    total = sum(counts.values())
    for name, cnt in counts.items():
        print(f"{name:>40s}: {cnt:,}")
    print(f"{'-'*60}\nTotal trainable parameters: {total:,}")


### DONT FORGET TO ADD YOUR CUSTOM DENOISING MODEL TO THE REGISTRY ###
DENOISER_MODEL_REGISTRY = {"MiniConvNet": MiniConvNet, "UNet3D": UNet3D}

# Example usage
if __name__ == "__main__":
    xb = torch.randn(1, 1, 57, 71, 93)
    model = UNet3D(in_channels=1, out_channels=1, features=[16, 32])
    out = model(xb)
    print("Output shape:", out.shape)

    detailed_param_count(model)

    torch.onnx.export(
        model,
        xb,
        "unet3d.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )
