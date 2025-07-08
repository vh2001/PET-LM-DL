import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils import LMNegPoissonLogLGradientLayer, to_np, plot_batch_intermediate_images


class MiniConvNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=8,
        num_hidden_layers=3,
    ):
        super().__init__()
        self.non_lin_func = nn.ReLU(inplace=True)

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
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32], renorm=True):
        super().__init__()
        self.renorm = renorm
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

    def forward(self, x):
        # PET images can have arbitrary global scales, but we don't want to
        # normalize before calculating the log-likelihood gradient
        # instead we calculate scales of all images in the batch and apply
        # them only before using the neural network

        # as scale we use the mean of the input images
        # if we are using early stopped OSEM images, the mean is well defined

        input_x = x  # Save the original input

        if self.renorm:
            sample_scales = x.mean(dim=(2, 3, 4), keepdim=True)
            x = x / sample_scales  # Normalize by the mean scale
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

        return F.relu(input_x - unet_out)


class LMNet(torch.nn.Module):
    def __init__(
        self,
        conv_nets: torch.nn.ModuleList | torch.nn.Module,
        num_blocks: int | None = None,
        use_data_fidelity: bool = True,
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
        """
        super().__init__()

        if isinstance(conv_nets, torch.nn.ModuleList):
            self.weight_sharing = False
            self.conv_net_list = conv_nets
            self.num_blocks: int = len(conv_nets)

        elif isinstance(conv_nets, torch.nn.Module):
            self.weight_sharing = True
            self.conv_net_list = torch.nn.ModuleList([conv_nets])

            if num_blocks is None:
                raise ValueError(
                    "num_blocks must be specified if conv_nets is a single module"
                )

            self.num_blocks: int = num_blocks
        else:
            raise ValueError("conv_nets must be a list of modules or a single module")

        self.lm_logL_grad_layer = LMNegPoissonLogLGradientLayer.apply
        self.nonneg_layer = nn.ReLU(inplace=True)
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

        x_intermed = []

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
