import torch
import torch.nn.functional as F # noqa: N812
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from typing import Self 

class MishDecomposed(nn.Module):
    """Mish activation decomposed into standard ONNX ops for TensorRT compatibility."""
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))

class CSELayer(nn.Module):
    def __init__(self, num_channels: int = 48, reduction_ratio: int = 2) -> None: 
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.squeezing = nn.Sequential(
            nn.Conv2d(num_channels, num_channels_reduced, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_channels_reduced, num_channels, 1, 1),
            nn.Hardsigmoid(True),
        )

    def forward(self, input_tensor: Tensor) -> Tensor: 
        squeeze_tensor = torch.mean(input_tensor, dim=[2, 3], keepdim=True)
        output_tensor = input_tensor * self.squeezing(squeeze_tensor)
        return output_tensor

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 2, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0, 
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )
        
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1, 
            stride=s,
            bias=bias,
        )

    def update_params(self) -> None:
        if self.update_params_flag: return 

        assert isinstance(self.conv[0].weight, Tensor)
        assert isinstance(self.conv[0].bias, Tensor)
        assert isinstance(self.conv[1].weight, Tensor)
        assert isinstance(self.conv[1].bias, Tensor)
        assert isinstance(self.conv[2].weight, Tensor)
        assert isinstance(self.conv[2].bias, Tensor)

        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach() if self.sk.bias is not None else torch.zeros_like(self.bias_concat)

        target_kernel_size = 3
        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.data = self.bias_concat
        self.update_params_flag = True 

    def forward(self, x: Tensor) -> Tensor: 
        if self.training:

            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0) 
            out = self.conv(x_pad) + self.sk(x)
            return out
        else:
            if not self.update_params_flag:
                self.update_params()
            return self.eval_conv(x)

class SeqConv3x3(nn.Module):
    """Sequential 1x1 -> 3x3(p=0) convolutions, designed for reparameterization."""
    def __init__(self, inp_planes: int, out_planes: int, depth_multiplier: int) -> None:
        super().__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mid_planes = int(out_planes * depth_multiplier)
        self.conv0 = torch.nn.Conv2d(
            self.inp_planes, self.mid_planes, kernel_size=1, padding=0, bias=True
        )
        self.conv1 = torch.nn.Conv2d(
             self.mid_planes, self.out_planes, kernel_size=3, padding=0, bias=True # Ensure padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        y0 = self.conv0(x)
        y0_padded = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        return self.conv1(y0_padded)

    def rep_params(self) -> tuple[Tensor, Tensor | None]: 
        """Calculates the parameters of the equivalent fused 3x3 convolution."""
        k0 = self.conv0.weight.data 
        b0 = self.conv0.bias.data if self.conv0.bias is not None else None
        k1 = self.conv1.weight.data 
        b1 = self.conv1.bias.data if self.conv1.bias is not None else None
        k0_mat = k0.squeeze(dim=(2, 3)) 
        RK = torch.tensordot(k1, k0_mat, dims=([1], [0])) 
        RK = RK.permute(0, 3, 1, 2).contiguous() 
        RB = None
        if b0 is not None and b1 is not None:
            k1_summed = k1.sum(dim=[2, 3]) 
            b0_effect = torch.matmul(k1_summed, b0)
            RB = b0_effect + b1
        elif b1 is not None:
             RB = b1
        return RK, RB

class RepConv(nn.Module):
    # --- Start Included Changes ---
    def __init__(self, in_dim: int = 3, out_dim: int = 32) -> None:
        super().__init__()
        # Initialize branches (ensure bias=True where expected by fusion logic)
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
        self.conv3 = Conv3XC(in_dim, out_dim, bias=True)
        # Fused convolution layer (ensure bias=True)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True) # Initialize alphas to 1
        self._is_fused = False # Flag to track fusion state

    def fuse(self) -> None:
        """Fuses the parameters of conv1, conv2, and conv3 into conv_3x3_rep."""
        if self._is_fused: return # Skip if already fused

        # Get weights and biases from branches
        conv1_w, conv1_b = self.conv1.rep_params()
        conv2_w = self.conv2.weight.data
        conv2_b = self.conv2.bias.data if self.conv2.bias is not None else None
        self.conv3.update_params() # Ensure conv3 is fused first
        conv3_w = self.conv3.eval_conv.weight.data
        conv3_b = self.conv3.eval_conv.bias.data if self.conv3.eval_conv.bias is not None else None

        zero_bias = torch.zeros(self.conv_3x3_rep.out_channels, device=self.conv_3x3_rep.weight.device)
        if conv1_b is None: conv1_b = zero_bias
        if conv2_b is None: conv2_b = zero_bias
        if conv3_b is None: conv3_b = zero_bias

        device = self.conv_3x3_rep.weight.device
        alpha_dev = self.alpha.to(device) 

        sum_weight = (
            alpha_dev[0] * conv1_w + alpha_dev[1] * conv2_w + alpha_dev[2] * conv3_w
        ).to(device)
        sum_bias = (
            alpha_dev[0] * conv1_b + alpha_dev[1] * conv2_b + alpha_dev[2] * conv3_b
        ).to(device)

        self.conv_3x3_rep.weight.data = sum_weight
        if self.conv_3x3_rep.bias is not None:
             self.conv_3x3_rep.bias.data = sum_bias
        self._is_fused = True

    def train(self, mode: bool = True) -> Self:
        """Switch between training and evaluation modes, handling fusion."""
        super().train(mode)
        if mode:
            self._is_fused = False
        else:
            try:
                self.fuse()
            except Exception as e:
                print(f"Warning: RepConv fusion failed during train(False): {e}")
        return self

    def forward(self, x: Tensor) -> Tensor: 
        if self.training:
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3
        else:
            if not self._is_fused:
                self.fuse()
            return self.conv_3x3_rep(x)


class OmniShift(nn.Module):
    """Depthwise convolution block with multiple kernel sizes (1x1, 3x3, 5x5 + identity), designed for reparameterization."""
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=True)

        self.alpha1 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha2 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha3 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha4 = nn.Parameter(torch.ones(1, dim, 1, 1))

        self.conv5x5_reparam = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=True)
        self._is_fused = False

    def reparam_5x5(self) -> None:
        """Fuses the identity, 1x1, 3x3, and 5x5 branches into conv5x5_reparam."""
        if self._is_fused: return

        w1x1 = self.conv1x1.weight.data 
        b1x1 = self.conv1x1.bias.data if self.conv1x1.bias is not None else None
        w3x3 = self.conv3x3.weight.data 
        b3x3 = self.conv3x3.bias.data if self.conv3x3.bias is not None else None
        w5x5 = self.conv5x5.weight.data 
        b5x5 = self.conv5x5.bias.data if self.conv5x5.bias is not None else None

        identity_kernel = torch.zeros_like(w5x5)
        identity_kernel[:, :, 2, 2] = 1.0 

        padded_w1x1 = F.pad(w1x1, (2, 2, 2, 2)) 
        padded_w3x3 = F.pad(w3x3, (1, 1, 1, 1))

        alpha1_d = self.alpha1.data.view(self.dim, 1, 1, 1)
        alpha2_d = self.alpha2.data.view(self.dim, 1, 1, 1)
        alpha3_d = self.alpha3.data.view(self.dim, 1, 1, 1)
        alpha4_d = self.alpha4.data.view(self.dim, 1, 1, 1)

        combined_weight = (
            alpha1_d * identity_kernel
            + alpha2_d * padded_w1x1
            + alpha3_d * padded_w3x3
            + alpha4_d * w5x5
        )

        combined_bias = torch.zeros(self.conv5x5_reparam.out_channels, device=combined_weight.device)
        if b1x1 is not None: combined_bias += self.alpha2.data.squeeze() * b1x1 # Use original alpha shape for squeeze
        if b3x3 is not None: combined_bias += self.alpha3.data.squeeze() * b3x3
        if b5x5 is not None: combined_bias += self.alpha4.data.squeeze() * b5x5

        self.conv5x5_reparam.weight.data = combined_weight
        if self.conv5x5_reparam.bias is not None:
            self.conv5x5_reparam.bias.data = combined_bias
        self._is_fused = True 

    def train(self, mode: bool = True) -> Self: 
        """Switch between training and evaluation modes, handling fusion."""
        super().train(mode)
        if mode:
             self._is_fused = False
        else:
            self.reparam_5x5()
        return self

    def forward(self, x: Tensor) -> Tensor: 
        if self.training:
            out1x1 = self.conv1x1(x)
            out3x3 = self.conv3x3(x)
            out5x5 = self.conv5x5(x)
            out = (
                self.alpha1 * x 
                + self.alpha2 * out1x1
                + self.alpha3 * out3x3
                + self.alpha4 * out5x5
            )
            return out
        else:
            if not self._is_fused:
                self.reparam_5x5()
            return self.conv5x5_reparam(x)


class ParPixelUnshuffle(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, down: int) -> None: # Added type hints
        super().__init__()
        self.pu = nn.PixelUnshuffle(down)
        self.poll = nn.Sequential(
            nn.MaxPool2d(kernel_size=down, stride=down),
            RepConv(in_dim, out_dim) 
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pu(x) + self.poll(x)


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """
    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0, 
        dccm: bool = True, 
        se: bool = False, 
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.hidden = int(expansion_ratio * dim) 
        self.fc1 = RepConv(dim, self.hidden * 2) 

        self.act = MishDecomposed()

        conv_channels = int(conv_ratio * dim)
        assert conv_channels <= self.hidden, "conv_channels must be <= hidden channels"
        self.split_indices = [self.hidden, self.hidden - conv_channels, conv_channels]
        assert sum(self.split_indices) == self.hidden * 2, "Split indices do not sum correctly"

        self.conv = nn.Sequential(
            ParPixelUnshuffle(dim, dim * 4, 2), 
            OmniShift(dim * 4), 
            CSELayer(dim * 4) if se else nn.Identity(),
            nn.PixelShuffle(2),
        )
        self.fc2 = RepConv(self.hidden, dim) if dccm else nn.Conv2d(self.hidden, dim, 1, 1, bias=True) # Ensure bias=True if not RepConv

    def forward(self, x: Tensor) -> Tensor: 
        shortcut = x
        x = self.norm(x)
        fc1_out = self.fc1(x)


        g, i, c = torch.split(fc1_out, self.split_indices, dim=1)


        c = self.conv(c)


        if i.shape[1] + c.shape[1] != self.hidden:
            raise ValueError(
                f"Channel mismatch before fc2: i({i.shape[1]}) + c({c.shape[1]}) != hidden({self.hidden})"
            )


        combined_i_c = torch.cat((i, c), dim=1)

        gated_combined = self.act(g) * combined_i_c

        fc2_out = self.fc2(gated_combined)


        x = self.act(fc2_out) 
        return x + shortcut

class RTMoSRT(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        dim: int = 32,
        ffn_expansion: float = 2,
        n_blocks: int = 2,
        unshuffle_mod: bool = False,
        dccm: bool = True,
        se: bool = True,
    ) -> None:
        super().__init__()
        self.output_scale = scale 
        self.internal_scale = scale 
        unshuffle = 0
        if scale < 4 and unshuffle_mod:
            if scale == 3:
                raise ValueError("Unshuffle_mod does not support 3x")
            unshuffle = 4 // scale
            self.internal_scale = 4 
        self.pad_factor = 8

        if unshuffle > 0:
            self.to_feat = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                RepConv(3 * unshuffle * unshuffle, dim)
            )
        else:
            self.to_feat = RepConv(3, dim)

        self.body = nn.Sequential(
            *[
                GatedCNNBlock(dim, ffn_expansion, dccm=dccm, se=se)
                for _ in range(n_blocks)
            ]
        )
        
        self.to_img = nn.Sequential(
            RepConv(dim, 3 * self.internal_scale**2), 
            nn.PixelShuffle(self.internal_scale),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_img_size(self, x: Tensor) -> tuple[Tensor, int, int]: 
        """Pads input image x so height and width are divisible by self.pad_factor."""
        _b, _c, h, w = x.shape
        mod_pad_h = (self.pad_factor - h % self.pad_factor) % self.pad_factor
        mod_pad_w = (self.pad_factor - w % self.pad_factor) % self.pad_factor
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "replicate")
        return x, h, w 

    def forward(self, x: Tensor) -> Tensor: 
        inp_x = x 
        h_orig, w_orig = x.shape[2:] 

        out, _, _ = self.check_img_size(x) 

        out = self.to_feat(out)
        out = self.body(out)
        out = self.to_img(out)

        h_out = h_orig * self.output_scale
        w_out = w_orig * self.output_scale

        out_cropped = out[:, :, :h_out, :w_out]

        if self.output_scale > 1:
            residual = F.interpolate(
                inp_x,
                scale_factor=self.output_scale,
                mode='bilinear', 
                align_corners=False 
            )
            residual = residual[:, :, :h_out, :w_out]
        else:
             residual = inp_x[:, :, :h_out, :w_out]

        return out_cropped + residual

def RTMoSRT_L(**kwargs):
    return RTMoSRT(unshuffle_mod=True, **kwargs)


def RTMoSRT_UL(**kwargs):
    return RTMoSRT(ffn_expansion=1.5, dccm=False, unshuffle_mod=True, **kwargs)
