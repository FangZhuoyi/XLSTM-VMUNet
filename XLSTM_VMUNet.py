import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Attempt to import selective scan functions from different modules
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# Overriding the __repr__ method of DropPath for better visualization
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to compute FLOPs (Floating Point Operations) for selective scan (reference version)
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex
    flops = 0
    if False:
        flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
    return flops


# Patch embedding layer: Converts an image into a sequence of patches.
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        # If patch_size is a single integer, convert it into a tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # Create a convolutional layer to project image patches into embedding space
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Optionally add normalization to the embedded patches
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # Apply the convolution and permute the output tensor to the desired shape (B, L, C)
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)  # Apply normalization if defined
        return x


# Patch merging layer: Reduces the spatial resolution of the input (downsampling).
class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # Reduction layer to merge 4 patches into 2 patches (downsampling)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)  # Layer normalization

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]  # Adjust the shape if height/width is odd
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2
        # Split the input into 4 patches (downsampling)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        # Concatenate the 4 patches and reshape to reduce spatial resolution
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)  # Downsample
        x = self.norm(x)  # Normalize the downsampled patches
        x = self.reduction(x)  # Apply linear reduction to lower the channel dimension
        return x


# Patch expansion layer: Increases the spatial resolution by expanding the patches.
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # Increase the dimensionality of the feature space
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)  # Normalize after expansion

    def forward(self, x):
        B, H, W, C = x.shape
        # Apply the linear expansion
        x = self.expand(x)
        # Rearrange the expanded features back into the original spatial dimensions
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)  # Normalize the expanded features
        return x


# Final patch expansion layer: Similar to PatchExpand2D, but with a larger expansion factor.
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # Final layer to expand the patch features with a larger scaling factor
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)  # Normalize after expansion

    def forward(self, x):
        B, H, W, C = x.shape
        # Apply the linear expansion
        x = self.expand(x)
        # Rearrange the expanded features to the original spatial dimensions
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)  # Normalize the expanded features
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        # Device and dtype configuration for tensor operations
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Initialize the parent class (nn.Module)
        super().__init__()

        # Model dimensionality and configuration
        self.d_model = d_model
        self.d_state = d_state  # Defines the state size, default is 16
        self.d_conv = d_conv  # Convolution kernel size
        self.expand = expand  # Expansion factor for intermediate layer
        self.d_inner = int(self.expand * self.d_model)  # Inner dimension based on expansion
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # Dynamic time rank (auto or specified)

        # Projection layer for input features
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Depthwise convolution layer, applied independently to each input channel
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # Depthwise convolution (each input channel processed independently)
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,  # Padding to keep spatial dimensions intact
            **factory_kwargs,
        )
        
        # Activation function: SiLU (Sigmoid Linear Unit)
        self.act = nn.SiLU()

        # Projection layers for states (used in the selective scan mechanism)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )

        # Stack the weights of the projections into a single parameter tensor (K=4)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj  # Remove the original x_proj to save memory

        # Initialize the dynamic time projection layers (dt_projs)
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )

        # Stack the weights and biases of the time projections
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs  # Remove the original dt_projs to save memory

        # Initialize scaling parameters A_logs and Ds for the selective scan mechanism
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # Set the forward function to use `forward_corev0` by default
        self.forward_core = self.forward_corev0

        # Output normalization layer
        self.out_norm = nn.LayerNorm(self.d_inner)
        
        # Final projection to map to the original model's output dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Dropout layer (if specified)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        # Initialize dynamic time projection layer
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize weights based on specified method (either "constant" or "random")
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize biases using softplus transformation to ensure they are within a specific range
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus initialization
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True  # Mark this bias as not requiring reinitialization

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # Initialize A_log parameter, used for state transformation
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Log transform for A
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # Initialize the D parameter for skip connections
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    # Forward pass function (Version 0) of the core processing in the network.
    def forward_corev0(self, x: torch.Tensor):
        # Set the selective scan function
        self.selective_scan = selective_scan_fn

        # Get the batch size (B), channels (C), height (H), and width (W) from the input tensor's shape
        B, C, H, W = x.shape
        # L represents the number of elements in each spatial dimension (height * width)
        L = H * W
        # K is the number of different feature maps/representations to process (hardcoded to 4)
        K = 4

        # Create a tensor combining the original input and its transposed version across spatial dimensions
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # Concatenate the original tensor with its flipped version (across the spatial dimensions)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, K, D, L)

        # Project the concatenated tensor into a new feature space using a weight matrix
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # Split the projected tensor into multiple components (dts, Bs, Cs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # Apply further projection to the dts tensor using the dt_projs_weight matrix
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # Convert tensors to float and reshape as needed for further processing
        xs = xs.float().view(B, -1, L)  # (B, K * D, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K * D, L)
        Bs = Bs.float().view(B, K, -1, L)  # (B, K, D_state, L)
        Cs = Cs.float().view(B, K, -1, L)  # (B, K, D_state, L)
        Ds = self.Ds.float().view(-1)  # (K * D)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K * D, D_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K * D)

        # Perform the selective scan over the processed features
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # Ensure that the output is of type float
        assert out_y.dtype == torch.float

        # Process the output by flipping and reshaping
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # Return the processed output tensors
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # An alternative version of the forward_core function with a different selective scan function.
    def forward_corev1(self, x: torch.Tensor):
        # Set the selective scan function for the alternative implementation
        self.selective_scan = selective_scan_fn_v1

        # Get the batch size (B), channels (C), height (H), and width (W) from the input tensor's shape
        B, C, H, W = x.shape
        # L represents the number of elements in each spatial dimension (height * width)
        L = H * W
        # K is the number of different feature maps/representations to process (hardcoded to 4)
        K = 4

        # Create a tensor combining the original input and its transposed version across spatial dimensions
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # Concatenate the original tensor with its flipped version (across the spatial dimensions)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, K, D, L)

        # Project the concatenated tensor into a new feature space using a weight matrix
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # Split the projected tensor into multiple components (dts, Bs, Cs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # Apply further projection to the dts tensor using the dt_projs_weight matrix
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # Convert tensors to float and reshape as needed for further processing
        xs = xs.float().view(B, -1, L)  # (B, K * D, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K * D, L)
        Bs = Bs.float().view(B, K, -1, L)  # (B, K, D_state, L)
        Cs = Cs.float().view(B, K, -1, L)  # (B, K, D_state, L)
        Ds = self.Ds.float().view(-1)  # (K * D)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K * D, D_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K * D)

        # Perform the selective scan over the processed features
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)

        # Ensure that the output is of type float
        assert out_y.dtype == torch.float

        # Process the output by flipping and reshaping
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # Return the processed output tensors
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # Main forward pass function for the entire model.
    def forward(self, x: torch.Tensor, **kwargs):
        # Get the batch size (B), height (H), width (W), and channels (C) from the input tensor's shape
        B, H, W, C = x.shape

        # Project the input tensor through the initial projection layer to split into two parts
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, D)

        # Rearrange the input tensor for processing (channels -> spatial dims)
        x = x.permute(0, 3, 1, 2).contiguous()
        # Apply convolution and activation function to the tensor
        x = self.act(self.conv2d(x))  # (B, D, H, W)

        # Call the core forward function to get multiple outputs
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        # Combine the outputs (summation of feature maps)
        y = y1 + y2 + y3 + y4
        # Rearrange the combined output and normalize it
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        # Multiply the output by the gating factor z (element-wise multiplication)
        y = y * F.silu(z)

        # Project the final output to the desired shape
        out = self.out_proj(y)

        # Optionally apply dropout
        if self.dropout is not None:
            out = self.dropout(out)

        # Return the final output
        return out
    
    
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        # Initialization of the VSSBlock (Vision State Space Block)
        # This block is a basic unit for self-attention and feature fusion

        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # Layer normalization applied to input tensor
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)  # 2D self-attention mechanism
        self.drop_path = DropPath(drop_path)  # DropPath for stochastic depth regularization

    def forward(self, input: torch.Tensor):
        # Forward pass through the block: applies self-attention with residual connection
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))  # Add residual connection after applying attention and DropPath
        return x

    
class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim  # The dimensionality of input features
        self.use_checkpoint = use_checkpoint  # If true, uses gradient checkpointing for memory efficiency

        # Create a list of VSSBlock instances, one for each block in the layer
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # Apply stochastic depth for each block
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)]  # Create `depth` number of VSSBlock instances
        )
        
        # Initialize weights for some parameters (e.g., "out_proj.weight") using He initialization
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:  # Special initialization for specific layers
                    p = p.clone().detach_()  # Prevent modification of original tensor
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))  # Apply He initialization
        self.apply(_init_weights)  # Apply weight initialization to the entire module

        # Downsampling layer (optional): Applied only in the intermediate layers, not the final layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None  # No downsampling in the final stage

    def forward(self, x):
        # Forward pass through the layer
        # Each block processes the input sequentially
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)  # Use checkpointing if enabled to save memory
            else:
                x = blk(x)  # Otherwise, apply the block normally
        
        # If downsampling is needed (e.g., not the last layer), apply downsampling
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for the decoder (upsampling phase).
    Args are similar to `VSSLayer`, but it includes upsampling logic instead of downsampling.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim  # The dimensionality of input features
        self.use_checkpoint = use_checkpoint  # Whether to use checkpointing for memory efficiency

        # Create a list of VSSBlock instances, one for each block in the layer
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)]  # Create `depth` number of VSSBlock instances
        )
        
        # Initialize weights for specific parameters
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        # Upsampling layer (optional): Applied only in the decoder stages
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None  # No upsampling in the first layer (input layer)

    def forward(self, x):
        # Apply upsampling first (if needed)
        if self.upsample is not None:
            x = self.upsample(x)

        # Apply each block sequentially with or without checkpointing
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes  # Number of output classes for classification
        self.num_layers = len(depths)  # Number of encoder layers
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]  # Adjust dimensions for each layer
        self.embed_dim = dims[0]  # Embedding dimension for input patches
        self.num_features = dims[-1]  # Feature dimensionality in the final layer
        self.dims = dims

        # Patch embedding layer (splits image into patches and embeds them)
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # Absolute positional embedding (optional, used in some transformer variants)
        self.ape = False  # Disabled by default
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)  # Dropout after patch embedding

        # Stochastic depth for each layer in the encoder and decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder layers (composed of VSSLayer blocks)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Decoder layers (composed of VSSLayer_up blocks)
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # Final upsampling and convolution layer for segmentation/classification
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)  # Final output layer for classification

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Custom weight initialization for Linear and LayerNorm layers.
        This ensures consistency across layers and avoids issues during training.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # Truncated normal initialization
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Zero initialization for bias
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # Zero bias initialization for normalization
            nn.init.constant_(m.weight, 1.0)  # Initialize scale (gamma) to 1

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify layers that should not have weight decay applied (e.g., positional encodings)
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # Specify other parameters (e.g., relative position bias) that shouldn't have weight decay
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # Extract features from the input image (backbone part)
        skip_list = []
        x = self.patch_embed(x)  # Patch embedding of input
        if self.ape:
            x = x + self.absolute_pos_embed  # Add positional embedding if enabled
        x = self.pos_drop(x)  # Dropout after positional encoding

        # Pass the input through the encoder layers, collecting skip connections
        for layer in self.layers:
            skip_list.append(x)  # Store feature maps for skip connections
            x = layer(x)

        return x, skip_list

    def forward_features_up(self, x, skip_list):
        # Decode the features by passing through decoder layers and adding skip connections
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])  # Add skip connections from encoder

        return x

    def forward_final(self, x):
        # Final upsampling and convolution for generating the output
        x = self.final_up(x)  # Final patch expansion
        x = x.permute(0, 3, 1, 2)  # Change tensor shape for convolution
        x = self.final_conv(x)  # Apply final convolution for classification/segmentation
        return x

    def forward_backbone(self, x):
        # Forward pass through the backbone for feature extraction (without upsampling)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        # Full forward pass: extract features, apply decoder, and generate final output
        x, skip_list = self.forward_features(x)  # Forward through the encoder
        x = self.forward_features_up(x, skip_list)  # Forward through the decoder with skip connections
        x = self.forward_final(x)  # Final upsampling and convolution
        return x


class CausalConv1D(nn.Module):
    """
    A causal 1D convolutional layer, ensuring that the model only has access to current 
    and past values, not future values, when processing a sequence. This is commonly 
    used in time-series forecasting and causal modeling.

    Args:
        in_channels (int): Number of input channels (features).
        out_channels (int): Number of output channels (filters).
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): The dilation factor for the convolution. Default is 1.
        **kwargs: Additional keyword arguments to pass to the Conv1d layer, such as stride, padding, etc.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        
        # Calculate padding based on kernel size and dilation to maintain causal property
        self.padding = (kernel_size - 1) * dilation
        
        # Define the Conv1d layer with specified parameters
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        """
        Forward pass through the causal convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor after applying the causal convolution. Shape: (batch_size, out_channels, sequence_length - padding).
        """
        x = self.conv(x)
        
        # Remove the padded values to ensure no future information is used
        return x[:, :, :-self.padding]  # Slice off the padding part


class BlockDiagonal(nn.Module):
    """
    A linear layer where the input and output are split into several blocks, 
    and each block is passed through its own separate linear layer.
    The blocks are then concatenated together to form the final output.

    Args:
        in_features (int): The number of input features (dimension of the input tensor).
        out_features (int): The number of output features (dimension of the output tensor).
        num_blocks (int): The number of blocks to split the input and output into.
    """
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        # Ensure that input and output features can be divided evenly into blocks
        assert in_features % num_blocks == 0, "in_features must be divisible by num_blocks"
        assert out_features % num_blocks == 0, "out_features must be divisible by num_blocks"
        
        # Calculate the number of features per block for input and output
        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks
        
        # Create a list of Linear layers for each block
        self.blocks = nn.ModuleList([
            nn.Linear(block_in_features, block_out_features)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Forward pass through the block diagonal linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor after passing through the block-wise linear layers and concatenating.
        """
        # Ensure input tensor is on the right device (this assumes device is defined elsewhere)
        x = x.to(device)
        
        # Split the input tensor into multiple chunks along the last dimension (features)
        x = x.chunk(self.num_blocks, dim=-1)
        
        # Process each chunk with its respective linear layer and collect the results
        x = [block(x_i) for block, x_i in zip(self.blocks, x)]
        
        # Concatenate the results of all blocks along the last dimension
        x = torch.cat(x, dim=-1)
        
        return x

    
class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4 / 3):
        """
        Initialize a single sLSTM Block.
        
        Parameters:
        - input_size: The size of the input feature vector.
        - hidden_size: The size of the hidden state.
        - num_heads: The number of attention heads (used in the block-diagonal structure).
        - proj_factor: Factor for projecting the hidden state to a larger space (default is 4/3).
        """
        super(sLSTMBlock, self).__init__()
        
        # Store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads  # Size per attention head
        self.proj_factor = proj_factor
        
        # Assertions to ensure compatibility
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert proj_factor > 0, "Projection factor must be positive"
        
        # Layers
        self.layer_norm = nn.LayerNorm(input_size)  # Layer normalization on input
        self.causal_conv = CausalConv1D(1, 1, 4)  # Causal convolution to maintain temporal causality
        
        # Block-diagonal weight matrices for input gates, forget gates, etc.
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)  # Input gate weight
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)  # Forget gate weight
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)  # Forget gate weight
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)  # Output gate weight
        
        # Recurrent weight matrices (for hidden states)
        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)
        
        # Group normalization for hidden states
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        
        # Projections for hidden state scaling
        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)

    def forward(self, x, prev_state):
        """
        Forward pass of the sLSTMBlock.
        
        Parameters:
        - x: Input tensor of shape (batch_size, input_size)
        - prev_state: Tuple of previous hidden state (h_prev, c_prev, n_prev, m_prev)
        
        Returns:
        - final_output: Output of the block after processing the input
        - new_state: Tuple of updated states (h_t, c_t, n_t, m_t)
        """
        # Unpack previous states (h_prev: hidden state, c_prev: cell state, etc.)
        h_prev, c_prev, n_prev, m_prev = prev_state
        
        # Transfer states to device (GPU/CPU)
        H_prev = h_prev.to(device)
        C_prev = c_prev.to(device)
        N_prev = n_prev.to(device)
        M_prev = m_prev.to(device)
        
        # Normalize the input
        x_norm = self.layer_norm(x)
        
        # Apply causal convolution to input (only uses past context, no future leakage)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))  # Apply silu activation
        
        # Compute gate values: input (z), output (o), input candidate (i_tilde), and forget candidate (f_tilde)
        z = torch.tanh(self.Wz(x) + self.Rz(H_prev))  # Update gate
        o = torch.sigmoid(self.Wo(x) + self.Ro(H_prev))  # Output gate
        i_tilde = self.Wi(x_conv) + self.Ri(H_prev)  # Input candidate
        f_tilde = self.Wf(x_conv) + self.Rf(H_prev)  # Forget candidate
        
        # Compute max for numerically stable computation of i and f
        m_t = torch.max(f_tilde + M_prev, i_tilde)
        
        # Compute the gates using exponentiation for numerical stability
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + M_prev - m_t)
        
        # Update the cell state and the hidden state
        c_t = f * C_prev + i * z  # New cell state
        n_t = f * N_prev + i  # New state for normalizing hidden state
        h_t = o * c_t / n_t  # New hidden state
        
        # Apply group normalization on the hidden state
        output = h_t
        output_norm = self.group_norm(output)
        
        # Project the hidden state through left and right projections
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        
        # Apply GELU activation
        output_gated = F.gelu(output_right)
        
        # Final output through a gated projection
        output = output_left * output_gated
        
        # Down-project the output to match the input size
        output = self.down_proj(output)
        
        # Add residual connection
        final_output = output + x  # Skip connection
        
        return final_output, (h_t, c_t, n_t, m_t)


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=4 / 3):
        """
        Initialize the structured LSTM (sLSTM) with multiple layers.
        
        Parameters:
        - input_size: The size of the input feature vector.
        - hidden_size: The size of the hidden state.
        - num_heads: The number of attention heads (used in block-diagonal weight structure).
        - num_layers: Number of sLSTM blocks (layers).
        - batch_first: Whether the input batch dimension comes first (default is False).
        - proj_factor: Projection factor for hidden state scaling.
        """
        super(sLSTM, self).__init__()
        
        # Store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor
        
        # Create a list of sLSTM blocks (one block per layer)
        self.layers = nn.ModuleList(
            [sLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)])

    def forward(self, x, state=None):
        """
        Forward pass through all layers of the sLSTM.
        
        Parameters:
        - x: Input tensor of shape (seq_len, batch_size, input_size)
        - state: Optional initial hidden state, should be a tuple of (h_prev, c_prev, n_prev, m_prev)
        
        Returns:
        - output: Final output of shape (seq_len, batch_size, hidden_size)
        - state: Updated hidden states for each layer
        """
        assert x.ndim == 3, "Input must be a 3D tensor (seq_len, batch_size, input_size)"
        
        if self.batch_first: x = x.transpose(0, 1)  # If batch_first, transpose to (batch_size, seq_len, input_size)
        
        seq_len, batch_size, _ = x.size()
        
        # Initialize state if not provided
        if state is not None:
            state = torch.stack(list(state))
            assert state.ndim == 4, "State must be a 4D tensor"
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4, "State must have 4 elements (h, c, n, m)"
            assert state_num_layers == self.num_layers, "State must match the number of layers"
            assert state_batch_size == batch_size, "Batch size must match input batch size"
            assert state_input_size == self.input_size, "Input size must match state input size"
            state = state.transpose(0, 1)  # Transpose state for layer iteration
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)  # Initialize state with zeros
        
        # Iterate through each time step
        output = []
        for t in range(seq_len):
            x_t = x[t]  # Get the input at time step t
            for layer in range(self.num_layers):
                # Pass through each layer of sLSTM
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))  # Update state for the layer
            output.append(x_t)
        
        # Stack the output of all time steps
        output = torch.stack(output)
        
        # If batch_first is True, transpose the output back
        if self.batch_first:
            output = output.transpose(0, 1)
        
        # Return the final output and updated states
        state = tuple(state.transpose(0, 1))  # Transpose state for layer-wise access
        return output, state


class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()

        # Initialize parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads  # Size of each attention head
        self.proj_factor = proj_factor  # Projection factor for upsampling the input size

        assert hidden_size % num_heads == 0  # Ensure that hidden size is divisible by the number of heads
        assert proj_factor > 0  # Ensure that projection factor is positive

        # Define layers
        self.layer_norm = nn.LayerNorm(input_size)  # Layer normalization for input
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))  # Left-side projection
        self.up_proj_right = nn.Linear(input_size, hidden_size)  # Right-side projection to hidden size
        self.down_proj = nn.Linear(hidden_size, input_size)  # Down-projection from hidden size back to input size

        # Causal convolution for time-step specific processing
        self.causal_conv = CausalConv1D(1, 1, 4)  # Causal convolution (not defined in code, assumed to be a custom layer)

        # Skip connection for better gradient flow
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)

        # Block diagonal matrices for attention (for each head)
        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)  # Query matrix
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)  # Key matrix
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)  # Value matrix

        # LSTM-style gates
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)  # Input gate
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)  # Forget gate
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)  # Output gate

        # Group normalization for regularizing the hidden states
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        """
        Forward pass through the mLSTM block.

        Args:
            x (Tensor): The input tensor of shape (batch_size, input_size).
            prev_state (tuple): The previous state tuple consisting of:
                                 - h_prev (hidden state from previous time step)
                                 - c_prev (cell state from previous time step)
                                 - n_prev (additional state, assumed to be a regularizer)
                                 - m_prev (memory state)

        Returns:
            output (Tensor): The final output tensor of the block.
            state (tuple): The updated state tuple consisting of:
                           - h_t (new hidden state)
                           - c_t (new cell state)
                           - n_t (new regularizing state)
                           - m_t (new memory state)
        """
        h_prev, c_prev, n_prev, m_prev = prev_state

        # Ensure the tensors are on the correct device (GPU/CPU)
        H_prev = h_prev.to(device)
        C_prev = c_prev.to(device)
        N_prev = n_prev.to(device)
        M_prev = m_prev.to(device)

        # Normalize input
        x_norm = self.layer_norm(x)

        # Project the input in two ways
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)

        # Apply causal convolution (unsqueeze to add a batch dimension)
        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))

        # Apply skip connection
        x_skip = self.skip_connection(x_conv)

        # Attention mechanism
        q = self.Wq(x_conv)  # Query
        k = self.Wk(x_conv) / (self.head_size ** 0.5)  # Key (scaled by head size)
        v = self.Wv(x_up_left)  # Value

        # LSTM-style gates
        i_tilde = self.Wi(x_conv)  # Input gate
        f_tilde = self.Wf(x_conv)  # Forget gate
        o = torch.sigmoid(self.Wo(x_up_left))  # Output gate (sigmoid activation)

        # Memory and state update
        m_t = torch.max(f_tilde + M_prev, i_tilde)  # Updated memory state
        i = torch.exp(i_tilde - m_t)  # Input activation (exponentiated)
        f = torch.exp(f_tilde + M_prev - m_t)  # Forget activation (exponentiated)

        # Cell state update (LSTM-style)
        c_t = f * C_prev + i * (v * k)  # New cell state

        # Additional regularizing state update
        n_t = f * N_prev + i * k  # Regularizing state

        # Hidden state update
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]  # Normalized hidden state

        # Normalize output and apply skip connection
        output_norm = self.group_norm(h_t)
        output = output_norm + x_skip

        # Apply right-side projection and down-projection
        output = output * F.silu(x_up_right)  # Apply activation
        output = self.down_proj(output)  # Project back to input size

        # Add final residual connection
        final_output = output + x
        return final_output, (h_t, c_t, n_t, m_t)


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=2):
        super(mLSTM, self).__init__()

        # Initialize parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first  # If True, input is assumed to be of shape (batch_size, seq_len, input_size)
        self.proj_factor_slstm = proj_factor  # Projection factor for the layers

        # Create a list of mLSTMBlock layers
        self.layers = nn.ModuleList(
            [mLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)])

    def forward(self, x, state=None):
        """
        Forward pass through the multi-layer mLSTM.

        Args:
            x (Tensor): The input tensor of shape (seq_len, batch_size, input_size) if batch_first=False.
            state (tuple or None): The initial hidden state (h_0, c_0, n_0, m_0) for each layer, or None to initialize as zeros.

        Returns:
            output (Tensor): The output tensor after processing through all layers.
            state (tuple): The final hidden state after all layers, consisting of updated values for h_t, c_t, n_t, m_t.
        """
        assert x.ndim == 3  # Ensure input tensor has 3 dimensions (seq_len, batch_size, input_size)

        if self.batch_first:
            x = x.transpose(0, 1)  # If batch_first=True, input shape should be (batch_size, seq_len, input_size)

        seq_len, batch_size, _ = x.size()

        # Initialize state if not provided
        if state is not None:
            state = torch.stack(list(state))
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4  # Ensure the state contains 4 components (h, c, n, m)
            assert state_num_layers == self.num_layers  # Ensure state has the same number of layers
            assert state_batch_size == batch_size  # Ensure state has correct batch size
            assert state_input_size == self.input_size  # Ensure state has correct input size
            state = state.transpose(0, 1)  # Transpose for easier processing
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)  # Initialize state to zeros

        output = []
        for t in range(seq_len):
            x_t = x[t]  # Get the input at time step t
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))  # Pass through each layer
                state[layer] = torch.stack(list(state_tuple))  # Update the state for each layer

            output.append(x_t)  # Store the output at each time step

        output = torch.stack(output)  # Stack the outputs across all time steps

        if self.batch_first:
            output = output.transpose(0, 1)  # If batch_first=True, transpose output to (batch_size, seq_len, hidden_size)

        state = tuple(state.transpose(0, 1))  # Return the updated state
        return output, state


# Define the xLSTM class
class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, layers, batch_first=False, proj_factor_slstm=4 / 3,
                 proj_factor_mlstm=2):
        """
        Initializes the xLSTM model which contains multiple layers of sLSTM and mLSTM blocks.
        
        Parameters:
        - input_size (int): The size of the input feature vector.
        - hidden_size (int): The size of the hidden state in each LSTM block.
        - num_heads (int): The number of attention heads for multi-headed LSTM blocks.
        - layers (list): A list of layer types, where each type is either 's' (for sLSTM) or 'm' (for mLSTM).
        - batch_first (bool): Whether the input tensor has shape (batch, seq_len, feature) or (seq_len, batch, feature).
        - proj_factor_slstm (float): Projection factor for sLSTM blocks.
        - proj_factor_mlstm (float): Projection factor for mLSTM blocks.
        """
        super(xLSTM, self).__init__()
        
        # Store the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.num_layers = len(layers)  # Number of LSTM layers in the network
        self.batch_first = batch_first  # Set the flag for batch-first input
        self.proj_factor_slstm = proj_factor_slstm  # sLSTM projection factor
        self.proj_factor_mlstm = proj_factor_mlstm  # mLSTM projection factor
        
        # Initialize the list of LSTM layers (either sLSTM or mLSTM)
        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                # Initialize sLSTM block
                layer = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor_slstm)
            elif layer_type == 'm':
                # Initialize mLSTM block
                layer = mLSTMBlock(input_size, hidden_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)

    def forward(self, x, state=None):
        """
        Forward pass for the xLSTM network.

        Parameters:
        - x (tensor): The input sequence, shape (seq_len, batch_size, input_size).
        - state (tuple, optional): The initial hidden state for the LSTM layers.

        Returns:
        - output (tensor): The output sequence after passing through the LSTM layers.
        - state (tuple): The updated hidden state after processing the entire sequence.
        """
        # Ensure the input is a 3D tensor: (seq_len, batch_size, input_size)
        assert x.ndim == 3
        
        # If batch_first is True, we transpose the input to (batch_size, seq_len, input_size)
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()  # Extract sequence length and batch size
        
        # Initialize hidden states if none are provided
        if state is not None:
            # Stack the state if provided and check its dimensions
            state = torch.stack(list(state))
            assert state.ndim == 4  # (num_layers, num_hidden, batch_size, hidden_size)
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.input_size
            state = state.transpose(0, 1)  # Transpose state for easier access
        else:
            # If no state is provided, initialize the state to zeros
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size)

        output = []  # List to store the outputs for each time step
        
        # Loop through each time step in the sequence
        for t in range(seq_len):
            x_t = x[t]  # Extract the feature vector at time step t
            for layer in range(self.num_layers):
                # Pass the input through the LSTM block at this layer
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))  # Update the state for this layer
            output.append(x_t)  # Append the output for this time step

        # Stack all time step outputs into a single tensor
        output = torch.stack(output)
        
        # If batch_first is True, transpose the output back to (batch_size, seq_len, feature)
        if self.batch_first:
            output = output.transpose(0, 1)
        
        # Return the output and the updated state
        state = tuple(state.transpose(0, 1))
        return output, state


# Define the XLSTM_VMUNet class
class XLSTM_VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=True,
                 xLSTM_layers=['s', 'm'],  # List specifying the types of LSTM layers
                 xLSTM_input_size=256,  # Input size for xLSTM
                 xLSTM_hidden_size=64,  # Hidden size for xLSTM
                 xLSTM_num_heads=1,  # Number of attention heads for xLSTM
                 ):
        """
        Initializes the XLSTM_VMUNet model which integrates a VSSM model with xLSTM layers.
        
        Parameters:
        - input_channels (int): The number of input channels for the input image.
        - num_classes (int): The number of output classes for segmentation (binary or multi-class).
        - depths (list): Depths for the encoder layers in VSSM.
        - depths_decoder (list): Depths for the decoder layers in VSSM.
        - drop_path_rate (float): The drop path rate for the VSSM model (used for regularization).
        - load_ckpt_path (str or bool): Path to the checkpoint for loading pretrained weights.
        - xLSTM_layers (list): List of LSTM block types ('s' for sLSTM, 'm' for mLSTM).
        - xLSTM_input_size (int): Input feature size for xLSTM.
        - xLSTM_hidden_size (int): Hidden state size for xLSTM.
        - xLSTM_num_heads (int): Number of attention heads for xLSTM.
        """
        super().__init__()
        
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        # Initialize the VSSM model (likely a U-Net variant or similar architecture)
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate)
        
        # Initialize the xLSTM model
        self.xLSTM = xLSTM(xLSTM_input_size, xLSTM_hidden_size, xLSTM_num_heads, xLSTM_layers)

    def forward(self, x):
        """
        Forward pass for the XLSTM_VMUNet model.
        
        Parameters:
        - x (tensor): The input image tensor, shape (batch_size, channels, height, width).
        
        Returns:
        - output (tensor): The output of the segmentation network after passing through VSSM and xLSTM.
        """
        # If the input image has only one channel, replicate it to 3 channels (for RGB input)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Pass the input through the VSSM model to get logits
        logits = self.vmunet(x)
        
        # Permute logits to have the shape (batch_size, height * width, channels) for xLSTM input
        logits = logits.permute(0, 2, 3, 1)
        
        # Flatten the spatial dimensions (height and width) into a single sequence
        logits = logits.view(logits.size(0), logits.size(1), -1)
        
        batch_size, seq_len, channels = logits.shape
        
        # Pass the reshaped logits through xLSTM
        x, _ = self.xLSTM(logits)
        
        # Apply a sigmoid activation if this is a binary segmentation task (num_classes == 1)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        else:
            # Return raw logits if this is a multi-class segmentation task
            return x

    def load_from(self):
        # Check if a checkpoint path is provided for loading pretrained weights
        if self.load_ckpt_path is not None:
            # Initialize the model dictionary from the current model's state_dict
            model_dict = self.vmunet.state_dict()

            # Load the checkpoint from the provided path
            modelCheckpoint = torch.load(self.load_ckpt_path)
            # Extract the pretrained model weights from the checkpoint
            pretrained_dict = modelCheckpoint['model']

            # Filter and collect only the weights that match in both the current model and the pretrained model
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            # Update the current model's state_dict with the matching weights
            model_dict.update(new_dict)

            # Print the sizes of the model dict, pretrained dict, and how many weights were updated
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)
            ))

            # Load the updated model state_dict into the current model
            self.vmunet.load_state_dict(model_dict)

            # Identify keys that were in the pretrained_dict but not in the model_dict
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("Encoder loaded finished!")

            # Reload the checkpoint to adjust the decoder layers
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']

            # Initialize a new empty dictionary to store adjusted decoder weights
            pretrained_dict = {}

            # Map the original layer names to new layer names for the decoder
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            # Again, collect only matching weights between the updated pretrained dict and the model's state_dict
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            # Update the model with the adjusted decoder weights
            model_dict.update(new_dict)

            # Print the sizes of the model dict, pretrained dict, and how many weights were updated
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)
            ))

            # Load the updated model state_dict into the current model
            self.vmunet.load_state_dict(model_dict)

            # Identify keys that were in the pretrained_dict but not in the new_dict
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("Decoder loaded finished!")
            