from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 3-bit precision (0-7 values).
    Pack efficiently: 8 values into 3 bytes.
    Uses asymmetric quantization for better precision with minimal overhead.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0  # group_size must be multiple of 8 for packing

    x = x.view(-1, group_size)
    # Asymmetric quantization: track min and max separately
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    # Avoid division by zero
    scale = x_max - x_min
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Normalize to [0, 1] range using actual min/max
    x_norm = (x - x_min) / scale
    # Quantize to 3 bits (0-7) - use floor(x + 0.5) for better rounding
    x_quant = torch.floor(x_norm * 7 + 0.5).clamp(0, 7).to(torch.uint8)
    
    # Pack 8 values into 3 bytes (3 bits each = 24 bits)
    # Reshape to process 8 values at a time
    x_quant = x_quant.view(-1, 8)
    x_packed = torch.zeros(x_quant.size(0), 3, dtype=torch.uint8, device=x.device)
    
    # Pack 8 3-bit values into 3 bytes (24 bits)
    x_packed[:, 0] = x_quant[:, 0] | (x_quant[:, 1] << 3) | ((x_quant[:, 2] & 0x3) << 6)
    x_packed[:, 1] = ((x_quant[:, 2] >> 2) & 0x1) | (x_quant[:, 3] << 1) | (x_quant[:, 4] << 4) | ((x_quant[:, 5] & 0x1) << 7)
    x_packed[:, 2] = ((x_quant[:, 5] >> 1) & 0x3) | (x_quant[:, 6] << 2) | (x_quant[:, 7] << 5)
    
    # Flatten packed data
    x_packed = x_packed.view(-1, group_size // 8 * 3)
    
    # Stack min and max for storage (shape: num_groups, 2)
    params = torch.stack([x_min.squeeze(-1), x_max.squeeze(-1)], dim=-1).to(torch.float16)
    
    return x_packed, params


def block_dequantize_3bit(x_packed: torch.Tensor, params: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Dequantize from 3-bit back to float32.
    params contains [min, max] for each group.
    """
    assert group_size % 8 == 0
    
    # x_packed has shape (num_groups * group_size // 8 * 3,)
    # Each group of 32 values is stored in 12 bytes (32 // 8 * 3)
    num_groups = params.size(0)
    bytes_per_group = group_size // 8 * 3
    
    # Reshape to (num_groups, bytes_per_group)
    x_packed = x_packed.view(num_groups, bytes_per_group)
    
    # Reshape to process 3 bytes at a time (which unpacks to 8 values)
    x_packed = x_packed.view(-1, 3)
    
    # Unpack 8 values from 3 bytes
    x_quant = torch.zeros(x_packed.size(0), 8, dtype=torch.uint8, device=x_packed.device)
    x_quant[:, 0] = x_packed[:, 0] & 0x7
    x_quant[:, 1] = (x_packed[:, 0] >> 3) & 0x7
    x_quant[:, 2] = ((x_packed[:, 0] >> 6) & 0x3) | ((x_packed[:, 1] & 0x1) << 2)
    x_quant[:, 3] = (x_packed[:, 1] >> 1) & 0x7
    x_quant[:, 4] = (x_packed[:, 1] >> 4) & 0x7
    x_quant[:, 5] = ((x_packed[:, 1] >> 7) & 0x1) | ((x_packed[:, 2] & 0x3) << 1)
    x_quant[:, 6] = (x_packed[:, 2] >> 2) & 0x7
    x_quant[:, 7] = (x_packed[:, 2] >> 5) & 0x7
    
    # Reshape to match group structure (num_groups, group_size)
    x_quant = x_quant.view(num_groups, group_size)
    
    # Extract min and max
    params = params.to(torch.float32)
    x_min = params[:, 0:1]
    x_max = params[:, 1:2]
    scale = x_max - x_min
    
    # Dequantize
    x_norm = x_quant.to(torch.float32) / 7.0
    x = x_norm * scale + x_min
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 64) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        num_elements = out_features * in_features
        num_groups = num_elements // group_size
        bytes_per_group = group_size // 8 * 3  # 8 values in 3 bytes
        num_packed = num_groups * bytes_per_group
        
        self.register_buffer(
            "weight_q3",
            torch.zeros(num_packed, dtype=torch.uint8),
            persistent=False,
        )
        # Store min and max for each group (shape: num_groups, 2)
        self.register_buffer(
            "weight_params",
            torch.zeros(num_groups, 2, dtype=torch.float16),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            
            # Quantize on original device to avoid device transfer precision loss
            weight_flat = weight.view(-1)
            packed, params = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.data = packed.view(-1)
            self.weight_params.data = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight_dequant = block_dequantize_3bit(self.weight_q3, self.weight_params, self._group_size)
            weight_dequant = weight_dequant.view(self._shape)
            return torch.nn.functional.linear(x, weight_dequant, self.bias)


class BigNet3Bit(torch.nn.Module):
    """
    A BigNet where all weights use 3-bit quantization for extra credit.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=64),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=64),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=64),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    # Extra credit: 3-bit quantization with proper bit packing to get below 9MB
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
