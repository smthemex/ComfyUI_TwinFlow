# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple
from einops import rearrange
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers import __version__ as diffusers_version

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        # Compute joint attention
        if diffusers_version > "0.36.0" or not  FLASH_ATTN_2_AVAILABLE:
            hidden_states=flash_attention(query, key, value,attn.heads,False, attention_mask)
        else:
             # Compute joint attention
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )


        # Reshape back
        if hidden_states.dim() == 4:
            # 从 [batch, seq_len, heads, head_dim] -> [batch, seq_len, heads * head_dim]
            hidden_states = hidden_states.flatten(2, 3)
        elif hidden_states.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected hidden_states dimensions: {hidden_states.dim()}")
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Attention block
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x
    
class RopeEmbedder(nn.Module):
    """
    Z-Image RoPE embedder with DyPE support and enhanced aspect ratio handling
    """
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
        dype: bool = False,  
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.dype = dype  # Enable DyPE by default

        self.current_timestep = 1.0  # 1.0 = pure noise, 0.0 = clean image
        self.base_resolution = 1024  
        self.patch_size = 16
        self.base_patches = self.base_resolution // self.patch_size  # 64

        # DyPE specific parameters - more conservative settings
        self.dype_start_sigma = 0.95  # Start DyPE earlier - NOW ACTUALLY USED
        self.dype_end_sigma = 0.70  # End DyPE later - NOW ACTUALLY USED
        self.dype_exponent = 1.0      # Reduced exponent to make DyPE less aggressive
        self.dype_scale = 1.0         # Scale factor for DyPE

    def set_timestep(self, timestep: float):
        """Set current timestep for DyPE. 
        Timestep is normalized to [0, 1] range where 1.0 = pure noise, 0.0 = clean image."""
        if self.dype:
            self.current_timestep = timestep

    def rope_params(self, index, dim, device=None):
        """Basic RoPE parameter calculation"""
        assert dim % 2 == 0
        if device is None:
            device = index.device if isinstance(index, torch.Tensor) else torch.device('cpu')

        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device=device)
        else:
            index = index.to(device)

        theta_tensor = torch.tensor(self.theta, dtype=torch.float32, device=device)
        arange_tensor = torch.arange(0, dim, 2, dtype=torch.float32, device=device)

        freqs = torch.outer(index.float(), 1.0 / torch.pow(theta_tensor, arange_tensor / dim))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @staticmethod
    def find_newbase_ntk(dim, base, scale):
        """Calculate the new base for NTK-aware scaling."""
        return base * (scale ** (dim / (dim - 2)))

    @staticmethod
    def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
        """Find the correction range for NTK-by-parts interpolation."""
        import numpy as np
        low = np.floor((dim * math.log(ori_max_pe_len / (low_ratio * 2 * math.pi))) / (2 * math.log(base)))
        high = np.ceil((dim * math.log(ori_max_pe_len / (high_ratio * 2 * math.pi))) / (2 * math.log(base)))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def linear_ramp_mask(min_val, max_val, dim, device=None):
        if min_val == max_val:
            max_val += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    def init_scale_rope(self, scale=1.0):
        """
        Initialize scale for rope considering both dimensions equally
        """
        self.input_scale = max(scale, 1.0)


    def get_scale(self, h_patches, w_patches):
        self.h_patches = h_patches
        self.w_patches = w_patches

    def rope_params_yarn(self, index, dim, device, axis_idx=0):
        """Enhanced YARN with balanced DyPE across all spatial dimensions"""

       
        if hasattr(self, 'input_scale') and self.input_scale > 1.0:
            current_patches = self.input_scale * min(self.h_patches, self.w_patches)
        else:
            current_patches = min(self.h_patches, self.w_patches)


        if current_patches <= self.base_patches or not self.dype:
            return self.rope_params(index, dim, device=device)

        scale = max(1.0, current_patches / self.base_patches)

        # YARN parameters
        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device=device, dtype=torch.float32)
        else:
            index = index.to(device).float()

        theta_tensor = torch.tensor(self.theta, dtype=torch.float32, device=device)
        arange_tensor = torch.arange(0, dim, 2, dtype=torch.float32, device=device)

        freqs_base = torch.outer(
            index,
            1.0 / torch.pow(theta_tensor, arange_tensor / dim)
        )

        freqs_linear = torch.outer(
            index,
            1.0 / (scale * torch.pow(theta_tensor, arange_tensor / dim))
        )

        new_base = self.find_newbase_ntk(dim, self.theta, scale)
        new_base_tensor = torch.tensor(new_base, dtype=torch.float32, device=device)
        freqs_ntk = torch.outer(
            index,
            1.0 / torch.pow(new_base_tensor, arange_tensor / dim)
        )
        # CRITICAL FIX: 使用统一的DyPE参数，确保所有维度的一致性
        if self.dype:
            beta_0 = beta_0 ** (2.0 * (self.current_timestep ** 2.0))
            beta_1 = beta_1 ** (2.0 * (self.current_timestep ** 2.0))
        
        # 第一次插值：线性和NTK之间
        low, high = self.find_correction_range(beta_0, beta_1, dim, self.theta, self.base_patches)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = (1 - self.linear_ramp_mask(low, high, dim // 2, device=device))
        freqs = freqs_linear * (1 - freqs_mask.unsqueeze(0)) + freqs_ntk * freqs_mask.unsqueeze(0)

        if self.dype:
            gamma_0 = gamma_0 ** (2.0 * (self.current_timestep ** 2.0))
            gamma_1 = gamma_1 ** (2.0 * (self.current_timestep ** 2.0))
        # 第二次插值：结果和基础之间
        low, high = self.find_correction_range(gamma_0, gamma_1, dim, self.theta, self.base_patches)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = (1 - self.linear_ramp_mask(low, high, dim // 2, device=device))
        freqs = freqs * (1 - freqs_mask.unsqueeze(0)) + freqs_base * freqs_mask.unsqueeze(0)

        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        # Apply scaling correction if scale > 1
        if scale > 1:
            mscale = 0.1 * math.log(scale) + 1.0
            freqs_complex = freqs_complex * mscale

        return freqs_complex

        
    def forward(
        self,
        pos_ids: torch.Tensor,  
        device: torch.device = None,
    ):
        """
        Compute RoPE with enhanced DyPE adjustment based on timestep and spatial resolution
        """
        device = device or pos_ids.device

        assert pos_ids.ndim == 2
        assert pos_ids.shape[-1] == 3  # F, H, W 

        result = []
        pos = pos_ids.float()

        for i in range(3):  
            index = pos[:, i]
            if self.dype: #i > 0:  # Apply scaling for height and width dimensions
                # Create mask to distinguish positive and negative indices
               if self.dype:  
                freqs_complex = self.rope_params_yarn(
                    index, self.axes_dims[i],  device, axis_idx=i)
            else:
                freqs_complex = self.rope_params(index, self.axes_dims[i], device)

            result.append(freqs_complex)

        return torch.cat(result, dim=-1)

class RopeEmbedder_:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        dype: bool = True,  
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.dype = dype
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False
        
        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens,dype = dype)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask if cap_padding_len > 0 else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)
            )

            # padded feature
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
            self.rope_embedder.get_scale( H_tokens, W_tokens)
            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padded_pos_ids = torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        if self.dype:
            # 归一化时间步到[0,1]范围，1表示纯噪声
            normalized_timestep = (t / self.t_scale).clamp(0.0, 1.0)
            normalized_timestep = 1.0 - normalized_timestep
            self.rope_embedder.set_timestep(normalized_timestep)
        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis, adaln_input)
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis)
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                unified = self._gradient_checkpointing_func(
                    layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):

    if attention_mask is not None:# [batch, seq_len, heads, head_dim]
        batch_size, seq_len_q, heads, head_dim = q.shape
        seq_len_k = k.shape[1]  # 获取键序列长度

        if attention_mask.dim() == 4:
            if attention_mask.shape[1] == 1 and attention_mask.shape[3] == seq_len_k:

                attention_mask = attention_mask.expand(-1, heads, -1, -1)
            elif attention_mask.shape[1] == 1 and attention_mask.shape[3] == seq_len_q:
                attention_mask = attention_mask.transpose(-1, -2)
                attention_mask = attention_mask.expand(-1, heads, -1, -1)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, -1, seq_len_q, seq_len_k)
            attention_mask = attention_mask.to(torch.bool)
        
        if attention_mask.dtype == torch.bool:
            attn_mask_float = torch.where(
                attention_mask, 
                torch.zeros_like(attention_mask, dtype=q.dtype),  # 0值
                torch.full_like(attention_mask, float("-inf"), dtype=q.dtype)  # 负无穷
            )
        else:
            attn_mask_float = attention_mask.to(dtype=q.dtype)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            attn_mask=attn_mask_float
        ).transpose(1, 2)
        return x.reshape(q.shape[0], q.shape[1], -1)  # reshape to [batch, seq_len, hidden_dim]

    elif compatibility_mode:
        q_reshaped = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k_reshaped = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v_reshaped = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q_reshaped = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k_reshaped = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v_reshaped = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q_reshaped, k_reshaped, v_reshaped)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q_reshaped = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k_reshaped = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v_reshaped = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q_reshaped, k_reshaped, v_reshaped)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q_reshaped = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k_reshaped = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v_reshaped = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q_reshaped, k_reshaped, v_reshaped)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q_reshaped = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k_reshaped = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v_reshaped = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x