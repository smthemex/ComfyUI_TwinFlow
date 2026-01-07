import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List, Optional, Dict, Any, Callable, Tuple

# Try to import ZImageTransformer2DModel from diffusers, fall back to local implementation
ZImageTransformer2DModel = None
TimestepEmbedder = None

try:
    from diffusers.models.transformers import ZImageTransformer2DModel
    from diffusers.models.transformers.transformer_z_image import TimestepEmbedder
except ImportError:
    try:
        from diffusers import ZImageTransformer2DModel
        from diffusers.models.transformers.transformer_z_image import TimestepEmbedder
    except ImportError:
        # Use local implementations
        from .local_z_image_components import ZImageTransformer2DModelBase as ZImageTransformer2DModel
        from .local_z_image_components import TimestepEmbedder

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.configuration_utils import register_to_config

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

class ZImageTransformer2DModelWrapper(ZImageTransformer2DModel):
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
    ):
        super().__init__(
            all_patch_size,
            all_f_patch_size,
            in_channels,
            dim,
            n_layers,
            n_refiner_layers,
            n_heads,
            n_kv_heads,
            norm_eps,
            qk_norm,
            cap_feat_dim,
            rope_theta,
            t_scale,
            axes_dims,
            axes_lens,
        )

        self.t_embedder_2 = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.t_embedder.float()
        self.t_embedder_2.float()

    def init_time_embed_2_weights(self):
        missing, unexpected = self.t_embedder_2.load_state_dict(self.t_embedder.state_dict())
        if len(missing) > 0:
            logger.warning(f"Missing keys in t_embedder state dict: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys in t_embedder state dict: {unexpected}")             

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        target_timestep=None, 
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        bsz = len(x)
        device = x[0].device

        calc_dtype = torch.float64
        t_high = t.to(dtype=calc_dtype)
        with torch.autocast(device_type=device.type, enabled=False):
            t_emb = self.t_embedder(t_high.abs() * self.t_scale)
            
            if target_timestep is not None:
                target_t_high = target_timestep.to(dtype=calc_dtype)
                delta_t = t_high - target_t_high
                delta_t_abs = delta_t.abs()
                t_emb_2 = self.t_embedder_2(target_t_high * self.t_scale - t_high * self.t_scale)
                t_emb = t_emb + \
                        t_emb_2 * delta_t_abs.unsqueeze(1) 

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
        adaln_input = t_emb.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
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
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)
        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
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

        return x, {}    