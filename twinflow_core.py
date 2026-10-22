# -*- coding: UTF-8 -*-
"""
TwinFlow core integration for ComfyUI (Z-Image first).

Reproduces the TwinFlow (https://github.com/inclusionAI/TwinFlow) few-step
sampling on top of ComfyUI's *native* DiT inference flow:

- Model loading / detection / memory management: comfy.sd + ModelPatcher
  (the checkpoint is a raw Z-Image state dict, auto-converted by
  comfy.model_detection.convert_diffusers_mmdit -> z_image_to_diffusers).
- The only model-side difference vs. stock Z-Image is the extra
  ``t_embedder_2`` (delta-timestep embedder).  It is attached to the loaded
  comfy NextDiT instance and the comfy ``t_embedder`` is wrapped so that

      adaln = emb(t') + emb2((tt' - t') * 1000) * |t' - tt'|

  with t' = 1 - sigma (comfy Z-Image convention: sigma = mixture fraction),
  tt' = 1 - target_sigma.  This is bit-for-bit the same math as the
  diffusers wrapper (transformer_z_image_.forward, t_scale=1000).
- The target timestep travels through the standard
  model_options["transformer_options"] channel (no comfy source changes).
- The UnifiedSampler (UCGM-S, Linear transport) is reimplemented as a
  comfy.samplers.Sampler; model calls go through CFGGuider/KSamplerX0Inpaint,
  so CFG, hooks, patches, inpaint all keep working.
"""

import math
import os
import sys
import weakref
from functools import partial

import torch

import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.ops
import comfy.model_management
import folder_paths
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy_api.latest import io


# ---------------------------------------------------------------------------
# model-side patch: t_embedder_2 + wrapper around comfy t_embedder
# ---------------------------------------------------------------------------

TWINFLOW_TARGET_KEY = "twinflow_target_timestep"  # in transformer_options, sigma-space target


class _TwinTEmbedderWrapper(torch.nn.Module):
    """Replaces NextDiT.t_embedder. Keeps the original embedding and, when a
    TwinFlow target timestep is present in transformer_options, adds the
    delta embedding exactly like TwinFlow's patched diffusers transformer."""

    def __init__(self, inner: TimestepEmbedder, t_embedder_2: TimestepEmbedder,
                 dm, time_scale: float):
        super().__init__()
        self.inner = inner
        self.t_embedder_2 = t_embedder_2
        # weakref: never register the parent DiT as a child module (would
        # create a cycle in the module tree and break state_dict()).
        self._dm_ref = weakref.ref(dm)
        self.time_scale = float(time_scale)

    @property
    def _dm(self):
        return self._dm_ref()

    def forward(self, t, dtype, **kwargs):
        base = self.inner(t, dtype)
        tt_sigma = getattr(self._dm, "_twinflow_target", None)
        if tt_sigma is None:
            return base
        # t arrives already multiplied by time_scale: t = t' * scale,
        # where t' = 1 - sigma in [0, 1] (0 = pure noise).
        if not torch.is_tensor(tt_sigma):
            tt_sigma = torch.tensor(float(tt_sigma), device=t.device)
        t_prime = t.detach().float() / self.time_scale
        # exact mirror of the diffusers wrapper transform
        #   tt' = sign(tt) * (1 - |tt|)   (sign(0) = 0 -> tt=0 maps to 0, NOT 1)
        # tt arrives in sampler/sigma space (1 = noise, 0 = data), tt >= 0.
        tt = tt_sigma.detach().float().reshape(-1)
        tt_prime = torch.where(tt > 0, 1.0 - tt.abs(), torch.zeros_like(tt))
        # mirror of diffusers patch:
        #   delta   = t' - tt'
        #   t_emb_2 = T2((tt' - t') * scale)
        #   t_emb   = T1(t' * scale) + t_emb_2 * |delta|
        delta = t_prime - tt_prime
        emb2 = self.t_embedder_2((tt_prime - t_prime) * self.time_scale, dtype=torch.float32)
        out = base.float() + emb2 * delta.abs().unsqueeze(1)
        return out.to(dtype)


def _make_t_embedder_2(operations):
    # TwinFlow: TimestepEmbedder(min(dim, 256), mid_size=1024) for dim=3840
    # -> Linear(256, 1024) + Linear(1024, 256); comfy's TimestepEmbedder with
    # hidden_size=1024 / output_size=256 has identical shapes and key names.
    return TimestepEmbedder(
        hidden_size=1024, output_size=256, frequency_embedding_size=256,
        dtype=torch.float32, device=torch.device("cpu"), operations=operations,
    )


def _twinflow_pre_hook(module, args, kwargs):
    topts = kwargs.get("transformer_options") or {}
    module._twinflow_target = topts.get(TWINFLOW_TARGET_KEY, None)
    if os.environ.get("TWINFLOW_DEBUG"):
        print("[TwinFlow] model call: target_tt = {}".format(module._twinflow_target), flush=True)
    return None


def attach_twinflow_zimage(patcher, sd):
    """Attach the TwinFlow delta-timestep embedder to a loaded comfy Z-Image
    (Lumina2/NextDiT) model.  sd is the *raw* checkpoint state dict (keys like
    t_embedder_2.mlp.0.weight).  Returns the ModelPatcher for chaining."""
    model = patcher.model
    dm = model.diffusion_model
    if getattr(dm, "_twinflow_attached", False):
        return patcher

    ops = comfy.ops.manual_cast  # weights stay fp32, cast per-compute
    t2 = _make_t_embedder_2(ops)

    weight_keys = ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]
    to_load = {}
    for suffix in weight_keys:
        k = "t_embedder_2." + suffix
        if k not in sd:
            raise RuntimeError(
                "[TwinFlow] '{}' not found in checkpoint - this file is not a "
                "TwinFlow Z-Image model (stock Z-Image has no t_embedder_2).".format(k))
        to_load[suffix] = sd[k].float().cpu()
    missing, unexpected = t2.load_state_dict(to_load, strict=True)
    if missing or unexpected:
        raise RuntimeError("[TwinFlow] t_embedder_2 load mismatch: {} / {}".format(missing, unexpected))

    time_scale = float(getattr(dm, "time_scale", 1000.0))
    wrapper = _TwinTEmbedderWrapper(dm.t_embedder, t2, dm, time_scale)
    dm.t_embedder = wrapper
    if not getattr(dm, "_twinflow_hook_attached", False):
        dm.register_forward_pre_hook(_twinflow_pre_hook, with_kwargs=True)
        dm._twinflow_hook_attached = True
    dm._twinflow_attached = True
    return patcher


def load_twinflow_zimage(model_path, model_options=None):
    """Load a TwinFlow Z-Image single-file checkpoint through comfy core and
    attach the TwinFlow delta embedder.  Returns a ModelPatcher."""
    model_options = dict(model_options or {})
    sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
    # core loading consumes the state dict (convert_diffusers_mmdit pops keys),
    # so reserve the TwinFlow-specific tensors first.
    twin_sd = {k: sd.pop(k) for k in list(sd.keys()) if k.startswith("t_embedder_2.")}
    patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
    if patcher is None:
        raise RuntimeError("[TwinFlow] comfy could not detect this checkpoint as a diffusion model: {}".format(model_path))
    attach_twinflow_zimage(patcher, twin_sd)
    return patcher


# ---------------------------------------------------------------------------
# Qwen-Image variant
# ---------------------------------------------------------------------------
# TwinFlow modifies the diffusers QwenImageTransformer2DModel by adding
# ``time_text_embed_2`` (QwenTimestepProjEmbeddings, same shape as
# time_text_embed) and computing
#
#     temb = time_text_embed(timestep) + time_text_embed_2(target) * (timestep - target)
#
# with timestep and target both in [0, 1] (1 = noise).  No abs(), no 1000
# scaling on the delta weight, and no sign() quirk (unlike the Z-Image patch).
# comfy's native Qwen-Image uses multiplier=1.0, so the DiT already receives
# sigma in [0, 1] == the TwinFlow timestep convention: no unit conversion.


class _TwinQwenTEmbedWrapper(torch.nn.Module):
    """Replaces QwenImageTransformer2DModel.time_text_embed."""

    def __init__(self, inner, time_text_embed_2, dm):
        super().__init__()
        self.inner = inner
        self.time_text_embed_2 = time_text_embed_2
        self._dm_ref = weakref.ref(dm)

    @property
    def _dm(self):
        return self._dm_ref()

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        base = self.inner(timestep, hidden_states, addition_t_cond)
        tt = getattr(self._dm, "_twinflow_target", None)
        if tt is None:
            return base
        if not torch.is_tensor(tt):
            tt = torch.tensor(float(tt), device=timestep.device)
        tt = tt.detach().float().reshape(-1)
        # exact mirror of transformer_qwenimage.forward:
        #   temb_2 = time_text_embed_2(target_timestep, hidden_states)
        #   temb   = temb + temb_2 * (timestep - target_timestep)
        emb2 = self.time_text_embed_2(tt.to(hidden_states.dtype), hidden_states)
        delta = (timestep.detach().float().reshape(-1) - tt).unsqueeze(1)
        return base.float() + emb2.float() * delta


def _make_qwen_time_embed_2(operations, embedding_dim):
    from comfy.ldm.qwen_image.model import QwenTimestepProjEmbeddings
    return QwenTimestepProjEmbeddings(
        embedding_dim=embedding_dim, pooled_projection_dim=None,
        use_additional_t_cond=False,
        dtype=torch.float32, device=torch.device("cpu"), operations=operations,
    )


def attach_twinflow_qwen(patcher, twin_sd):
    """Attach TwinFlow time_text_embed_2 to a loaded comfy Qwen-Image DiT.
    twin_sd holds the reserved time_text_embed_2.* tensors."""
    model = patcher.model
    dm = model.diffusion_model
    if getattr(dm, "_twinflow_attached", False):
        return patcher

    ops = comfy.ops.manual_cast
    t2 = _make_qwen_time_embed_2(ops, dm.inner_dim)

    to_load = {}
    prefix = "time_text_embed_2.timestep_embedder."
    for k in list(twin_sd.keys()):
        if k.startswith(prefix):
            to_load[k[len(prefix):]] = twin_sd[k].float().cpu()
    if not to_load:
        raise RuntimeError(
            "[TwinFlow] 'time_text_embed_2.*' tensors not found - this file is not a "
            "TwinFlow Qwen-Image model (stock Qwen-Image has no time_text_embed_2).")
    missing, unexpected = t2.timestep_embedder.load_state_dict(to_load, strict=True)
    if missing or unexpected:
        raise RuntimeError("[TwinFlow] time_text_embed_2 load mismatch: {} / {}".format(missing, unexpected))

    wrapper = _TwinQwenTEmbedWrapper(dm.time_text_embed, t2, dm)
    dm.time_text_embed = wrapper
    if not getattr(dm, "_twinflow_hook_attached", False):
        dm.register_forward_pre_hook(_twinflow_pre_hook, with_kwargs=True)
        dm._twinflow_hook_attached = True
    dm._twinflow_attached = True
    return patcher


def _ensure_custom_nodes_on_path():
    cn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if cn_dir not in sys.path:
        sys.path.insert(0, cn_dir)


def _dequant_to_float(t):
    """GGUF-quantized tensors (Dif_GGUF) -> float32; plain tensors pass through."""
    try:
        from ComfyUI_Dif_GGUF.ggml_ops import dequantize_tensor
        return dequantize_tensor(t, torch.float32)
    except Exception:
        return t.float()


def load_twinflow_qwen(model_path, model_options=None):
    """Load a TwinFlow Qwen-Image checkpoint through comfy core.
    Supports diffusers-named safetensors (comfy native detection) and
    GGUF via the ComfyUI_Dif_GGUF plugin (lazy-dequant reader/loader).
    Returns a ModelPatcher."""
    model_options = dict(model_options or {})
    if model_path.endswith(".gguf"):
        _ensure_custom_nodes_on_path()
        try:
            from ComfyUI_Dif_GGUF.dif_gguf_node import read_gguf, load_gguf_model, GGMLOps, GGUFModelPatcher, is_quantized
        except Exception as e:
            raise RuntimeError(
                "[TwinFlow] GGUF loading requires the ComfyUI_Dif_GGUF plugin "
                "(custom_nodes/ComfyUI_Dif_GGUF): {}".format(e))
        sd = read_gguf(model_path)
        twin_sd = {k: sd.pop(k) for k in list(sd.keys()) if k.startswith("time_text_embed_2.")}
        has_quant = any(is_quantized(v) for v in sd.values())
        patcher = load_gguf_model(sd, GGMLOps() if has_quant else None)
        if patcher is None:
            raise RuntimeError("[TwinFlow] comfy could not detect the GGUF model: {}".format(model_path))
        if has_quant:
            patcher = GGUFModelPatcher.clone(patcher)
            patcher.size = 0
        twin_sd = {k: _dequant_to_float(v) for k, v in twin_sd.items()}
        attach_twinflow_qwen(patcher, twin_sd)
        return patcher

    sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
    twin_sd = {k: sd.pop(k) for k in list(sd.keys()) if k.startswith("time_text_embed_2.")}
    patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
    if patcher is None:
        raise RuntimeError("[TwinFlow] comfy could not detect this checkpoint as a diffusion model: {}".format(model_path))
    if type(patcher.model).__name__ != "QwenImage":
        raise RuntimeError("[TwinFlow] checkpoint detected as {} instead of QwenImage - wrong file?".format(type(patcher.model).__name__))
    attach_twinflow_qwen(patcher, twin_sd)
    return patcher


# ---------------------------------------------------------------------------
# sampler-side: UnifiedSampler (UCGM-S, Linear transport) over comfy machinery
# ---------------------------------------------------------------------------

def _kumaraswamy(t, a, b, c):
    return (1 - (1 - t**a) ** b) ** c


# Built-in per-step presets, verbatim from the official TwinFlow inference.py.
# The official configs are NOT per-model: the same table serves Qwen-Image and
# Z-Image alike; only rfba_gap_steps[1] varies with the step count (2-NFE 0.6 /
# 4-NFE 0.5).  stochast_ratio and time_dist_ctrl are constant, so they are
# baked in here instead of being exposed as node inputs.
def _auto_config(steps):
    if steps <= 2:
        return dict(stochast_ratio=1.0, rfba_gap_steps=[0.001, 0.6], time_dist_ctrl=[1.0, 1.0, 1.0])
    return dict(stochast_ratio=1.0, rfba_gap_steps=[0.001, 0.5], time_dist_ctrl=[1.0, 1.0, 1.0])


def _build_twinflow_timesteps(steps, time_dist_ctrl, rfba_gap_steps):
    """Exact port of UnifiedSampler.sampling_loop schedule (sampling_order=1).
    Returns mixture fractions (== comfy sigma space), descending, ending 0."""
    num_steps = steps
    if (rfba_gap_steps[1] - 0.0) == 0.0:
        num_steps = num_steps + 1
    t_steps = torch.linspace(rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64)
    if (rfba_gap_steps[1] - 0.0) == 0.0:
        t_steps = t_steps[:-1]
    t_steps = _kumaraswamy(t_steps, *time_dist_ctrl)
    t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])
    return t_steps


def twinflow_sampling_function(model_k, noise, sigmas, extra_args, callback, disable,
                               sampling_style="few"):
    """
    TwinFlow UnifiedSampler loop.  Always uses the TwinFlow schedule with the
    official configs baked in via _auto_config (keyed by the actual step
    count; identical for Qwen-Image and Z-Image).

    model_k: KSamplerX0Inpaint wrapping the comfy CFGGuider; called as
             model_k(x, sigma, denoise_mask, model_options, seed) -> x0 pred.
    sigma == TwinFlow sampler t (Linear transport, t=1 noise / t=0 data).
    """
    model_options = extra_args["model_options"]
    seed = extra_args.get("seed", 0)
    denoise_mask = extra_args.get("denoise_mask", None)
    topts = model_options.setdefault("transformer_options", {})

    preset = _auto_config(len(sigmas) - 1)
    stochast_ratio = preset["stochast_ratio"]
    t_steps = _build_twinflow_timesteps(
        len(sigmas) - 1, list(preset["time_dist_ctrl"]),
        list(preset["rfba_gap_steps"])).to(noise.device)

    total_steps = len(t_steps) - 1
    pbar = comfy.utils.ProgressBar(total_steps)

    input_dtype = noise.dtype
    x_cur = noise.to(torch.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        if sampling_style == "few":
            t_tgt = torch.zeros_like(t_cur)
        elif sampling_style == "mul":
            t_tgt = t_cur
        elif sampling_style == "any":
            t_tgt = t_next
        else:
            raise ValueError("Unknown sampling_style: {}".format(sampling_style))

        # target timestep in sigma space -> DiT via transformer_options
        topts[TWINFLOW_TARGET_KEY] = float(t_tgt)

        # comfy expects a per-batch 1-D timestep tensor
        t_in = t_cur.to(input_dtype).reshape(1).expand(x_cur.shape[0])
        denoised = model_k(x_cur.to(input_dtype), t_in,
                           denoise_mask, model_options=model_options, seed=seed)
        D = denoised.to(torch.float64)

        # Linear transport (alpha_in=t, gamma_in=1-t, integ_st=1):
        #   x_hat = x_t - t * F ;  z_hat = x_t + (1-t) * F
        # with F = (x_t - x0_pred) / t   (velocity, noise - x0 direction)
        t_c = t_cur if t_cur.abs() > 1e-6 else torch.tensor(1e-6, device=t_cur.device, dtype=torch.float64)
        F_t = (x_cur - D) / t_c
        x_hat = D
        z_hat = x_cur + (1.0 - t_c) * F_t

        if stochast_ratio > 0:
            gen = torch.Generator(device="cpu").manual_seed((seed + 1 + i) % (2**31 - 1))
            noi = torch.randn(x_cur.size(), generator=gen, dtype=torch.float32).to(x_cur)
        else:
            noi = 0.0
        sr = float(stochast_ratio)
        x_next = (1.0 - t_next) * x_hat + t_next * (z_hat * ((1 - sr) ** 0.5) + noi * (sr ** 0.5))

        x_cur = x_next
        pbar.update(1)
        if callback is not None:
            callback({"i": i, "denoised": x_hat.to(input_dtype), "x": x_next.to(input_dtype),
                      "sigma": float(t_cur), "total_steps": total_steps})

    topts.pop(TWINFLOW_TARGET_KEY, None)
    return x_cur.to(torch.float32)


# ---------------------------------------------------------------------------
# V3 nodes
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fn_fast": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


class TwinFlow_ZImage_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_ZImage_Loader",
            display_name="TwinFlow Z-Image Loader (core)",
            category="TwinFlow",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model_name, weight_dtype="default") -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model_options = {}
        if weight_dtype != "default":
            model_options["dtype"] = _DTYPE_MAP[weight_dtype]
            if weight_dtype == "fp8_e4m3fn_fast":
                model_options["fp8_optimizations"] = True
        patcher = load_twinflow_zimage(model_path, model_options=model_options)
        return io.NodeOutput(patcher)


class TwinFlow_QwenImage_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # GGUF checkpoints live in models/gguf, safetensors in models/diffusion_models
        files = list(dict.fromkeys(
            folder_paths.get_filename_list("gguf") + folder_paths.get_filename_list("diffusion_models")))
        return io.Schema(
            node_id="TwinFlow_QwenImage_Loader",
            display_name="TwinFlow Qwen-Image Loader (core)",
            category="TwinFlow",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=files),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                               tooltip="safetensors only; GGUF ignores this"),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model_name, weight_dtype="default") -> io.NodeOutput:
        model_path = folder_paths.get_full_path("gguf", model_name)
        if model_path is None:
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model_options = {}
        if weight_dtype != "default" and not model_path.endswith(".gguf"):
            model_options["dtype"] = _DTYPE_MAP[weight_dtype]
            if weight_dtype == "fp8_e4m3fn_fast":
                model_options["fp8_optimizations"] = True
        patcher = load_twinflow_qwen(model_path, model_options=model_options)
        return io.NodeOutput(patcher)


class TwinFlow_Sampler_Select(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_Sampler_Select",
            display_name="TwinFlow Sampler (UnifiedSampler)",
            category="TwinFlow",
            is_experimental=True,
            inputs=[
                io.Combo.Input("sampling_style", options=["few", "any", "mul"],
                               tooltip="few: official default (target = data, t_tgt=0); "
                                       "any: target = next sigma; mul: target = current sigma"),
            ],
            outputs=[io.Sampler.Output(display_name="sampler")],
        )

    @classmethod
    def execute(cls, sampling_style) -> io.NodeOutput:
        sampler_function = partial(
            twinflow_sampling_function,
            sampling_style=sampling_style,
        )
        sampler = comfy.samplers.KSAMPLER(sampler_function)
        return io.NodeOutput(sampler)
