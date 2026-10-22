# ComfyUI_TwinFlow (core)

[TwinFlow](https://github.com/inclusionAI/TwinFlow): *Realizing One-step Generation on Large Models with Self-adversarial Flows* — running on ComfyUI's **native DiT inference flow** for both Z-Image and Qwen-Image.

# Update
* Use comfyUI core to fit twinflow /使用comfyUI核心来适配twinflow,避免diffuser版本问题

# Checkpoints 

*  [Qwen-Image GGUF/comfy](https://huggingface.co/smthem/TwinFlow-Qwen-Image-v1.0-diffusers-gguf/tree/main) 
*  [Z-Image GGUF and safetensors/comfy](https://huggingface.co/smthem/TwinFlow-Z-Image-Turbo-diffuser-gguf)
*  [Qwen-Image VAE & CLIP/comfy](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files)
*  [Z-Image VAE & CLIP/comfy](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files)

```
├── ComfyUI/models/diffusion_models/ 
|     ├── TwinFlow-Qwen-Image-v1.0-int8_convrot.safetensors # or Q6-K, Q8_0, int8_convrot, w4a8 推荐
|     ├── TwinFlow-Z-Image-Turbo-comfy_int8_convrot.safetensors # or Q6-K, Q8_0, BF16
├── ComfyUI/models/gguf
|     ├── TwinFlow-Qwen-Image-diffusers-Q6_K.gguf  # or Q8_0、 BF16
|     ├── TwinFlow-Z-Image-Turbo-diffuser-Q8_0.gguf # or Q6-k，BF16
├── ComfyUI/models/vae
|     ├──qwen_image_vae.safetensors
|     ├──ae.safetensors #z-image use flux vae
├── ComfyUI/models/clip 
|     ├──qwen_2.5_vl_7b_fp8_scaled.safetensors # or bf16
|     ├──qwen_3_4b.safetensors # z image
```

## Example

* qwen-image
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/qwen-twinflow.png)

* z-image 
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/z-image-twinflow.png)

## Notes

- The official TwinFlow demo runs **without CFG** (`cfg_scale=0.0`, "should be zero") for
  TwinFlow-Qwen-Image. In ComfyUI terms that means setting the KSampler **CFG = 1.0** (ComfyUI's
  `cfg=1` skips the uncond pass; `cfg=0` would output the *unconditional* prediction instead).
  CFG > 1 still runs but deviates from the official configuration — with 1–2 step generation
  there are no later steps to absorb the distortion, so quality usually degrades.
- No shift node is needed: the TwinFlow schedule replaces ComfyUI's sigma schedule entirely
  (`ModelSamplingFlux` / `ModelSamplingDiscreteFlow` shift only shapes the stock schedule, and
  the DiT receives the raw sigma either way).

## Citation

```
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}
```
