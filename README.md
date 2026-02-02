# ComfyUI_TwinFlow
[TwinFlow](https://github.com/inclusionAI/TwinFlow): Realizing One-step Generation on Large Models with Self-adversarial Flows - use it in ComfyUI


## Update
* add z-image dype support，infer 3840*2160 use 2 step，need more Vram 
* 新增 z的dype支持，1步质量一般，超大尺寸需要多步数，或许也会崩，当然显存占用也会增大； 

## Previous 
* Fix if diffusers>0.36.0 got flash attn can't use attn mask error and MAC can't use attn2 (z-image only)  /  高版本或开发版的diffuser的z image 其attn mask 会报错（已知的官方错误），改成常规的，MAC的attn问题可能也是在这里，做了简单的修复；
* Fix lora can't use  error  修复lora无法使用的问题，加入lora adapter 清理代码，避免混用，可以参考示例图或新的lora节点
* upload new z-image gguf model，type is lumia2 now  上传新的z image gguf ，减少大小，解决context 都设置为f16的错误，类型改成lunmia2 ，跟city96大佬的一致；
* Now supports 1-step or any number of steps for image generation. Thanks to [QAQdev](https://github.com/QAQdev) for the code support - please give them a star!

## Requirements
* **diffusers >= 0.36.0** (required for Z-Image support)

## Tips
* LoRA support added via @oliveagle PR - use LoRA when inferring with 4 steps (untested, no guarantees)
* **Offload modes:**
  * `clip`: Only unload ComfyUI CLIP (recommended for most cases)
  * `none`: Don't unload anything (use if running many prompts repeatedly)
  * `all`: Unload all models
* Z-Image and Qwen-Image GGUF support - Qwen-Image GGUF has been re-quantized, please update to avoid dtype errors
* **Z-Image Q8 performance:**
  * 12GB VRAM: 1024x768 in 2-3s/image (without offloading)
  * 24GB VRAM: ~1.3s/image
* **Qwen-Image performance:**
  * 12GB VRAM (50 blocks): 1024x768 in ~15s/image with GPU offloading
* If VRAM > 16GB, set block number to 0 for maximum inference speed

## 1. Installation  

In the `./ComfyUI/custom_nodes` directory, run:

```bash
git clone https://github.com/smthemex/ComfyUI_TwinFlow
```

## 2. Requirements  

```bash
pip install -r requirements.txt
```

## 3. Checkpoints 

* 3.1 [Qwen-Image GGUF](https://huggingface.co/smthem/TwinFlow-Qwen-Image-v1.0-diffusers-gguf/tree/main) - GGUF format only (safetensors available from other developers)
* 3.2 [Z-Image GGUF and safetensors](https://huggingface.co/smthem/TwinFlow-Z-Image-Turbo-diffuser-gguf)
* 3.3 [Qwen-Image VAE & CLIP](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files)
* 3.4 [Z-Image VAE & CLIP](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files)

```
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

# 4. Example
* z-image dype
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/zimage_dype.png)

* qwen-image lora
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/example.png)
* z-image lora
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/example_z.png)

# 5. Citation
```
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}

```
