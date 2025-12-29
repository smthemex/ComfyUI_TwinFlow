# ComfyUI_TwinFlow
[TwinFlow](https://github.com/inclusionAI/TwinFlow): Realizing One-step Generation on Large Models with Self-adversarial Flows,you can use it in comfyUI

# Tips
* z image will coming soon ,
* infer 1027*768 in 12G Vram just need 15s / 12G (50 block)显存 开GPU卸载 单图12G vram只需要15秒 (50 block)
* if VRAM>16, set block number to 0 to use high VRAM / 大显存设置lock number为0 以达到最块推理速度；

# 1.Installation  

* In the ' ./ComfyUI/custom_nodes ' directory, run the following:   

```
git clone https://github.com/smthemex/ComfyUI_TwinFlow

```

# 2.requirements  

```
pip install -r requirements.txt
```

# 3.checkpoints 

* 3.1 [gguf](https://huggingface.co/smthem/TwinFlow-Qwen-Image-v1.0-diffusers-gguf/tree/main)  only support gguf now / 有一个 的其他开发者打包safetensor 但是没测试，迟点我整一个吧
* 3.2 [ z-image gguf and safetensors](https://huggingface.co/smthem/TwinFlow-Z-Image-Turbo-diffuser-gguf) / z-image的模型地址
* 3.3 qwen-image [（vae，clip）](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files)  /常规clip和vae
* 3.4 z-image [（vae，clip）](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files)  /常规z-image clip和vae

```
├── ComfyUI/models/gguf
|     ├── TwinFlow-Qwen-Image-diffusers-Q6_K.gguf  # or Q8_0、 BF16
├── ComfyUI/models/vae
|     ├──qwen_image_vae.safetensors
|     ├──ae.safetensors #z-image use flux
├── ComfyUI/models/clip 
|     ├──qwen_2.5_vl_7b_fp8_scaled.safetensors # or bf16
|     ├──qwen_3_4b.safetensors # z image
```

# 4. Example
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/example.png)


# 5. Citation
```
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}

```
