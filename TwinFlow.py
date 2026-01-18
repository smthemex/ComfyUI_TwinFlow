# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import re
import traceback
from diffusers.hooks import apply_group_offloading
from .diffusers_patch.modeling_qwen_image import QwenImage
from .diffusers_patch.z_image.modeling_z_image import ZImage
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm
from functools import partial
from .unified_sampler import UnifiedSampler

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  cotyle dir

# --- 辅助函数：规范化适配器名称 ---
def get_safe_name(path):
    base_name = os.path.splitext(os.path.basename(path))[0]
    # 将中文、空格、特殊字符统一替换为下划线，防止 peft/diffusers 无法识别
    safe_name = re.sub(r'[^\w]', '_', base_name)
    if safe_name[0].isdigit():
        safe_name = "lora_" + safe_name
    return safe_name

class TwinFlow_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_SM_Model",
            display_name="TwinFlow_SM_Model",
            category="TwinFlow",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),  
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None

        if any(path and 'qwen' in path.lower() for path in [dit_path, gguf_path]):
            model = QwenImage( os.path.join(node_cr_path, "Qwen-Image"),dit_path,gguf_path, aux_time_embed=True, device="cpu")   
        else:
            model = ZImage( os.path.join(node_cr_path, "Z-Image"),dit_path,gguf_path, aux_time_embed=True, device="cpu")
        return io.NodeOutput(model)


class TwinFlow_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_SM_KSampler",
            display_name="TwinFlow_SM_KSampler",
            category="TwinFlow",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"), 
                io.Int.Input("width", default=1024, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("steps", default=2, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=10, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Combo.Input("force_offload", ["all", "none","clip"], default="none"),
                io.Combo.Input("sampling_style", ["any", "mul",], default="any"),
            ], 
            outputs=[
                io.Latent.Output(display_name="latents"),
            ],
        )
    @classmethod
    def execute(cls, model,positive,width,height,steps,seed,block_num,force_offload,sampling_style) -> io.NodeOutput:
        raw_embeds=positive[0][0]
        if raw_embeds.dtype == torch.uint8 or not raw_embeds.is_floating_point():
            raw_embeds = raw_embeds.to(torch.float32)
        batch_size, seq_len, _ = raw_embeds.shape
        prompt_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        prompt_attention_mask = prompt_attention_mask.repeat(1, 1, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * 1, seq_len)

        if force_offload!="none":
            cf_models=mm.loaded_models()
            try:
                for pipe in cf_models:
                    if force_offload=="clip" and "AutoencodingEngine"==type(pipe.model).__name__: 
                        continue
                    pipe.unpatch_model(device_to=torch.device("cpu"))
            except: pass

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
       
        if 2>=steps:
            stochast_ratio=0.8 if steps==1 else 1.0
            sampler_config = {
                "sampling_steps": steps, 
                "stochast_ratio": stochast_ratio,
                "extrapol_ratio": 0.0,
                "sampling_order": 1,
                "time_dist_ctrl": [1.0, 1.0, 1.0],
                "rfba_gap_steps": [0.001, 0.7],
                "sampling_style": "few"
            }
        elif 2<steps and sampling_style=="any":
            sampler_config = {
                "sampling_steps": steps,
                "stochast_ratio": 0.0,
                "extrapol_ratio": 0.0,
                "sampling_order": 1,
                "time_dist_ctrl": [1.0, 1.0, 1.0],
                "rfba_gap_steps": [0.001, 0.5],
                "sampling_style": sampling_style
                }
        else:
            sampler_config = {
                "sampling_steps": steps,
                "stochast_ratio": 0.0,
                "extrapol_ratio": 0.0,
                "sampling_order": 1,
                "time_dist_ctrl": [1.17, 0.8, 1.1],
                "rfba_gap_steps": [0.001, 0.0],
                "sampling_style": sampling_style
            }

        sampler = partial(UnifiedSampler().sampling_loop, **sampler_config)
        if block_num>0:
            apply_group_offloading(model.transformer.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        else:
            model.model.to(device)
            model.transformer.to(device)
            if hasattr(model.transformer, 'transformer'):
                model.transformer.transformer.to(device)
            model.device = device

        demox = model.sample(
            None ,
            cfg_scale=0.0, 
            seed=seed,
            height=height,
            width=width,
            sampler=sampler,
            prompt_embeds=raw_embeds,
            prompt_attention_mask=prompt_attention_mask,
            block_num=block_num,
        )
        if len(demox.shape)!=5 and isinstance(model, QwenImage):
            demox=demox.unsqueeze(0) 
        out={"samples":demox} 

        return io.NodeOutput(out)


class TwinFlow_SM_LoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_SM_LoraLoader",
            display_name="TwinFlow_SM_LoraLoader",
            category="TwinFlow",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_1", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("lora_2", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_1", default=1.0, min=-0.1, max=10.0, step=0.1, display_mode=io.NumberDisplay.number),
                io.Float.Input("strength_2", default=1.0, min=-0.1, max=10.0, step=0.1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_1, lora_2, strength_1, strength_2) -> io.NodeOutput:
        # 1. 整理有效的 LoRA 路径和对应的权重
        lora_configs = []
        if lora_1 != "none":
            path = folder_paths.get_full_path("loras", lora_1)
            if path: lora_configs.append({"path": path, "scale": strength_1, "name": get_safe_name(path)})
        
        if lora_2 != "none":
            path = folder_paths.get_full_path("loras", lora_2)
            if path: lora_configs.append({"path": path, "scale": strength_2, "name": get_safe_name(path)})

        # 获取模型中当前已有的适配器列表
        try:
            all_adapters_dict = model.model.get_list_adapters()
            present_adapters = all_adapters_dict.get('transformer', [])
        except:
            present_adapters = []

        # 2. 清理不需要的适配器 (重要：防止干扰和 set() 报错)
        needed_names = [c["name"] for c in lora_configs]
        for old_name in present_adapters:
            if old_name not in needed_names:
                try:
                    model.model.delete_adapters(old_name)
                    print(f"[TwinFlow] 已卸载不再使用的 LoRA: {old_name}")
                except: pass

        if not lora_configs:
            return io.NodeOutput(model)

        # 3. 加载缺失的权重并记录激活列表
        final_adapter_names = []
        final_scales = []
        
        for config in lora_configs:
            name = config["name"]
            path = config["path"]
            
            if name not in present_adapters:
                print(f"[TwinFlow] 正在加载 LoRA: {name}")
                try:
                    model.model.load_lora_weights(path, adapter_name=name)
                except Exception as e:
                    print(f"[TwinFlow] LoRA 加载失败 ({name}): {e}")
                    traceback.print_exc()
                    continue # 跳过加载失败的

            final_adapter_names.append(name)
            final_scales.append(config["scale"])

        # 4. 统一激活适配器
        if final_adapter_names:
            try:
                print(f"[TwinFlow] 激活适配器: {final_adapter_names}, 权重: {final_scales}")
                model.model.set_adapters(final_adapter_names, adapter_weights=final_scales)
            except Exception as e:
                print(f"[TwinFlow] 激活适配器失败: {e}")
                traceback.print_exc()

        return io.NodeOutput(model)


class TwinFlow_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TwinFlow_SM_Model,
            TwinFlow_SM_KSampler,
            TwinFlow_SM_LoraLoader,
        ]
async def comfy_entrypoint() -> TwinFlow_SM_Extension:
    return TwinFlow_SM_Extension()
