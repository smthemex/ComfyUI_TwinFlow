 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
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
                io.Combo.Input("steps", [2,4]),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=10, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Combo.Input("force_offload", ["all", "none","clip"]),
            ], # io.Float.Input("noise", default=0.0, min=0.0, max=1.0,step=0.01,display_mode=io.NumberDisplay.number),
            outputs=[
                io.Latent.Output(display_name="latents"),
            ],
        )
    @classmethod
    def execute(cls, model,positive,width,height,steps,seed,block_num,force_offload) -> io.NodeOutput:
        raw_embeds=positive[0][0]
        if raw_embeds.dtype == torch.uint8 or not raw_embeds.is_floating_point(): #sometimes clip embeds are uint8 dtype @klossm
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
                        print("pass vae offload")
                        continue
                    pipe.unpatch_model(device_to=torch.device("cpu"))
                    print(f"Unpatching models.{pipe}")
            except: pass

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        max_gpu_memory = torch.cuda.max_memory_allocated()
        print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
       
        if 4==steps:
            # # 4 NFE config
            sampler_config = {
                "sampling_steps": 4,
                "stochast_ratio": 1.0,
                "extrapol_ratio": 0.0,
                "sampling_order": 1,
                "time_dist_ctrl": [1.0, 1.0, 1.0],
                "rfba_gap_steps": [0.001, 0.5],
            }
        else:
            # 2 NFE config
            sampler_config = {
                "sampling_steps": 2,
                "stochast_ratio": 1.0,
                "extrapol_ratio": 0.0,
                "sampling_order": 1,
                "time_dist_ctrl": [1.0, 1.0, 1.0],
                "rfba_gap_steps": [0.001, 0.6],
            }

        sampler = partial(UnifiedSampler().sampling_loop, **sampler_config)
        if block_num>0:
        # apply offloading
            apply_group_offloading(model.transformer.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        else:
            model.model.to(device)
            model.transformer.to(device)
            if hasattr(model.transformer, 'transformer'):
                model.transformer.transformer.to(device)
            model.device = device
        # infer
        demox = model.sample(
            None ,
            cfg_scale=0.0, # should be zero
            seed=seed,
            height=height,
            width=width,
            sampler=sampler,
            prompt_embeds=raw_embeds,
            prompt_attention_mask=prompt_attention_mask,
            block_num=block_num,
        )
        #print(demox.shape) #torch.Size([1, 16, 128, 96])
        if len(demox.shape)!=5 and isinstance(model, QwenImage): #qwen need 5D
            demox=demox.unsqueeze(0) 
        out={"samples":demox} #BCTHW
        # if block_num==0: # don't offload
        #     model.transformer.transformer.to("cpu")
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
                io.Combo.Input("lora_name", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength) -> io.NodeOutput:
        if strength == 0 or lora_name == "none" or not lora_name:
            return io.NodeOutput(model)

        # Try to find LoRA in ComfyUI loras folder first
        lora_path = None
        try:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        except:
            pass

        # If not found, try as direct path
        if lora_path is None and os.path.exists(lora_name):
            lora_path = lora_name

        if lora_path is None:
            print(f"Warning: LoRA '{lora_name}' not found")
            return io.NodeOutput(model)

        # Load LoRA using diffusers' load_lora_weights method
        try:
            # Both QwenImage and ZImage wrap a pipeline in self.model
            pipeline = getattr(model, 'model', None)
            if pipeline is None:
                # If no wrapped pipeline, try loading directly on model
                pipeline = model

            adapter_name = f"lora_{id(model)}"
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipeline.set_adapter(adapter_name)

            # Apply strength by scaling the adapter
            # Note: diffusers LoRA strength is typically applied during forward pass
            # For exact strength control, we may need to adjust adapter weights
            if strength != 1.0:
                print(f"LoRA strength {strength} requested. Note: diffusers LoRA uses weight_name param for adapter selection. Current implementation applies at full strength.")

            print(f"Successfully loaded LoRA: {lora_name}")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            import traceback
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
async def comfy_entrypoint() -> TwinFlow_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return TwinFlow_SM_Extension()
