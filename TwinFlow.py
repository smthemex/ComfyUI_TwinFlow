 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from diffusers.hooks import apply_group_offloading
from .diffusers_patch.modeling_qwen_image import QwenImage
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
                io.Custom("TwinFlow_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None    
        model = QwenImage( os.path.join(node_cr_path, "Qwen-Image"),dit_path,gguf_path, aux_time_embed=True, device="cpu")   
        return io.NodeOutput(model)




class TwinFlow_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TwinFlow_SM_KSampler",
            display_name="TwinFlow_SM_KSampler",
            category="TwinFlow",
            inputs=[
                io.Custom("TwinFlow_SM_Model").Input("model"),
                io.Conditioning.Input("positive"), 
                io.Int.Input("width", default=1024, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Combo.Input("steps", [2,4]),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=10, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
            ], # io.Float.Input("noise", default=0.0, min=0.0, max=1.0,step=0.01,display_mode=io.NumberDisplay.number),
            outputs=[
                io.Latent.Output(display_name="latents"),
            ],
        )
    @classmethod
    def execute(cls, model,positive,width,height,steps,seed,block_num,) -> io.NodeOutput:

        batch_size, seq_len, _ = positive[0][0].shape
        prompt_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        prompt_attention_mask = prompt_attention_mask.repeat(1, 1, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * 1, seq_len)

        cf_models=mm.loaded_models()
        try:
            for pipe in cf_models:   
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
            model.transformer.transformer.to(device)
        # infer
        demox = model.sample(
            None ,
            cfg_scale=0.0, # should be zero
            seed=seed,
            height=height,
            width=width,
            sampler=sampler,
            prompt_embeds=positive[0][0],
            prompt_attention_mask=prompt_attention_mask,
        )
        print(demox.shape)
        if len(demox.shape)!=5:
            demox=demox.unsqueeze(0)
        out={"samples":demox} #BCTHW
        model.transformer.transformer.to("cpu")
        return io.NodeOutput(out)

from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/TwinFlow_SM_Extension")
async def get_hello(request):
    return web.json_response("TwinFlow_SM_Extension")

class TwinFlow_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TwinFlow_SM_Model,
            TwinFlow_SM_KSampler,
        ]
async def comfy_entrypoint() -> TwinFlow_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return TwinFlow_SM_Extension()
