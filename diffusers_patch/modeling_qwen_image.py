import torch
from diffusers import  QwenImageTransformer2DModel,GGUFQuantizationConfig
from .pipeline_qwenimage import QwenImagePipeline
from .transformer_qwenimage import QwenImageTransformer2DModelWrapper
from contextlib import contextmanager
import sys
import os
from omegaconf import OmegaConf
@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

class GenTransformer(torch.nn.Module):
    def __init__(self, transformer, vae_scale_factor, aux_time_embed) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.config.in_channels // 4
        self.vae_scale_factor = vae_scale_factor
        self.aux_time_embed = aux_time_embed

    def forward(self, x_t, t, c, tt=None):
        x_t_ = x_t.unsqueeze(1)
        packed_x_t = QwenImagePipeline._pack_latents(
            x_t_,
            x_t_.shape[0],
            x_t_.shape[2],
            x_t_.shape[3],
            x_t_.shape[4],
        )

        img_shapes = [[(1, x_t.shape[-2] // 2, x_t.shape[-1] // 2)]] * len(x_t)
        txt_seq_lens = c[1].int().sum(dim=1).tolist() if c[1] is not None else None
        txt_seq_lens = [int(i) for i in txt_seq_lens]

        encoder_hs = c[0][:, : max(txt_seq_lens)]
        encoder_hs_mask = c[1][:, : max(txt_seq_lens)]

        attn_mask = torch.cat(
            (
                torch.where(
                    encoder_hs_mask == 1,
                    torch.tensor(
                        0.0, device=packed_x_t.device, dtype=packed_x_t.dtype,
                    ),#encoder_hs_mask.dtype
                    torch.tensor(
                        float("-inf"),
                        device=packed_x_t.device,
                        dtype=packed_x_t.dtype,
                    ), #
                ),
                torch.zeros(
                    encoder_hs_mask.shape[0],
                    packed_x_t.shape[1],
                    device=packed_x_t.device,
                    dtype=packed_x_t.dtype,
                ),#encoder_hs_mask.dtype,
            ),
            dim=1,
        )
        attn_mask = (
            attn_mask[:, None, None, :]
            .expand(attn_mask.shape[0], 1, attn_mask.shape[1], attn_mask.shape[1])
            .contiguous()
        )

        transformer_kwargs = {
            "hidden_states": packed_x_t,
            "timestep": t,
            "guidance": None,
            "encoder_hidden_states_mask": encoder_hs_mask,
            "encoder_hidden_states": encoder_hs,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "attention_kwargs": {"attention_mask": attn_mask},
            "return_dict": False,
        }
        if self.aux_time_embed:
            assert tt is not None, "tt must be provided when aux_time_embed is True"
            transformer_kwargs["target_timestep"] = tt

        prediction = self.transformer(**transformer_kwargs)[0]

        prediction = QwenImagePipeline._unpack_latents(
            prediction,
            height=x_t.shape[-2] * self.vae_scale_factor,
            width=x_t.shape[-1] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        return prediction.squeeze(2)

    def forward_with_cfg(self, x, t, c, cfg_scale, cfg_interval=None, tt=None):
        t = t.flatten()
        if t[0] >= cfg_interval[0] and t[0] <= cfg_interval[1]:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, c)

            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            eps = torch.cat([half_eps, half_eps], dim=0)
            eps = torch.cat([eps, rest], dim=1)
        else:
            half = x[: len(x) // 2]
            t = t[: len(t) // 2]
            c = [c_[: len(c_) // 2] for c_ in c]
            half_eps = self.forward(half, t, c)
            eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


class QwenImage(torch.nn.Module):
    def __init__(
        self,
        model_id,
        dit_path=None,
        gguf_path=None,
        aux_time_embed=False,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        max_sequence_length=1024,
        device="cuda",
    ) -> None:
        super().__init__()

        self.aux_time_embed = aux_time_embed
    
        if aux_time_embed:
            with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModelWrapper):
                transformer_cls = QwenImageTransformer2DModelWrapper
        else:
            transformer_cls = QwenImageTransformer2DModel
        if dit_path is not None:
            qwen_transformer = transformer_cls.from_single_file(dit_path,config=os.path.join(model_id, "transformer"),torch_dtype=torch.bfloat16)  
        elif gguf_path is not None:
            qwen_transformer = transformer_cls.from_single_file(
                gguf_path,
                config=os.path.join(model_id, "transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,) 
        else:
            raise ValueError("Please provide either dit_path or gguf_path")
            
        # qwen_transformer = transformer_cls.from_pretrained(
        #     model_id,
        #     subfolder="transformer",
        #     torch_dtype=imgs_dtype,
        #     low_cpu_mem_usage=False,
        # )
        VAE=OmegaConf.load(os.path.join(model_id, "vae/config.json")) 
        self.model = QwenImagePipeline.from_pretrained(
            model_id, torch_dtype=imgs_dtype, transformer=qwen_transformer,VAE=VAE,
        )

        self.transformer = GenTransformer(
            self.model.transformer, self.model.vae_scale_factor, self.aux_time_embed
        ).to(device)

        self.device = device
        self.max_sequence_length = max_sequence_length

        self.imgs_dtype = imgs_dtype
        self.text_dtype = text_dtype

        # self.model.vae = (
        #     self.model.vae.to(dtype=self.imgs_dtype)
        #     .requires_grad_(False)
        #     .eval()
        #     .to(device)
        # )
        # self.model.text_encoder = (
        #     self.model.text_encoder.to(dtype=self.text_dtype)
        #     .requires_grad_(False)
        #     .eval()
        #     .to(device)
        # )

    def forward(self, x_t, t, c, tt=None):
        return self.transformer(x_t, t, c, tt)

    def train(self, mode: bool = True):
        self.transformer.train()
        return self

    def eval(self, mode: bool = True):
        self.transformer.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        self.transformer.requires_grad_(requires_grad)
        return self

    def encode_prompt(self, prompt, do_cfg=True):
        if do_cfg:
            input_args = {
                "prompt": tuple(prompt) + tuple(len(prompt) * ["Generate an image."]),
                "prompt_embeds": None,
                "prompt_embeds_mask": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }

            prompt_embeds_, prompt_attention_mask_ = self.model.encode_prompt(
                **input_args
            )
            prompt_embeds, neg_prompt_embeds = prompt_embeds_.chunk(2)
            prompt_attention_mask, neg_prompt_attention_mask = (
                prompt_attention_mask_.chunk(2)
            )
            del prompt_embeds_, prompt_attention_mask_

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                neg_prompt_embeds.to(self.imgs_dtype),
                neg_prompt_attention_mask.to(self.imgs_dtype),
            )
        else:
            input_args = {
                "prompt": prompt,
                "prompt_embeds": None,
                "prompt_embeds_mask": None,
                "device": self.device,
                "num_images_per_prompt": 1,
                "max_sequence_length": self.max_sequence_length,
            }
            prompt_embeds, prompt_attention_mask = self.model.encode_prompt(
                **input_args
            )

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                None,
                None,
            )

    @torch.no_grad()
    def pixels_to_latents(self, pixels):
        pixel_values = pixels.to(self.model.vae.dtype).unsqueeze(2)
        pixel_latents = self.model.vae.encode(pixel_values).latent_dist.mean
        pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

        latents_mean = (
            torch.tensor(self.model.vae.config.latents_mean)
            .view(1, 1, self.model.vae.config.z_dim, 1, 1)
            .to(pixel_latents.device, pixel_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
            1, 1, self.model.vae.config.z_dim, 1, 1
        ).to(pixel_latents.device, pixel_latents.dtype)
        pixel_latents = (pixel_latents - latents_mean) * latents_std
        return pixel_latents.squeeze(1)

    @torch.no_grad()
    def latents_to_pixels(self, latents):
        x_cur = latents.to(self.model.vae.dtype).unsqueeze(2)
        x_cur_mean = (
            torch.tensor(self.model.vae.config.latents_mean)
            .view(1, self.model.vae.config.z_dim, 1, 1, 1)
            .to(x_cur.device, x_cur.dtype)
        )
        x_cur_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
            1, self.model.vae.config.z_dim, 1, 1, 1
        ).to(x_cur.device, x_cur.dtype)
        latents = x_cur / x_cur_std + x_cur_mean
        pixels = self.model.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return pixels

    @torch.no_grad()
    def sample(
        self,
        prompts,
        cfg_scale=0.0,
        seed=42,
        height=512,
        width=512,
        times=1,
        return_traj=False,
        sampler=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        return_latents=True,
    ):
        do_cfg = cfg_scale > 0.0      
        
        if prompts is not None:    
            (
                prompt_embeds,
                prompt_attention_mask,
                neg_prompt_embeds,
                neg_prompt_attention_mask,
            ) = self.encode_prompt(prompts, do_cfg)
            batch_size = len(prompts)
        else:
            model_device = next(self.transformer.transformer.parameters()).device
            #print(model_device)
            prompt_embeds.to(model_device,self.imgs_dtype)
            prompt_attention_mask.to(model_device,self.imgs_dtype)
            batch_size=prompt_embeds.shape[0]


        if isinstance(seed, list):
            assert len(seed) == batch_size * times, (
                f"Length of seed list ({len(seed)}) must match total number of samples ({batch_size * times})"
            )
            noise = torch.cat(
                [
                    torch.randn(
                        [
                            1,
                            self.transformer.in_channels,
                            height // self.model.vae_scale_factor,
                            width // self.model.vae_scale_factor,
                        ],
                        dtype=self.imgs_dtype,
                        generator=torch.Generator(device="cpu").manual_seed(s),
                    )
                    for s in seed
                ],
                dim=0,
            ).cuda()
        else:
            noise = torch.randn(
                [
                    batch_size * times,
                    self.transformer.in_channels,
                    height // self.model.vae_scale_factor,
                    width // self.model.vae_scale_factor,
                ],
                dtype=self.imgs_dtype,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).cuda()

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )
            latents = torch.cat([noise] * 2)
            model_fn = self.transformer.forward_with_cfg
        else:
            latents = noise
            prompt_embeds = torch.cat(times * [prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(times * [prompt_attention_mask], dim=0)
            model_fn = self.transformer

        if do_cfg:
            model_kwargs = dict(
                c=[prompt_embeds, pooled_prompt_embeds],
                cfg_scale=cfg_scale,
                cfg_interval=[0.0, 1.0],
            )
        else:
            model_kwargs = dict(c=[prompt_embeds, pooled_prompt_embeds])

        latents = sampler(latents, model_fn, **model_kwargs)

        if do_cfg:
            latents, _ = latents.chunk(2, dim=1)

        if latents.shape[0] > 10:
            latents = latents[1::2].reshape(-1, *latents.shape[2:])
        else:
            latents = latents.reshape(-1, *latents.shape[2:])

        latents = latents if return_traj else latents[-batch_size * times :]
        if return_latents:
            x_cur = latents.to(self.imgs_dtype).unsqueeze(2)
            x_cur_mean = (
                torch.tensor(self.model.vae.latents_mean)
                .view(1, self.model.vae.z_dim, 1, 1, 1)
                .to(x_cur.device, x_cur.dtype)
            )
            x_cur_std = 1.0 / torch.tensor(self.model.vae.latents_std).view(
                1, self.model.vae.z_dim, 1, 1, 1
            ).to(x_cur.device, x_cur.dtype)
            latents = x_cur / x_cur_std + x_cur_mean
            return latents
        if return_traj:
            images = []
            for i in range(len(latents)):
                latent = latents[i : i + 1].cuda()
                image = self.latents_to_pixels(latent)
                images.append(image)
            images = torch.cat(images, dim=0)
            return images
        else:
            CHUNK_SIZE = 8
            if latents.shape[0] <= CHUNK_SIZE:
                images = self.latents_to_pixels(latents.cuda())
            else:
                images = torch.cat(
                    [
                        self.latents_to_pixels(chunk.cuda())
                        for chunk in latents.split(CHUNK_SIZE, dim=0)
                    ],
                    dim=0,
                )

        return images
