import os
import torch
from functools import partial
from torchvision.utils import save_image
from diffusers import QwenImagePipeline

from .diffusers_patch.modeling_qwen_image import QwenImage
from .unified_sampler import UnifiedSampler




# seed = 42
# torch.manual_seed(seed)
# device = torch.device("cuda")

# model_path = "path/to/model"
# prompt = "一张逼真的年轻东亚女性肖像，位于画面中心偏左的位置，带着浅浅的微笑直视观者。她身着以浓郁的红色和金色为主的传统中式服装。她的头发被精心盘起，饰有精致的红色和金色花卉和叶形发饰。她的眉心之间额头上绘有一个小巧、华丽的红色花卉图案。她左手持一把仿古扇子，扇面上绘有一位身着传统服饰的女性、一棵树和一只鸟的场景。她的右手向前伸出，手掌向上，托着一个悬浮的发光的霓虹黄色灯牌，上面写着“TwinFlow So Fast”，这是画面中最亮的元素。背景是模糊的夜景，带有暖色调的人工灯光，一场户外文化活动或庆典。在远处的背景中，她头部的左侧略偏，是一座高大、多层、被暖光照亮的西安大雁塔。中景可见其他模糊的建筑和灯光，暗示着一个繁华的城市或文化背景。光线是低调的，灯牌为她的脸部和手部提供了显著的照明。整体氛围神秘而迷人。人物的头部、手部和上半身完全可见，下半身被画面底部边缘截断。图像具有中等景深，主体清晰聚焦，背景柔和模糊。色彩方案温暖，以红色、金色和闪电的亮黄色为主。"
# height = 1024
# width = 768

# model = QwenImage(model_path, aux_time_embed=True, device=device)

# # 4 NFE config
# sampler_config = {
#     "sampling_steps": 4,
#     "stochast_ratio": 1.0,
#     "extrapol_ratio": 0.0,
#     "sampling_order": 1,
#     "time_dist_ctrl": [1.0, 1.0, 1.0],
#     "rfba_gap_steps": [0.001, 0.5],
# }

# # 2 NFE config
# sampler_config = {
#     "sampling_steps": 2,
#     "stochast_ratio": 1.0,
#     "extrapol_ratio": 0.0,
#     "sampling_order": 1,
#     "time_dist_ctrl": [1.0, 1.0, 1.0],
#     "rfba_gap_steps": [0.001, 0.6],
# }

# sampler = partial(UnifiedSampler().sampling_loop, **sampler_config)

# demox = model.sample(
#     [prompt],
#     cfg_scale=0.0, # should be zero
#     seed=seed,
#     height=height,
#     width=width,
#     sampler=sampler,
#     return_traj=False,
# )

# demox = demox.squeeze(0)  # [C, H, W]
# img_path = os.path.join(".", f"test.png")
# save_image((demox + 1) / 2, img_path)