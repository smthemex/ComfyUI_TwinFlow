# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import math
import comfy.utils
import cv2
import folder_paths
from comfy.utils import common_upscale,ProgressBar
from safetensors.torch import load_file
cur_path = os.path.dirname(os.path.abspath(__file__))


def encode_image( image, vae):
    if image is None: 
        return None
    ref_latents=None
    samples = image.movedim(-1, 1)
    total = int(1024 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    image = s.movedim(1, -1)
    if vae is not None:
        ref_latents = vae.encode(image[:, :, :, :3])
    return ref_latents

def add_mean(latents):
    vae_config={"latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.916
        ],}
    latents_mean = (torch.tensor(vae_config["latents_mean"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(vae_config["latents_std"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image_latent_height, image_latent_width = latents.shape[3:]
    image_latents = pack_latents_(
        latents, 1, 16, image_latent_height, image_latent_width)
    return image_latents

def pack_latents_(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents



def load_lora(model, lora_1, lora_2, lora_scale1, lora_scale2):
    lora_path_1=folder_paths.get_full_path("loras", lora_1) if lora_1 != "none" else None
    lora_path_2=folder_paths.get_full_path("loras", lora_2) if lora_2 != "none" else None
    # lora_list=[i for i in [lora_path_1,lora_path_2] if i is not None]
    # lora_scales=[lora_scale1,lora_scale2]
    all_adapters = model.get_list_adapters()
    dit_list=[]
    if all_adapters:
        dit_list= all_adapters.get('transformer',[])+all_adapters.get('transformer_2',[])
    if lora_path_1 is not None:
        adapter_name=os.path.splitext(os.path.basename(lora_path_1))[0].replace(".", "_")
        dit_list2=all_adapters.get('transformer_2',[])
        if dit_list2:
            if adapter_name in dit_list: #dit_list
                pass
            else: 
                for i in dit_list2:
                    model.delete_adapters(i)
                    print(f"去除dit中未加载的lora: {i}")   
                try:  
                    model.load_lora_weights(lora_path_1, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                except KeyError as e:
                    try:
                        print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_1}")
                        state_dict = torch.load(lora_path_1, map_location="cpu",weights_only=False) if not lora_path_1.endswith(".safetensors") else load_file(lora_path_1,)
                        processed_state_dict = preprocess_lora_state_dict(state_dict)
                        model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                        model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                    except:
                        print(f"加载LoRA权重失败: {e}")
                        pass
        else: 
            try:  
                model.load_lora_weights(lora_path_1, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                model.set_adapters([adapter_name], adapter_weights=lora_scale1)
            except KeyError as e:
                try:
                    print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_1}")
                    state_dict = torch.load(lora_path_1, map_location="cpu",weights_only=False) if not lora_path_1.endswith(".safetensors") else load_file(lora_path_1,)
                    processed_state_dict = preprocess_lora_state_dict(state_dict)
                    model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                    del processed_state_dict
                except:
                    print(f"加载LoRA权重失败: {e}")
                    pass      
    if lora_path_2 is not None:
        adapter_name=os.path.splitext(os.path.basename(lora_path_2))[0].replace(".", "_")
        dit_list=all_adapters.get('transformer',[])
        if dit_list:
            if adapter_name in dit_list: #dit_list
                pass
            else: 
                for i in dit_list:
                    model.delete_adapters(i)
                    print(f"去除dit中未加载的lora: {i}")
                try:
                    model.load_lora_weights(lora_path_2, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                except KeyError as e:
                    try:
                        print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_2}")
                        state_dict = torch.load(lora_path_2, map_location="cpu",weights_only=False) if not lora_path_2.endswith(".safetensors") else load_file(lora_path_2,)
                        processed_state_dict = preprocess_lora_state_dict(state_dict)
                        model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                        model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                        del processed_state_dict
                    except:
                        print(f"加载LoRA权重失败: {e}")
                        pass
        else:
            try:
                model.load_lora_weights(lora_path_2, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                model.set_adapters([adapter_name], adapter_weights=lora_scale2)
            except KeyError as e:
                try:
                    print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_2}")
                    state_dict = torch.load(lora_path_2, map_location="cpu",weights_only=False) if not lora_path_2.endswith(".safetensors") else load_file(lora_path_2,)
                    processed_state_dict = preprocess_lora_state_dict(state_dict)
                    model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                    del processed_state_dict
                except:
                    print(f"加载LoRA权重失败: {e}")
                    pass

    return model


def preprocess_lora_state_dict(state_dict):

    processed_dict = state_dict.copy()
    keys_to_remove = [
        'head.head.diff_b',
        'head.head.diff_m',
        'head.head.diff',
        'patch_embedding.diff',
        'patch_embedding.diff_b',
        'blocks.*.diff_m',  # 匹配所有blocks的diff_m
        'head.head.lora_down'
        'diffusion_model.head.head.diff'
        'diffusion_model.head.head.diff_b'
        'diffusion_model.head.lora_down'
        
    ]
    keys_to_delete = []
    for key in processed_dict.keys():
        if key.endswith('.diff_m'):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        processed_dict.pop(key, None)
        print(f"移除键: {key}")    
    for key in keys_to_remove:
        if key in processed_dict:
            processed_dict.pop(key, None)
            print(f"移除键: {key}")
    return processed_dict

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img



def cv2tensor(img,bgr2rgb=True):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).permute(1, 2, 0).unsqueeze(0)  



def images_generator(img_list: list, ):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"
    
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images_list(img_list: list, ):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def get_video_files(directory, extensions=None):
    if extensions is None:
        extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
    extensions = [ext.lower() for ext in extensions]
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()[1:] 
            if ext in extensions:
                full_path = os.path.join(root, file)
                video_files.append(full_path)             
    return video_files
