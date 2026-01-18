import re
import traceback

# --- 新增：规范化适配器名称，解决中文和特殊字符导致的 ValueError ---
def sanitize_adapter_name(name):
    if name is None:
        return "default_adapter"
    # 获取文件名（不含路径和后缀）
    base_name = os.path.splitext(os.path.basename(name))[0]
    # 将所有非字母数字字符替换为下划线
    sanitized = re.sub(r'[^\w]', '_', base_name)
    # 确保不以数字开头
    if sanitized[0].isdigit():
        sanitized = "lora_" + sanitized
    return sanitized

# --- 优化：预处理 LoRA 权重字典 ---
def preprocess_lora_state_dict(state_dict):
    processed_dict = state_dict.copy()
    # 定义 Qwen 系列 LoRA 中需要剔除的冗余键值
    black_list = [
        '.diff_b', '.diff_m', '.diff', 
        'patch_embedding.diff', 
        'head.head.lora_down'
    ]
    
    keys_to_delete = [
        key for key in processed_dict.keys() 
        if any(mark in key for mark in black_list)
    ]
    
    for key in keys_to_delete:
        processed_dict.pop(key, None)
    
    return processed_dict

# --- 优化：核心加载逻辑（带清理功能） ---
def load_lora(model, lora_1, lora_2, lora_scale1, lora_scale2):
    # 1. 获取完整的 LoRA 路径
    lora_path_1 = folder_paths.get_full_path("loras", lora_1) if lora_1 != "none" else None
    lora_path_2 = folder_paths.get_full_path("loras", lora_2) if lora_2 != "none" else None

    # 2. 【显存清理】获取并卸载所有已存在的旧适配器
    try:
        all_adapters_info = model.get_list_adapters()
        if all_adapters_info:
            # 提取 transformer 和 transformer_2 中所有的适配器名称
            to_delete = set()
            for key in ['transformer', 'transformer_2']:
                if key in all_adapters_info:
                    to_delete.update(all_adapters_info[key])
            
            for adapter in to_delete:
                print(f"[TwinFlow] 正在清理旧适配器: {adapter}")
                try:
                    model.delete_adapters(adapter)
                except:
                    pass
            # 强制清理碎片
            gc_cleanup() 
    except Exception as e:
        print(f"[TwinFlow] 清理适配器时出现非预期错误: {e}")

    # 3. 加载第一个 LoRA (通常映射到 transformer_2)
    if lora_path_1 is not None:
        adapter_name_1 = sanitize_adapter_name(lora_path_1)
        print(f"[TwinFlow] 正在加载第一个 LoRA: {adapter_name_1}")
        try:
            try:
                model.load_lora_weights(lora_path_1, adapter_name=adapter_name_1, **{"load_into_transformer_2": True})
            except Exception:
                # 备选方案：手动解析
                sd = torch.load(lora_path_1, map_location="cpu") if not lora_path_1.endswith(".safetensors") else load_file(lora_path_1)
                model.load_lora_weights(preprocess_lora_state_dict(sd), adapter_name=adapter_name_1, **{"load_into_transformer_2": True})
            
            model.set_adapters([adapter_name_1], adapter_weights=lora_scale1)
        except Exception as e:
            print(f"[TwinFlow] 加载 LoRA 1 失败: {lora_path_1}")
            traceback.print_exc()

    # 4. 加载第二个 LoRA (通常映射到 transformer)
    if lora_path_2 is not None:
        adapter_name_2 = sanitize_adapter_name(lora_path_2)
        print(f"[TwinFlow] 正在加载第二个 LoRA: {adapter_name_2}")
        try:
            try:
                model.load_lora_weights(lora_path_2, adapter_name=adapter_name_2, **{"load_into_transformer_2": False})
            except Exception:
                sd = torch.load(lora_path_2, map_location="cpu") if not lora_path_2.endswith(".safetensors") else load_file(lora_path_2)
                model.load_lora_weights(preprocess_lora_state_dict(sd), adapter_name=adapter_name_2, **{"load_into_transformer_2": False})
            
            model.set_adapters([adapter_name_2], adapter_weights=lora_scale2)
        except Exception as e:
            print(f"[TwinFlow] 加载 LoRA 2 失败: {lora_path_2}")
            traceback.print_exc()

    return model
