"""
DeepEncoder Vision Tower for GEM
从DeepSeek-OCR模型中提取并使用SAM ViT-B + CLIP-L双编码器

两种模式:
1. 简化模式 (默认): 只提取全局特征，不做动态切分
   - 适用于: 标准尺寸ECG图像 (如336x336)
   - 不需要 dynamic_preprocess

2. 完整模式: 复用DeepSeek-OCR的动态切分，提取局部+全局特征
   - 适用于: 高分辨率图像（需要捕获细节）
   - 需要 dynamic_preprocess
   
用法:
    # 简化模式（推荐用于ECG）
    tower = DeepEncoderVisionTower(
        vision_tower="/path/to/deepseek-ocr-model",
        args=args,
        use_dynamic_preprocess=False  # 默认
    )
    
    # 完整模式
    tower = DeepEncoderVisionTower(
        vision_tower="/path/to/deepseek-ocr-model", 
        args=args,
        use_dynamic_preprocess=True
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Optional, Tuple, List
from PIL import Image, ImageOps

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from deepseek_ocr.deepencoder import (
    build_sam_vit_b, 
    build_clip_l, 
    MlpProjector,
    ImageEncoderViT,
    VitModel
)
from addict import Dict as adict


# ==================== 从 DeepSeek-OCR 复用的预处理函数 ====================

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近的宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    """
    动态预处理：将图像切分成多个patches
    
    这是DeepSeek-OCR用于处理高分辨率图像的策略：
    - 根据图像宽高比选择最佳切分方式
    - 返回切分后的patches和切分比例
    
    Args:
        image: PIL Image
        min_num: 最小切分块数
        max_num: 最大切分块数  
        image_size: 每个patch的尺寸
        use_thumbnail: 是否添加缩略图
        
    Returns:
        processed_images: 切分后的图像列表
        target_aspect_ratio: (width_blocks, height_blocks)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        
    assert len(processed_images) == blocks
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images, target_aspect_ratio


# ==================== DeepEncoder Vision Tower ====================

class DeepEncoderVisionTower(nn.Module):
    """
    封装DeepSeek-OCR的双编码器架构 (SAM ViT-B + CLIP-L)
    用于替代GEM中的CLIPVisionTower处理ECG图像
    
    架构:
        输入图像 -> SAM ViT-B (高分辨率特征) 
                 -> CLIP-L (以SAM输出作为patch embedding)
                 -> 特征concat (2048维)
                 
    支持的模式 (参考DeepSeek-OCR):
        - tiny:   base_size=512,  image_size=512,  crop_mode=False, 64 tokens
        - small:  base_size=640,  image_size=640,  crop_mode=False, 100 tokens
        - base:   base_size=1024, image_size=1024, crop_mode=False, 256 tokens (默认)
        - large:  base_size=1280, image_size=1280, crop_mode=False, 400 tokens
        - gundam: base_size=1024, image_size=640,  crop_mode=True,  多分辨率裁剪
                 
    关于 crop_mode (动态预处理):
        - crop_mode=False: 直接处理整张图像，提取全局特征
          输出: [B, num_patches, 2048]
          
        - crop_mode=True: 先切分图像，分别提取局部和全局特征，然后拼接
          输出: [B, (local_patches + global_patches), 2048]
    """
    
    # 预定义的模式配置
    MODE_CONFIGS = {
        "tiny":   {"base_size": 512,  "image_size": 512,  "crop_mode": False},
        "small":  {"base_size": 640,  "image_size": 640,  "crop_mode": False},
        "base":   {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large":  {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640,  "crop_mode": True},
    }
    
    def __init__(
        self, 
        vision_tower, 
        args, 
        delay_load=False, 
        use_dynamic_preprocess=False,  # 保持向后兼容
        mode: str = "base",            # 新增：模式选择
        base_size: int = None,         # 新增：可覆盖base_size
        image_size: int = None,        # 新增：可覆盖image_size  
        crop_mode: bool = None,        # 新增：可覆盖crop_mode
    ):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower  # DeepSeek-OCR模型路径
        
        # 加载模式配置
        mode_config = self.MODE_CONFIGS.get(mode, self.MODE_CONFIGS["base"])
        
        # 支持自定义覆盖
        self.sam_image_size = base_size if base_size is not None else mode_config["base_size"]
        self.local_image_size = image_size if image_size is not None else mode_config["image_size"]
        
        # crop_mode 优先级: 显式参数 > use_dynamic_preprocess (向后兼容) > 模式默认值
        if crop_mode is not None:
            self.crop_mode = crop_mode
        elif use_dynamic_preprocess:
            self.crop_mode = True
        else:
            self.crop_mode = mode_config["crop_mode"]
        
        # 保持向后兼容
        self.use_dynamic_preprocess = self.crop_mode
        
        # CLIP配置 (固定值，由预训练模型决定)
        self.clip_image_size = 224  # CLIP-L 标准输入尺寸
        self.patch_size = 14        # CLIP-L 标准 patch size
        
        self.mode = mode
        
        # 编码器 (延迟初始化)
        self.sam_model: Optional[ImageEncoderViT] = None
        self.vision_model: Optional[VitModel] = None
        
        # 输出维度: SAM(1024) + CLIP(1024) = 2048
        self._hidden_size = 2048
        
        # 选择层和特征 (兼容CLIPVisionTower接口)
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # 图像预处理 (用于完整模式)
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
    
    def load_model(self, device_map=None):
        """加载模型权重"""
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return
        
        # 1. 构建空模型
        self.sam_model = build_sam_vit_b(checkpoint=None)
        self.vision_model = build_clip_l()
        
        # 2. 尝试加载预训练权重
        if self.vision_tower_name and os.path.exists(self.vision_tower_name):
            self._load_pretrained_weights(self.vision_tower_name)
        else:
            print(f"[Warning] No pretrained weights found at {self.vision_tower_name}, using random initialization")
        
        # 3. 冻结编码器
        self.sam_model.requires_grad_(False)
        self.vision_model.requires_grad_(False)
        
        self.is_loaded = True
        print(f"Loaded DeepEncoder vision tower from {self.vision_tower_name}")
        print(f"  - Mode: {self.mode} (base_size={self.sam_image_size}, image_size={self.local_image_size}, crop_mode={self.crop_mode})")
        print(f"  - Processing: {'动态切分 (局部+全局)' if self.crop_mode else '全局特征'}")
    
    def _load_pretrained_weights(self, model_path: str):
        """
        从DeepSeek-OCR完整模型权重中提取deepencoder部分
        
        权重映射:
        - model.sam_model.* -> SAM ViT-B
        - model.vision_model.* -> CLIP-L
        
        支持的格式:
        - 目录: 包含 *.safetensors 或 pytorch_model*.bin
        - 单个文件: .safetensors 或 .bin/.pt
        """
        print(f"Loading DeepEncoder weights from: {model_path}")
        
        # 加载完整模型的state_dict
        state_dict = self._load_state_dict(model_path)
        
        # 提取SAM模型权重
        sam_weights = {}
        for k, v in state_dict.items():
            if 'sam_model.' in k:
                new_key = k.split('sam_model.')[-1]
                sam_weights[new_key] = v
        
        if sam_weights:
            missing, unexpected = self.sam_model.load_state_dict(sam_weights, strict=False)
            print(f"  SAM model: loaded {len(sam_weights)} weights, missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("  [Warning] No SAM weights found in checkpoint")
        
        # 提取CLIP-L模型权重
        clip_weights = {}
        for k, v in state_dict.items():
            if 'vision_model.' in k:
                new_key = k.split('vision_model.')[-1]
                clip_weights[new_key] = v
        
        if clip_weights:
            missing, unexpected = self.vision_model.load_state_dict(clip_weights, strict=False)
            print(f"  CLIP-L model: loaded {len(clip_weights)} weights, missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("  [Warning] No CLIP-L weights found in checkpoint")
    
    def _load_state_dict(self, model_path: str) -> dict:
        """加载state_dict，支持多种格式"""
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            safe_load_file = None
            
        if os.path.isdir(model_path):
            safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin') or f.endswith('.pt')]
            
            state_dict = {}
            
            if safetensor_files and safe_load_file:
                for sf in sorted(safetensor_files):
                    sf_path = os.path.join(model_path, sf)
                    print(f"    Loading {sf}...")
                    shard = safe_load_file(sf_path)
                    state_dict.update(shard)
            elif bin_files:
                for bf in sorted(bin_files):
                    bf_path = os.path.join(model_path, bf)
                    print(f"    Loading {bf}...")
                    shard = torch.load(bf_path, map_location='cpu')
                    state_dict.update(shard)
            else:
                raise FileNotFoundError(f"No weight files found in {model_path}")
            
            return state_dict
        
        elif model_path.endswith('.safetensors') and safe_load_file:
            return safe_load_file(model_path)
        
        elif model_path.endswith('.bin') or model_path.endswith('.pt'):
            return torch.load(model_path, map_location='cpu')
        
        else:
            raise ValueError(f"Unsupported weight file format: {model_path}")
    
    def _encode_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码单张图像 (内部方法)
        
        Args:
            image: [C, H, W] 或 [1, C, H, W]
            
        Returns:
            features: [num_patches, 2048]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        device = image.device
        dtype = image.dtype
        
        # SAM输入 (resize到1024x1024)
        sam_input = F.interpolate(
            image, 
            size=(self.sam_image_size, self.sam_image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # CLIP输入 (resize到224x224)
        clip_input = F.interpolate(
            image,
            size=(self.clip_image_size, self.clip_image_size),
            mode='bilinear', 
            align_corners=False
        )
        
        # SAM编码 -> [1, 1024, H', W']
        sam_features = self.sam_model(sam_input)
        
        # CLIP编码 -> [1, num_patches+1, 1024]
        clip_features = self.vision_model(clip_input, sam_features)
        
        # 特征融合
        clip_patch_features = clip_features[:, 1:]  # 去掉CLS
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)
        
        # 对齐patch数量
        min_patches = min(clip_patch_features.shape[1], sam_flat.shape[1])
        clip_patch_features = clip_patch_features[:, :min_patches]
        sam_flat = sam_flat[:, :min_patches]
        
        # Concat -> [1, num_patches, 2048]
        combined = torch.cat([clip_patch_features, sam_flat], dim=-1)
        
        return combined.squeeze(0)  # [num_patches, 2048]
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像 (标准forward，不带动态切分)
        
        输入: [B, C, H, W]
        输出: [B, num_patches, 2048]
            
        注意: 如果需要使用 crop_mode=True 的动态切分功能，
              请使用 forward_with_dynamic_preprocess() 方法
        
        Args:
            images: 输入图像tensor [B, C, H, W]
            
        Returns:
            features: 编码特征 [B, num_patches, 2048]
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype
        
        # 确保模型在正确设备
        self.sam_model = self.sam_model.to(device=device, dtype=dtype)
        self.vision_model = self.vision_model.to(device=device, dtype=dtype)
        
        # 简化模式: 直接编码整张图像
        all_features = []
        for i in range(batch_size):
            features = self._encode_single_image(images[i])
            all_features.append(features)
        
        # Stack -> [B, num_patches, 2048]
        return torch.stack(all_features, dim=0)
    
    @torch.no_grad()
    def forward_with_dynamic_preprocess(
        self, 
        pil_images: List[Image.Image],
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> List[torch.Tensor]:
        """
        Gundam模式 (crop_mode=True): 带动态切分的编码
        
        复用DeepSeek-OCR的处理策略:
        1. 对大图进行动态切分 (使用 local_image_size=640 作为切分尺寸)
        2. 分别编码局部patches和全局图像 (使用 sam_image_size 作为全局尺寸)
        3. 拼接: [局部特征, 全局特征]
        
        Args:
            pil_images: PIL Image列表
            device: 目标设备
            dtype: 目标数据类型
            
        Returns:
            features_list: 每张图像的特征tensor列表
                          每个tensor形状: [num_total_patches, 2048]
        """
        from torchvision import transforms
        
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
            
        self.sam_model = self.sam_model.to(device=device, dtype=dtype)
        self.vision_model = self.vision_model.to(device=device, dtype=dtype)
        
        # 图像预处理transform
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std)
        ])
        
        features_list = []
        
        for pil_image in pil_images:
            pil_image = pil_image.convert("RGB")
            w, h = pil_image.size
            
            # 判断是否需要切分
            if w <= self.local_image_size and h <= self.local_image_size:
                # 小图: 只需要全局特征
                global_view = ImageOps.pad(
                    pil_image, 
                    (self.sam_image_size, self.sam_image_size),
                    color=tuple(int(x * 255) for x in self.image_mean)
                )
                global_tensor = image_transform(global_view).to(device=device, dtype=dtype)
                
                features = self._encode_single_image(global_tensor)
                features_list.append(features)
                
            else:
                # 大图: 动态切分 + 全局
                patches, crop_ratio = dynamic_preprocess(
                    pil_image, 
                    min_num=2, 
                    max_num=9, 
                    image_size=self.local_image_size
                )
                
                # 编码局部patches
                local_features_list = []
                for patch in patches:
                    patch_tensor = image_transform(patch).to(device=device, dtype=dtype)
                    patch_features = self._encode_single_image(patch_tensor)
                    local_features_list.append(patch_features)
                
                # 拼接局部特征
                local_features = torch.cat(local_features_list, dim=0)  # [total_local_patches, 2048]
                
                # 编码全局图像
                global_view = ImageOps.pad(
                    pil_image,
                    (self.sam_image_size, self.sam_image_size),
                    color=tuple(int(x * 255) for x in self.image_mean)
                )
                global_tensor = image_transform(global_view).to(device=device, dtype=dtype)
                global_features = self._encode_single_image(global_tensor)  # [global_patches, 2048]
                
                # 拼接: [局部, 全局]
                combined_features = torch.cat([local_features, global_features], dim=0)
                features_list.append(combined_features)
        
        return features_list
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self._hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        if self.sam_model is not None:
            return next(self.sam_model.parameters()).dtype
        return torch.float32
    
    @property
    def device(self):
        if self.sam_model is not None:
            return next(self.sam_model.parameters()).device
        return torch.device('cpu')
    
    @property
    def hidden_size(self):
        """输出特征维度: 2048 (SAM 1024 + CLIP 1024)"""
        return self._hidden_size
    
    @property
    def num_patches_per_side(self):
        """每边的patch数量 (基于CLIP配置)"""
        return self.clip_image_size // self.patch_size  # 224 / 14 = 16
    
    @property
    def num_patches(self):
        """总patch数量 (简化模式)"""
        return self.num_patches_per_side ** 2  # 256
    
    @property
    def config(self):
        """返回兼容的配置对象"""
        class _Config:
            def __init__(self, hidden_size, image_size, patch_size):
                self.hidden_size = hidden_size
                self.image_size = image_size
                self.patch_size = patch_size
        return _Config(self._hidden_size, self.clip_image_size, self.patch_size)


class DeepEncoderImageProcessor:
    """
    DeepEncoder的图像预处理器
    兼容CLIPImageProcessor接口
    """
    
    def __init__(
        self,
        size: int = 224,
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
    ):
        self.size = {"height": size, "width": size, "shortest_edge": size}
        self.crop_size = {"height": size, "width": size}
        self.image_mean = mean
        self.image_std = std
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def preprocess(self, images, return_tensors="pt"):
        """预处理图像"""
        if not isinstance(images, list):
            images = [images]
        
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                tensor = self.transform(img)
            elif isinstance(img, torch.Tensor):
                tensor = img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            processed.append(tensor)
        
        if return_tensors == "pt":
            return {"pixel_values": torch.stack(processed)}
        return processed
    
    def __call__(self, images, return_tensors="pt"):
        return self.preprocess(images, return_tensors)
