"""
DeepEncoder Vision Tower for GEM/LLaVA Architecture
====================================================

集成 DeepSeek-OCR 的双编码器（SAM + CLIP）到 GEM/LLaVA 架构中。

支持模式：
1. Base Mode: 只使用全局视图，1024x1024，无动态切分（推荐用于初期训练）
2. Gundam Mode: 动态切分 + 全局视图（更高性能，但需要特殊的 collate_fn）

关键特性：
- SAM (Segment Anything Model) 提供低层空间特征
- CLIP-L 提供高层语义特征
- 双流特征拼接 (2048维)
- 支持 image_newline 保持 2D 空间结构
- 兼容 LLaVA 的标准接口

作者: AI Assistant
日期: 2025-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Optional, Tuple, List, Union
from PIL import Image, ImageOps
import math

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


# ==================== 辅助函数: 动态切分 (用于 Gundam Mode) ====================

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    找到最接近的宽高比用于动态切分

    Args:
        aspect_ratio: 原始图片宽高比
        target_ratios: 候选宽高比列表 [(w1, h1), (w2, h2), ...]
        width: 原始图片宽度
        height: 原始图片高度
        image_size: 单个patch的尺寸

    Returns:
        tuple: 最佳宽高比 (width_ratio, height_ratio)
    """
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
            # 如果宽高比相同，选择面积更接近的
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    """
    DeepSeek-OCR 的动态切分策略

    根据图片的宽高比，自适应地将图片切分成 N 个小块（N ∈ [min_num, max_num]）

    Args:
        image (PIL.Image): 输入图片
        min_num (int): 最小切分块数
        max_num (int): 最大切分块数
        image_size (int): 单个切分块的尺寸（默认640）
        use_thumbnail (bool): 是否额外添加缩略图

    Returns:
        tuple: (processed_images, target_aspect_ratio)
            - processed_images: 切分后的图片列表
            - target_aspect_ratio: (width_ratio, height_ratio)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 生成所有可能的宽高比组合
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最佳宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # 计算目标尺寸
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 缩放并切分
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

    # 可选：添加缩略图
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


# ==================== 图像处理器 ====================

class DeepEncoderImageProcessor:
    """
    DeepEncoder 图像预处理器

    实现两种模式：
    1. Base Mode: Resize & Pad 到 1024x1024（保持比例）
    2. Gundam Mode: 动态切分 + 全局视图

    Args:
        base_size (int): 全局视图的尺寸（默认1024）
        patch_size (int): 切分块的尺寸（默认640，用于Gundam模式）
        mean (tuple): 归一化均值
        std (tuple): 归一化标准差
    """

    def __init__(
        self,
        base_size: int = 1024,
        patch_size: int = 640,
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
    ):
        self.base_size = base_size
        self.patch_size = patch_size
        self.crop_size = {"height": base_size, "width": base_size}
        self.image_mean = mean
        self.image_std = std

    def preprocess(self, images, return_tensors="pt", mode="base"):
        """
        预处理图片

        Args:
            images: PIL.Image 或 list of PIL.Image
            return_tensors: 返回类型 ("pt" for PyTorch tensors)
            mode: "base" 或 "gundam"

        Returns:
            dict: {"pixel_values": tensor, "crop_info": ...}
        """
        if not isinstance(images, list):
            images = [images]

        if mode == "base":
            return self._preprocess_base_mode(images, return_tensors)
        elif mode == "gundam":
            return self._preprocess_gundam_mode(images, return_tensors)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _preprocess_base_mode(self, images, return_tensors="pt"):
        """Base Mode: 只处理全局视图"""
        from torchvision.transforms.functional import to_tensor, normalize

        processed = []
        for img in images:
            if not isinstance(img, Image.Image):
                if isinstance(img, torch.Tensor):
                    processed.append(img)
                continue

            img = img.convert("RGB")
            fill_color = tuple(int(x * 255) for x in self.image_mean)

            # 保持比例缩放并填充
            padded_img = ImageOps.pad(img, (self.base_size, self.base_size), color=fill_color)

            # 转 Tensor 并归一化
            tensor = to_tensor(padded_img)
            tensor = normalize(tensor, mean=self.image_mean, std=self.image_std)
            processed.append(tensor)

        if return_tensors == "pt":
            return {"pixel_values": torch.stack(processed)}
        return processed

    def _preprocess_gundam_mode(self, images, return_tensors="pt"):
        """
        Gundam Mode: 动态切分 + 全局视图

        注意：此模式下，不同图片可能产生不同数量的patches！
        需要使用自定义的 collate_fn
        """
        from torchvision.transforms.functional import to_tensor, normalize

        batch_results = []

        for img in images:
            if not isinstance(img, Image.Image):
                raise ValueError("Gundam mode requires PIL.Image input")

            img = img.convert("RGB")
            fill_color = tuple(int(x * 255) for x in self.image_mean)

            # 1. 处理全局视图
            global_img = ImageOps.pad(img, (self.base_size, self.base_size), color=fill_color)
            global_tensor = normalize(to_tensor(global_img), mean=self.image_mean, std=self.image_std)

            # 2. 动态切分
            if img.size[0] <= self.patch_size and img.size[1] <= self.patch_size:
                # 小图片，无需切分
                patches_tensor = torch.zeros(1, 3, self.patch_size, self.patch_size)  # dummy
                crop_ratio = (1, 1)
            else:
                # 大图片，动态切分
                patches_list, crop_ratio = dynamic_preprocess(img, image_size=self.patch_size)
                patches_tensor = torch.stack([
                    normalize(to_tensor(p), mean=self.image_mean, std=self.image_std)
                    for p in patches_list
                ])

            batch_results.append({
                "global_view": global_tensor,
                "patches": patches_tensor,
                "crop_ratio": crop_ratio
            })

        if return_tensors == "pt":
            return batch_results
        return batch_results

    def __call__(self, images, return_tensors="pt", mode="base"):
        return self.preprocess(images, return_tensors, mode)


# ==================== Vision Tower ====================

class DeepEncoderVisionTower(nn.Module):
    """
    DeepEncoder Vision Tower - DeepSeek-OCR 双编码器

    架构：
        Image -> SAM (1024D) + CLIP (1024D) -> Concat (2048D) -> Projector

    特性：
        - 支持 Base Mode 和 Gundam Mode
        - 集成 image_newline 保持 2D 结构
        - 兼容 LLaVA/GEM 的标准接口

    Args:
        vision_tower (str): 预训练权重路径
        args: 配置对象（需要包含 delay_load, use_gundam_mode 等）
        delay_load (bool): 是否延迟加载权重
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower

        # ============ 核心配置 ============
        # 锁定为 DeepSeek-OCR 的训练尺寸
        self.base_size = 1024  # 全局视图尺寸
        self.patch_size_spatial = 16  # SAM的patch size
        self.patch_size_local = 640  # Gundam模式的切分块尺寸

        # 模式选择
        self.use_gundam_mode = getattr(args, 'use_gundam_mode', False)

        # 输出特征维度: SAM(1024) + CLIP(1024) = 2048
        self._hidden_size = 2048

        # ============ 模型组件 ============
        self.sam_model: Optional[ImageEncoderViT] = None
        self.vision_model: Optional[VitModel] = None

        # Projector: 2048 -> LLM hidden size (例如4096)
        # 注意：这里只做特征融合，不做降维（降维由外部的 mm_projector 完成）
        self.projector = None

        # 空间分隔符 (关键！让模型理解2D结构)
        # 在每一行的末尾添加 newline token
        self.image_newline = nn.Parameter(torch.empty(self._hidden_size))
        nn.init.normal_(self.image_newline, std=0.02)

        # 视图分隔符（用于 Gundam 模式，分隔 patches 和 global view）
        if self.use_gundam_mode:
            self.view_separator = nn.Parameter(torch.empty(self._hidden_size))
            nn.init.normal_(self.view_separator, std=0.02)

        # ============ 加载模型 ============
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        """加载 SAM + CLIP 模型和预训练权重"""
        if self.is_loaded:
            print("[DeepEncoder] Already loaded, skipping...")
            return

        print(f"[DeepEncoder] Loading components from {self.vision_tower_name}...")

        # 1. 构建模型结构
        self.sam_model = build_sam_vit_b(checkpoint=None)
        self.vision_model = build_clip_l()

        # 2. 构建 Projector (2048 -> 2048, 简单的线性层)
        self.projector = MlpProjector(adict(
            projector_type="linear",
            input_dim=2048,
            n_embed=self._hidden_size
        ))

        # 3. 加载预训练权重
        if self.vision_tower_name and os.path.exists(self.vision_tower_name):
            self._load_pretrained_weights(self.vision_tower_name)
        else:
            print(f"[Warning] Path {self.vision_tower_name} not found, using random init")

        # 4. 冻结参数（可选，根据训练策略调整）
        freeze_vision_tower = True  # 通常冻结预训练的视觉编码器
        if freeze_vision_tower:
            self.sam_model.requires_grad_(False)
            self.vision_model.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.image_newline.requires_grad_(False)
            if hasattr(self, 'view_separator'):
                self.view_separator.requires_grad_(False)
            print("[DeepEncoder] Vision tower frozen (only mm_projector will be trained)")
        else:
            print("[DeepEncoder] Vision tower unfrozen (full model training)")

        self.is_loaded = True
        print(f"[DeepEncoder] Loaded successfully! Mode: {'Gundam' if self.use_gundam_mode else 'Base'}")

    def _load_pretrained_weights(self, model_path: str):
        """
        加载预训练权重

        支持的格式：
        1. 单个 .bin / .pt 文件
        2. 文件夹（包含多个 .bin / .safetensors 文件）
        """
        print(f"[DeepEncoder] Loading weights from {model_path}...")
        state_dict = self._load_state_dict(model_path)

        # 加载 SAM 权重
        sam_keys = {k.replace('sam_model.', ''): v for k, v in state_dict.items() if 'sam_model' in k}
        if sam_keys:
            missing, unexpected = self.sam_model.load_state_dict(sam_keys, strict=False)
            print(f"[DeepEncoder SAM] Loaded {len(sam_keys)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # 加载 CLIP 权重
        clip_keys = {k.replace('vision_model.', ''): v for k, v in state_dict.items() if 'vision_model' in k}
        if clip_keys:
            missing, unexpected = self.vision_model.load_state_dict(clip_keys, strict=False)
            print(f"[DeepEncoder CLIP] Loaded {len(clip_keys)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # 加载 Projector 权重
        proj_keys = {k.replace('projector.', ''): v for k, v in state_dict.items() if 'projector' in k}
        if proj_keys:
            missing, unexpected = self.projector.load_state_dict(proj_keys, strict=False)
            print(f"[DeepEncoder Projector] Loaded {len(proj_keys)} keys")

        # 加载 newline 权重
        if 'image_newline' in state_dict:
            self.image_newline.data = state_dict['image_newline']
            print("[DeepEncoder] Loaded image_newline")

    def _load_state_dict(self, model_path: str) -> dict:
        """加载 state dict（支持单文件和文件夹）"""
        if os.path.isfile(model_path):
            return torch.load(model_path, map_location='cpu')

        # 文件夹模式：合并所有权重文件
        state_dict = {}
        for f in os.listdir(model_path):
            file_path = os.path.join(model_path, f)
            if f.endswith('.bin') or f.endswith('.pt'):
                state_dict.update(torch.load(file_path, map_location='cpu'))
            elif f.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict.update(load_file(file_path))
                except ImportError:
                    print("[Warning] safetensors not installed, skipping .safetensors files")
        return state_dict

    # ==================== 核心编码逻辑 ====================

    def _encode_single_view(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码单个视图（全局或局部patch）

        Args:
            image: [B, 3, H, W] (通常 H=W=1024 或 640)

        Returns:
            features: [B, N_tokens, 2048] 其中 N_tokens = H/16 * W/16
        """
        # 1. SAM 提取低层特征
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            sam_features = self.sam_model(image)  # [B, 1024, H/16, W/16]

            # 2. CLIP 提取高层特征 (利用 SAM 特征)
            clip_features = self.vision_model(image, sam_features)  # [B, N_tokens+1, 1024]

            # 3. 拼接特征
            clip_content = clip_features[:, 1:]  # 去掉 CLS token: [B, N_tokens, 1024]
            sam_content = sam_features.flatten(2).permute(0, 2, 1)  # [B, N_tokens, 1024]

            # 确保长度一致
            L = min(clip_content.shape[1], sam_content.shape[1])
            combined = torch.cat([
                clip_content[:, :L],
                sam_content[:, :L]
            ], dim=-1)  # [B, N_tokens, 2048]

            # 4. 通过 Projector
            features = self.projector(combined)  # [B, N_tokens, 2048]

        return features

    def _add_newlines(self, features: torch.Tensor) -> torch.Tensor:
        """
        添加 image_newline 以保持 2D 空间结构

        Args:
            features: [B, N_tokens, D] 其中 N_tokens = H*W

        Returns:
            features_with_newline: [B, N_tokens + H, D]  (每行末尾添加一个 newline token)
        """
        B, N, D = features.shape
        grid_size = int(math.sqrt(N))  # 假设是正方形 grid

        if grid_size * grid_size != N:
            print(f"[Warning] Feature shape {features.shape} is not a perfect square, skipping newlines")
            return features

        # Reshape 为 2D grid: [B, H, W, D]
        feat_grid = features.view(B, grid_size, grid_size, D)

        # 构造 newline 列: [B, H, 1, D]
        newline = self.image_newline.view(1, 1, 1, -1).expand(
            B, grid_size, 1, -1
        ).to(features.device, features.dtype)

        # 横向拼接: [B, H, W+1, D]
        feat_with_newline = torch.cat([feat_grid, newline], dim=2)

        # 展平: [B, H*(W+1), D]
        return feat_with_newline.view(B, -1, D)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (Base Mode)

        Args:
            images: [B, 3, 1024, 1024]

        Returns:
            features: [B, N_tokens, 2048] 其中 N_tokens ≈ 273 (16x16 grid + 16 newlines + 1 separator)
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # 设备对齐
        device = images.device
        dtype = images.dtype
        self.sam_model = self.sam_model.to(device)
        self.vision_model = self.vision_model.to(device)

        # 编码
        features = self._encode_single_view(images)  # [B, 256, 2048]

        # 添加 newlines
        features = self._add_newlines(features)  # [B, 272, 2048] (16*16 + 16)

        return features

    def forward_gundam(self, images_data: List[dict]) -> List[torch.Tensor]:
        """
        前向传播 (Gundam Mode)

        Args:
            images_data: List of dict, 每个dict包含:
                - global_view: [3, 1024, 1024]
                - patches: [N_patches, 3, 640, 640]
                - crop_ratio: (width_ratio, height_ratio)

        Returns:
            List of tensors: 每个 tensor 的形状为 [N_total_tokens, 2048]
                其中 N_total_tokens = patches_tokens + global_tokens + 1 (separator)
        """
        device = self.device
        dtype = self.dtype

        results = []
        for data in images_data:
            global_view = data["global_view"].unsqueeze(0).to(device, dtype)
            patches = data["patches"].to(device, dtype)
            crop_ratio = data["crop_ratio"]

            # 1. 编码全局视图
            global_features = self._encode_single_view(global_view)  # [1, 256, 2048]
            global_features = self._add_newlines(global_features)  # [1, 272, 2048]
            global_features = global_features.squeeze(0)  # [272, 2048]

            # 2. 编码局部 patches（如果有）
            if patches.shape[0] > 1:  # 非 dummy patches
                patches_features = self._encode_single_view(patches)  # [N, 100, 2048]

                # 重排 patches 为 2D grid
                width_ratio, height_ratio = crop_ratio
                N, num_tokens, D = patches_features.shape
                grid_size = int(math.sqrt(num_tokens))  # 10 for 640/64

                # [N, 100, 2048] -> [height_ratio, width_ratio, 10, 10, 2048]
                patches_features = patches_features.view(
                    height_ratio, width_ratio, grid_size, grid_size, D
                )

                # -> [height_ratio*10, width_ratio*10, 2048]
                patches_features = patches_features.permute(0, 2, 1, 3, 4).contiguous()
                patches_features = patches_features.view(
                    height_ratio * grid_size,
                    width_ratio * grid_size,
                    D
                )

                # 添加 newlines
                newline = self.image_newline.view(1, 1, -1).expand(
                    height_ratio * grid_size, 1, -1
                ).to(device, dtype)
                patches_features = torch.cat([patches_features, newline], dim=1)
                patches_features = patches_features.view(-1, D)  # [N_tokens, 2048]

                # 拼接: patches + global + separator
                combined = torch.cat([
                    patches_features,
                    global_features,
                    self.view_separator.unsqueeze(0).to(device, dtype)
                ], dim=0)
            else:
                # 没有 patches，只用全局视图
                combined = torch.cat([
                    global_features,
                    self.view_separator.unsqueeze(0).to(device, dtype)
                ], dim=0)

            results.append(combined)

        return results

    # ==================== 属性接口（兼容 LLaVA） ====================

    @property
    def dummy_feature(self):
        return torch.zeros(1, self._hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.sam_model is not None:
            return self.sam_model.patch_embed.proj.weight.dtype
        return torch.float32

    @property
    def device(self):
        if self.sam_model is not None:
            return self.sam_model.patch_embed.proj.weight.device
        return torch.device('cpu')

    @property
    def config(self):
        """兼容性属性，LLaVA 某些代码会读取"""
        class Config:
            image_size = 1024
            hidden_size = 2048
            num_patches = 256
            num_patches_per_side = 16
        return Config()

    @property
    def hidden_size(self):
        """输出特征维度"""
        return self._hidden_size

    @property
    def num_patches(self):
        """有效 patch 数量（不含 newline）"""
        return 256  # 16x16 grid

    @property
    def num_patches_per_side(self):
        """每边的 patch 数量"""
        return 16

    @property
    def image_processor(self):
        """返回对应的图像处理器"""
        if not hasattr(self, '_image_processor'):
            self._image_processor = DeepEncoderImageProcessor(
                base_size=self.base_size,
                patch_size=self.patch_size_local
            )
        return self._image_processor


# ==================== 工具函数 ====================

def build_deepencoder_tower(vision_tower_path: str, args, **kwargs):
    """
    便捷函数：构建 DeepEncoder Vision Tower

    Args:
        vision_tower_path: 预训练权重路径
        args: 配置对象
        **kwargs: 额外参数（如 delay_load）

    Returns:
        DeepEncoderVisionTower 实例
    """
    return DeepEncoderVisionTower(vision_tower_path, args, **kwargs)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing DeepEncoder Vision Tower")
    print("=" * 60)

    # 模拟配置
    class Args:
        unfreeze_mm_vision_tower = False
        use_gundam_mode = True  # 测试 Base Mode

    args = Args()

    # 1. 测试图像处理器
    print("\n[Test 1] Image Processor (Base Mode)")
    processor = DeepEncoderImageProcessor()
    dummy_img = Image.new('RGB', (2200, 1700), color='red')  # 模拟心电图（长条形）
    processed = processor(dummy_img, mode="base")
    print(f"Input: {dummy_img.size} -> Output: {processed['pixel_values'].shape}")

    # 2. 测试 Vision Tower
    print("\n[Test 2] Vision Tower Forward Pass")
    model = DeepEncoderVisionTower(vision_tower='dummy_path', args=args, delay_load=False)

    # 模拟输入
    dummy_tensor = torch.randn(2, 3, 1024, 1024)

    with torch.no_grad():
        features = model(dummy_tensor)

    print(f"Input: {dummy_tensor.shape} -> Output: {features.shape}")
    print(f"Expected: [2, 272, 2048] (16*16 grid + 16 newlines)")

    # 3. 测试 Gundam Mode（可选）
    if args.use_gundam_mode:
        print("\n[Test 3] Gundam Mode")
        processor_gundam = DeepEncoderImageProcessor()
        dummy_img_large = Image.new('RGB', (2200, 1700), color='blue')
        processed_gundam = processor_gundam([dummy_img_large], mode="gundam")

        print(f"Input: {dummy_img_large.size}")
        print(f"Global view: {processed_gundam[0]['global_view'].shape}")
        print(f"Patches: {processed_gundam[0]['patches'].shape}")
        print(f"Crop ratio: {processed_gundam[0]['crop_ratio']}")
        model = DeepEncoderVisionTower(vision_tower='dummy_path', args=args, delay_load=False)
        with torch.no_grad():
            features_gundam = model.forward_gundam(processed_gundam)
            print(f"Output: {features_gundam.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
