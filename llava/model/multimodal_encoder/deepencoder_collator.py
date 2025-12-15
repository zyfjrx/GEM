"""
Custom Data Collator for DeepEncoder Gundam Mode
=================================================

解决动态切分导致的 Tensor 形状不一致问题

关键思路：
1. 不在 batch 维度上 stack 动态形状的 patches
2. 将每个样本作为独立单元处理
3. 在模型 forward 时逐样本编码

作者: AI Assistant
日期: 2025-01
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import warnings


@dataclass
class DeepEncoderCollator:
    """
    DeepEncoder 的自定义 Data Collator
    
    支持两种模式：
    1. Base Mode: 标准的 batch collate（所有图片都是 1024x1024）
    2. Gundam Mode: 动态形状 collate（patches 数量不同）
    
    Args:
        mode (str): "base" 或 "gundam"
        tokenizer: HuggingFace tokenizer
        max_length (int): 最大序列长度
    """
    
    mode: str = "base"
    tokenizer: Any = None
    max_length: int = 2048
    
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate 函数
        
        Args:
            instances: List of dict, 每个 dict 包含:
                - input_ids: token ids
                - labels: 标签
                - images: 图像 tensor 或 dict (Gundam mode)
                - ecgs: 心电图数据
                - image_sizes: 图像尺寸（可选）
        
        Returns:
            batch: Dict of tensors
        """
        if self.mode == "base":
            return self._collate_base(instances)
        elif self.mode == "gundam":
            return self._collate_gundam(instances)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _collate_base(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Base Mode: 标准 collate
        
        所有图片都是相同尺寸，可以直接 stack
        """
        batch = {}
        
        # 1. 处理文本输入
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance.get('labels', None) for instance in instances]
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        if labels[0] is not None:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100  # IGNORE_INDEX
            )
            batch['labels'] = labels
        
        batch['input_ids'] = input_ids
        batch['attention_mask'] = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 2. 处理图像
        if 'images' in instances[0] and instances[0]['images'] is not None:
            images = [instance['images'] for instance in instances]
            
            # 检查是否可以 stack（shape 相同）
            if all(img.shape == images[0].shape for img in images):
                batch['images'] = torch.stack(images)
            else:
                # 不同尺寸，保持 list 形式
                batch['images'] = images
        
        # 3. 处理心电图
        if 'ecgs' in instances[0] and instances[0]['ecgs'] is not None:
            ecgs = [instance['ecgs'] for instance in instances]
            batch['ecgs'] = torch.stack(ecgs)
        
        # 4. 处理图像尺寸信息（用于 anyres）
        if 'image_sizes' in instances[0]:
            batch['image_sizes'] = [instance['image_sizes'] for instance in instances]
        
        return batch
    
    def _collate_gundam(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Gundam Mode: 动态形状 collate
        
        关键：不 stack patches，保持 list 形式
        """
        batch = {}
        
        # 1. 文本输入（与 base 相同）
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance.get('labels', None) for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        if labels[0] is not None:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100
            )
            batch['labels'] = labels
        
        batch['input_ids'] = input_ids
        batch['attention_mask'] = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 2. 处理图像（Gundam 模式）
        if 'images' in instances[0] and instances[0]['images'] is not None:
            images_data = []
            for instance in instances:
                img_data = instance['images']
                if isinstance(img_data, dict):
                    # Gundam 模式的 dict 格式
                    images_data.append(img_data)
                else:
                    # 兼容 base 模式
                    warnings.warn("Gundam collator received base mode images, converting...")
                    images_data.append({
                        'global_view': img_data,
                        'patches': torch.zeros(1, 3, 640, 640),  # dummy
                        'crop_ratio': (1, 1)
                    })
            
            # 保持 list 形式，不 stack！
            batch['images'] = images_data
            batch['is_gundam_mode'] = True
        
        # 3. 心电图（与 base 相同）
        if 'ecgs' in instances[0] and instances[0]['ecgs'] is not None:
            ecgs = [instance['ecgs'] for instance in instances]
            batch['ecgs'] = torch.stack(ecgs)
        
        return batch


@dataclass
class DeepEncoderCollatorWithPadding:
    """
    带智能 Padding 的 Collator（用于 Gundam Mode）
    
    策略：
    1. 找到 batch 中最大的 patches 数量 max_N
    2. 对所有样本的 patches 进行 padding 到 max_N
    3. 使用 mask 标记 padding 部分
    
    优点：可以 stack 成规整的 tensor
    缺点：有一些计算浪费（padding 部分）
    """
    
    tokenizer: Any = None
    max_length: int = 2048
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # 1. 文本部分（标准处理）
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance.get('labels', None) for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        if labels[0] is not None:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100
            )
            batch['labels'] = labels
        
        batch['input_ids'] = input_ids
        batch['attention_mask'] = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 2. 图像部分（Gundam 模式 + Padding）
        if 'images' in instances[0]:
            images_data = [instance['images'] for instance in instances]
            
            # 提取全局视图和patches
            global_views = []
            patches_list = []
            crop_ratios = []
            
            for img_data in images_data:
                if isinstance(img_data, dict):
                    global_views.append(img_data['global_view'])
                    patches_list.append(img_data['patches'])
                    crop_ratios.append(img_data['crop_ratio'])
                else:
                    # 兼容 base 模式
                    global_views.append(img_data)
                    patches_list.append(torch.zeros(1, 3, 640, 640))
                    crop_ratios.append((1, 1))
            
            # Stack 全局视图（形状相同）
            batch['global_views'] = torch.stack(global_views)
            
            # Padding patches
            max_patches = max(p.shape[0] for p in patches_list)
            padded_patches = []
            patches_mask = []
            
            for patches in patches_list:
                N = patches.shape[0]
                if N < max_patches:
                    # Padding
                    pad = torch.zeros(max_patches - N, *patches.shape[1:])
                    padded = torch.cat([patches, pad], dim=0)
                else:
                    padded = patches
                
                # Mask: 1 表示真实数据，0 表示 padding
                mask = torch.cat([
                    torch.ones(N),
                    torch.zeros(max_patches - N)
                ])
                
                padded_patches.append(padded)
                patches_mask.append(mask)
            
            batch['patches'] = torch.stack(padded_patches)  # [B, max_N, C, H, W]
            batch['patches_mask'] = torch.stack(patches_mask)  # [B, max_N]
            batch['crop_ratios'] = crop_ratios
        
        # 3. 心电图
        if 'ecgs' in instances[0] and instances[0]['ecgs'] is not None:
            ecgs = [instance['ecgs'] for instance in instances]
            batch['ecgs'] = torch.stack(ecgs)
        
        return batch


# ==================== 工具函数 ====================

def get_deepencoder_collator(mode: str = "base", tokenizer=None, **kwargs):
    """
    便捷函数：获取对应模式的 collator
    
    Args:
        mode: "base", "gundam", or "gundam_padded"
        tokenizer: HuggingFace tokenizer
        **kwargs: 额外参数
    
    Returns:
        Collator 实例
    """
    if mode == "base":
        return DeepEncoderCollator(mode="base", tokenizer=tokenizer, **kwargs)
    elif mode == "gundam":
        return DeepEncoderCollator(mode="gundam", tokenizer=tokenizer, **kwargs)
    elif mode == "gundam_padded":
        return DeepEncoderCollatorWithPadding(tokenizer=tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("Testing DeepEncoder Collators...")
    
    # 模拟 tokenizer
    class MockTokenizer:
        pad_token_id = 0
    
    tokenizer = MockTokenizer()
    
    # 模拟数据（Gundam 模式）
    instances = [
        {
            'input_ids': torch.tensor([1, 2, 3, 4]),
            'labels': torch.tensor([1, 2, 3, 4]),
            'images': {
                'global_view': torch.randn(3, 1024, 1024),
                'patches': torch.randn(6, 3, 640, 640),  # 6 patches
                'crop_ratio': (2, 3)
            },
            'ecgs': torch.randn(12, 5000)
        },
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'labels': torch.tensor([1, 2, 3]),
            'images': {
                'global_view': torch.randn(3, 1024, 1024),
                'patches': torch.randn(9, 3, 640, 640),  # 9 patches (不同!)
                'crop_ratio': (3, 3)
            },
            'ecgs': torch.randn(12, 5000)
        }
    ]
    
    # 测试 Gundam Collator
    print("\n[Test 1] Gundam Collator (List Mode)")
    collator_gundam = get_deepencoder_collator("gundam", tokenizer)
    batch = collator_gundam(instances)
    
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Images: {len(batch['images'])} samples")
    print(f"  Sample 0 patches: {batch['images'][0]['patches'].shape}")
    print(f"  Sample 1 patches: {batch['images'][1]['patches'].shape}")
    print(f"ECGs: {batch['ecgs'].shape}")
    
    # 测试 Padded Collator
    print("\n[Test 2] Gundam Collator (Padded Mode)")
    collator_padded = get_deepencoder_collator("gundam_padded", tokenizer)
    batch_padded = collator_padded(instances)
    
    print(f"Global Views: {batch_padded['global_views'].shape}")
    print(f"Patches (padded): {batch_padded['patches'].shape}")
    print(f"Patches Mask: {batch_padded['patches_mask'].shape}")
    print(f"  Sample 0 mask: {batch_padded['patches_mask'][0]}")
    print(f"  Sample 1 mask: {batch_padded['patches_mask'][1]}")
    
    print("\nAll tests passed!")

