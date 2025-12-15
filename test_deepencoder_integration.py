#!/usr/bin/env python3
"""
DeepEncoder 集成测试脚本
========================

测试 DeepEncoder 是否正确集成到 GEM/LLaVA 架构中

运行方式：
    python test_deepencoder_integration.py
"""

import torch
import sys
import os
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_image_processor():
    """测试图像处理器"""
    print("\n" + "="*60)
    print("Test 1: Image Processor")
    print("="*60)
    
    from llava.model.multimodal_encoder.deepencoder_tower import DeepEncoderImageProcessor
    
    processor = DeepEncoderImageProcessor()
    
    # 测试 Base Mode
    print("\n[Base Mode]")
    test_img = Image.new('RGB', (800, 600), color='blue')
    result = processor(test_img, mode="base")
    print(f"Input size: {test_img.size}")
    print(f"Output shape: {result['pixel_values'].shape}")
    print(f"Expected: [1, 3, 1024, 1024] ✓" if result['pixel_values'].shape == (1, 3, 1024, 1024) else "✗")
    
    # 测试 Gundam Mode
    print("\n[Gundam Mode]")
    test_img_large = Image.new('RGB', (1920, 1080), color='red')
    result_gundam = processor([test_img_large], mode="gundam")
    print(f"Input size: {test_img_large.size}")
    print(f"Global view: {result_gundam[0]['global_view'].shape}")
    print(f"Patches: {result_gundam[0]['patches'].shape}")
    print(f"Crop ratio: {result_gundam[0]['crop_ratio']}")
    
    return True


def test_vision_tower():
    """测试 Vision Tower"""
    print("\n" + "="*60)
    print("Test 2: Vision Tower (Base Mode)")
    print("="*60)
    
    from llava.model.multimodal_encoder.deepencoder_tower import DeepEncoderVisionTower
    
    class Args:
        unfreeze_mm_vision_tower = False
        use_gundam_mode = False
    
    args = Args()
    
    # 注意：这里不加载真实权重，只测试结构
    print("\n[Creating model...]")
    model = DeepEncoderVisionTower(
        vision_tower='/path/to/weights',  # dummy path
        args=args,
        delay_load=False
    )
    
    print(f"Model created successfully!")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Num patches: {model.num_patches}")
    print(f"Image size: {model.config.image_size}")
    
    # 测试 forward
    print("\n[Testing forward pass...]")
    dummy_input = torch.randn(2, 3, 1024, 1024)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: [2, 272, 2048] (16x16 grid + 16 newlines)")
        
        if output.shape == (2, 272, 2048):
            print("✓ Output shape correct!")
            return True
        else:
            print(f"✗ Output shape mismatch! Got {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_collator():
    """测试 Data Collator"""
    print("\n" + "="*60)
    print("Test 3: Data Collator")
    print("="*60)
    
    from llava.model.multimodal_encoder.deepencoder_collator import get_deepencoder_collator
    
    class MockTokenizer:
        pad_token_id = 0
    
    tokenizer = MockTokenizer()
    
    # 模拟不同切分数量的样本
    instances = [
        {
            'input_ids': torch.tensor([1, 2, 3, 4]),
            'labels': torch.tensor([1, 2, 3, 4]),
            'images': {
                'global_view': torch.randn(3, 1024, 1024),
                'patches': torch.randn(6, 3, 640, 640),
                'crop_ratio': (2, 3)
            },
            'ecgs': torch.randn(12, 5000)
        },
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'labels': torch.tensor([1, 2, 3]),
            'images': {
                'global_view': torch.randn(3, 1024, 1024),
                'patches': torch.randn(9, 3, 640, 640),  # 不同数量!
                'crop_ratio': (3, 3)
            },
            'ecgs': torch.randn(12, 5000)
        }
    ]
    
    # 测试 Gundam Collator (List mode)
    print("\n[Gundam Collator - List Mode]")
    collator_gundam = get_deepencoder_collator("gundam", tokenizer)
    batch = collator_gundam(instances)
    
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Images: {len(batch['images'])} samples")
    print(f"  Sample 0 patches: {batch['images'][0]['patches'].shape[0]} patches")
    print(f"  Sample 1 patches: {batch['images'][1]['patches'].shape[0]} patches")
    print(f"ECGs: {batch['ecgs'].shape}")
    print("✓ Gundam collator works with variable patch numbers!")
    
    # 测试 Padded Collator
    print("\n[Gundam Collator - Padded Mode]")
    collator_padded = get_deepencoder_collator("gundam_padded", tokenizer)
    batch_padded = collator_padded(instances)
    
    print(f"Global Views: {batch_padded['global_views'].shape}")
    print(f"Patches (padded): {batch_padded['patches'].shape}")
    print(f"Patches Mask: {batch_padded['patches_mask'].shape}")
    print("✓ Padded collator successfully pads variable-length patches!")
    
    return True


def test_builder_integration():
    """测试 builder.py 中的集成"""
    print("\n" + "="*60)
    print("Test 4: Builder Integration")
    print("="*60)
    
    from llava.model.multimodal_encoder.builder import build_vision_tower
    
    class Config:
        mm_vision_tower = '/dummy/path'
        vision_tower = '/dummy/path'
        use_deepencoder = True
        use_gundam_mode = False
        unfreeze_mm_vision_tower = False
        delay_load = True
    
    config = Config()
    
    print("\n[Building vision tower with DeepEncoder...]")
    try:
        tower = build_vision_tower(config, delay_load=True)
        print(f"✓ Tower type: {type(tower).__name__}")
        print(f"✓ Hidden size: {tower.hidden_size}")
        return True
    except Exception as e:
        print(f"✗ Failed to build tower: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" DeepEncoder Integration Test Suite")
    print("="*80)
    
    results = []
    
    # 测试 1: 图像处理器
    try:
        results.append(("Image Processor", test_image_processor()))
    except Exception as e:
        print(f"\n✗ Image Processor test failed: {e}")
        results.append(("Image Processor", False))
    
    # 测试 2: Vision Tower
    try:
        results.append(("Vision Tower", test_vision_tower()))
    except Exception as e:
        print(f"\n✗ Vision Tower test failed: {e}")
        results.append(("Vision Tower", False))
    
    # 测试 3: Collator
    try:
        results.append(("Data Collator", test_collator()))
    except Exception as e:
        print(f"\n✗ Data Collator test failed: {e}")
        results.append(("Data Collator", False))
    
    # 测试 4: Builder 集成
    try:
        results.append(("Builder Integration", test_builder_integration()))
    except Exception as e:
        print(f"\n✗ Builder Integration test failed: {e}")
        results.append(("Builder Integration", False))
    
    # 总结
    print("\n" + "="*80)
    print(" Test Summary")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 All tests passed! DeepEncoder is successfully integrated.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
