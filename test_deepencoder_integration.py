#!/usr/bin/env python3
"""
测试DeepEncoder集成是否正常工作
Usage: python test_deepencoder_integration.py --deepseek_ocr_path /path/to/deepseek-ocr-model
"""

import argparse
import torch
import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_deepencoder_tower():
    """测试DeepEncoderVisionTower的基本功能"""
    print("=" * 60)
    print("Testing DeepEncoder Vision Tower")
    print("=" * 60)
    
    from llava.model.multimodal_encoder.deepencoder_tower import DeepEncoderVisionTower
    
    # 创建mock args
    class Args:
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
    
    args = Args()
    
    # 测试不加载预训练权重的情况
    print("\n1. Testing with random initialization (no pretrained weights)...")
    tower = DeepEncoderVisionTower(
        vision_tower=None, 
        args=args, 
        delay_load=True
    )
    tower.load_model()
    
    print(f"   - Hidden size: {tower.hidden_size}")
    print(f"   - Num patches: {tower.num_patches}")
    print(f"   - Num patches per side: {tower.num_patches_per_side}")
    print(f"   - SAM image size: {tower.sam_image_size}")
    print(f"   - CLIP image size: {tower.clip_image_size}")
    
    # 测试forward pass
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 336, 336)  # 标准输入尺寸
    
    with torch.no_grad():
        output = tower(dummy_input)
    
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Expected output: [batch=2, num_patches, hidden_size=2048]")
    
    assert output.shape[-1] == 2048, f"Expected hidden size 2048, got {output.shape[-1]}"
    print("   ✓ Hidden size correct!")
    
    print("\n" + "=" * 60)
    print("DeepEncoder Tower Test PASSED!")
    print("=" * 60)
    
    return tower


def test_with_pretrained_weights(model_path):
    """测试加载DeepSeek-OCR预训练权重"""
    print("\n" + "=" * 60)
    print(f"Testing with pretrained weights from: {model_path}")
    print("=" * 60)
    
    from llava.model.multimodal_encoder.deepencoder_tower import DeepEncoderVisionTower
    
    class Args:
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
    
    args = Args()
    
    tower = DeepEncoderVisionTower(
        vision_tower=model_path,
        args=args,
        delay_load=False
    )
    
    # 测试forward
    dummy_input = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        output = tower(dummy_input)
    
    print(f"\n   - Output shape: {output.shape}")
    print(f"   - Output mean: {output.mean().item():.6f}")
    print(f"   - Output std: {output.std().item():.6f}")
    
    print("\n" + "=" * 60)
    print("Pretrained Weights Test PASSED!")
    print("=" * 60)
    
    return tower


def test_build_vision_tower():
    """测试通过builder构建DeepEncoderVisionTower"""
    print("\n" + "=" * 60)
    print("Testing build_vision_tower with use_deepencoder=True")
    print("=" * 60)
    
    from llava.model.multimodal_encoder.builder import build_vision_tower
    
    class MockConfig:
        mm_vision_tower = None
        vision_tower = None
        use_deepencoder = True
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
        s2 = False
    
    config = MockConfig()
    
    tower = build_vision_tower(config, delay_load=True)
    tower.load_model()
    
    print(f"   - Tower type: {type(tower).__name__}")
    print(f"   - Hidden size: {tower.hidden_size}")
    
    assert tower.hidden_size == 2048, "Expected hidden size 2048 for DeepEncoder"
    print("   ✓ Build test passed!")
    
    print("\n" + "=" * 60)
    print("Builder Test PASSED!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test DeepEncoder integration")
    parser.add_argument(
        "--deepseek_ocr_path", 
        type=str, 
        default=None,
        help="Path to DeepSeek-OCR model weights (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    args = parser.parse_args()
    
    print("\n" + "#" * 60)
    print("# GEM + DeepEncoder Integration Test")
    print("#" * 60)
    
    # 基本功能测试
    test_deepencoder_tower()
    
    # Builder测试
    test_build_vision_tower()
    
    # 预训练权重测试（可选）
    if args.deepseek_ocr_path and os.path.exists(args.deepseek_ocr_path):
        test_with_pretrained_weights(args.deepseek_ocr_path)
    else:
        print("\n[Info] Skipping pretrained weights test (no path provided)")
    
    print("\n" + "#" * 60)
    print("# ALL TESTS PASSED!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

