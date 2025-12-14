#!/usr/bin/env python3
"""
测试DeepEncoder集成是否正常工作
Usage: python test_deepencoder_integration.py --deepseek_ocr_path /path/to/deepseek-ocr-model
       python test_deepencoder_integration.py --compare_structure  # 对比模型结构
"""

import argparse
import torch
import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare_model_structure():
    """对比原始DeepEncoder和集成后的模型结构"""
    print("\n" + "=" * 80)
    print("对比模型结构: 原始 DeepEncoder vs 集成后的 DeepEncoderVisionTower")
    print("=" * 80)
    
    # 1. 加载原始 DeepEncoder 组件
    print("\n" + "-" * 40)
    print("【原始 DeepEncoder 组件】")
    print("-" * 40)
    
    from deepseek_ocr.deepencoder import build_sam_vit_b, build_clip_l
    
    original_sam = build_sam_vit_b(checkpoint=None)
    original_clip = build_clip_l()
    
    print("\n📦 SAM ViT-B (ImageEncoderViT):")
    print(f"   类型: {type(original_sam).__name__}")
    sam_params = sum(p.numel() for p in original_sam.parameters())
    print(f"   参数量: {sam_params:,} ({sam_params/1e6:.2f}M)")
    
    print("\n📦 CLIP-L (VitModel):")
    print(f"   类型: {type(original_clip).__name__}")
    clip_params = sum(p.numel() for p in original_clip.parameters())
    print(f"   参数量: {clip_params:,} ({clip_params/1e6:.2f}M)")
    
    print(f"\n   总参数量: {(sam_params + clip_params):,} ({(sam_params + clip_params)/1e6:.2f}M)")
    
    # 2. 加载集成后的 DeepEncoderVisionTower (直接导入模块避免复杂依赖)
    print("\n" + "-" * 40)
    print("【集成后的 DeepEncoderVisionTower】")
    print("-" * 40)
    
    # 直接导入deepencoder_tower模块以避免llava包的复杂依赖
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deepencoder_tower", 
        os.path.join(os.path.dirname(__file__), "llava/model/multimodal_encoder/deepencoder_tower.py")
    )
    deepencoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepencoder_module)
    DeepEncoderVisionTower = deepencoder_module.DeepEncoderVisionTower
    
    class Args:
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
    
    tower = DeepEncoderVisionTower(vision_tower=None, args=Args(), delay_load=True)
    tower.load_model()
    
    tower_sam_params = sum(p.numel() for p in tower.sam_model.parameters())
    tower_clip_params = sum(p.numel() for p in tower.vision_model.parameters())
    
    print(f"\n📦 tower.sam_model:")
    print(f"   类型: {type(tower.sam_model).__name__}")
    print(f"   参数量: {tower_sam_params:,} ({tower_sam_params/1e6:.2f}M)")
    
    print(f"\n📦 tower.vision_model (CLIP-L):")
    print(f"   类型: {type(tower.vision_model).__name__}")
    print(f"   参数量: {tower_clip_params:,} ({tower_clip_params/1e6:.2f}M)")
    
    print(f"\n   总参数量: {(tower_sam_params + tower_clip_params):,} ({(tower_sam_params + tower_clip_params)/1e6:.2f}M)")
    
    # 3. 对比结构
    print("\n" + "-" * 40)
    print("【结构对比】")
    print("-" * 40)
    
    # 对比 SAM
    print("\n🔍 SAM ViT-B 结构对比:")
    sam_match = compare_state_dict_keys(original_sam, tower.sam_model, "SAM")
    
    # 对比 CLIP
    print("\n🔍 CLIP-L 结构对比:")
    clip_match = compare_state_dict_keys(original_clip, tower.vision_model, "CLIP-L")
    
    # 4. 详细结构打印
    print("\n" + "-" * 40)
    print("【详细模型结构】")
    print("-" * 40)
    
    print("\n📋 原始 SAM 模型层:")
    print_model_layers(original_sam, prefix="  ", max_depth=2)
    
    print("\n📋 原始 CLIP-L 模型层:")
    print_model_layers(original_clip, prefix="  ", max_depth=2)
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("【总结】")
    print("=" * 80)
    
    if sam_match and clip_match:
        print("✅ 模型结构完全一致!")
    else:
        print("⚠️ 模型结构存在差异，请检查上方详情")
    
    print(f"\n参数量对比:")
    print(f"  原始 SAM:  {sam_params:,} vs 集成 SAM:  {tower_sam_params:,} → {'✅ 一致' if sam_params == tower_sam_params else '❌ 不一致'}")
    print(f"  原始 CLIP: {clip_params:,} vs 集成 CLIP: {tower_clip_params:,} → {'✅ 一致' if clip_params == tower_clip_params else '❌ 不一致'}")
    
    return sam_match and clip_match


def compare_state_dict_keys(model1, model2, name):
    """对比两个模型的state_dict keys"""
    keys1 = set(model1.state_dict().keys())
    keys2 = set(model2.state_dict().keys())
    
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2
    
    print(f"   共同的 keys: {len(common)}")
    print(f"   只在原始模型中: {len(only_in_1)}")
    print(f"   只在集成模型中: {len(only_in_2)}")
    
    if only_in_1:
        print(f"   ⚠️ 原始模型独有: {list(only_in_1)[:5]}{'...' if len(only_in_1) > 5 else ''}")
    if only_in_2:
        print(f"   ⚠️ 集成模型独有: {list(only_in_2)[:5]}{'...' if len(only_in_2) > 5 else ''}")
    
    # 对比shape
    shape_mismatch = []
    for key in common:
        shape1 = model1.state_dict()[key].shape
        shape2 = model2.state_dict()[key].shape
        if shape1 != shape2:
            shape_mismatch.append((key, shape1, shape2))
    
    if shape_mismatch:
        print(f"   ⚠️ Shape 不匹配:")
        for key, s1, s2 in shape_mismatch[:5]:
            print(f"      {key}: {s1} vs {s2}")
    else:
        print(f"   ✅ 所有共同 keys 的 shape 一致")
    
    return len(only_in_1) == 0 and len(only_in_2) == 0 and len(shape_mismatch) == 0


def print_model_layers(model, prefix="", max_depth=2, current_depth=0):
    """打印模型层结构"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{prefix}├─ {name}: {type(module).__name__} ({num_params:,} params)")
        if current_depth < max_depth - 1:
            print_model_layers(module, prefix + "│  ", max_depth, current_depth + 1)


def test_deepencoder_tower():
    """测试DeepEncoderVisionTower的基本功能"""
    print("=" * 60)
    print("Testing DeepEncoder Vision Tower")
    print("=" * 60)
    
    # 直接导入以避免复杂依赖链
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deepencoder_tower", 
        os.path.join(os.path.dirname(__file__), "llava/model/multimodal_encoder/deepencoder_tower.py")
    )
    deepencoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepencoder_module)
    DeepEncoderVisionTower = deepencoder_module.DeepEncoderVisionTower
    
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
    
    # 直接导入以避免复杂依赖链
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deepencoder_tower", 
        os.path.join(os.path.dirname(__file__), "llava/model/multimodal_encoder/deepencoder_tower.py")
    )
    deepencoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepencoder_module)
    DeepEncoderVisionTower = deepencoder_module.DeepEncoderVisionTower
    
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
    """测试通过builder构建DeepEncoderVisionTower (模拟builder逻辑)"""
    print("\n" + "=" * 60)
    print("Testing build_vision_tower logic with use_deepencoder=True")
    print("=" * 60)
    
    # 直接导入以避免复杂依赖链
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deepencoder_tower", 
        os.path.join(os.path.dirname(__file__), "llava/model/multimodal_encoder/deepencoder_tower.py")
    )
    deepencoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepencoder_module)
    DeepEncoderVisionTower = deepencoder_module.DeepEncoderVisionTower
    
    class MockConfig:
        mm_vision_tower = None
        vision_tower = None
        use_deepencoder = True
        deepencoder_mode = 'base'
        deepencoder_base_size = None
        deepencoder_image_size = None
        deepencoder_crop_mode = None
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
        s2 = False
    
    config = MockConfig()
    
    # 模拟 builder.py 的逻辑
    deepencoder_mode = getattr(config, 'deepencoder_mode', 'base')
    deepencoder_base_size = getattr(config, 'deepencoder_base_size', None)
    deepencoder_image_size = getattr(config, 'deepencoder_image_size', None)
    deepencoder_crop_mode = getattr(config, 'deepencoder_crop_mode', None)
    
    tower = DeepEncoderVisionTower(
        config.vision_tower, 
        args=config,
        mode=deepencoder_mode,
        base_size=deepencoder_base_size,
        image_size=deepencoder_image_size,
        crop_mode=deepencoder_crop_mode,
        delay_load=True
    )
    tower.load_model()
    
    print(f"   - Tower type: {type(tower).__name__}")
    print(f"   - Hidden size: {tower.hidden_size}")
    print(f"   - Mode: {tower.mode}")
    print(f"   - SAM image size (base_size): {tower.sam_image_size}")
    print(f"   - Local image size: {tower.local_image_size}")
    print(f"   - Crop mode: {tower.crop_mode}")
    
    assert tower.hidden_size == 2048, "Expected hidden size 2048 for DeepEncoder"
    assert tower.mode == 'base', "Expected mode 'base'"
    assert tower.sam_image_size == 1024, "Expected base_size 1024 for base mode"
    print("   ✓ Build test passed!")
    
    print("\n" + "=" * 60)
    print("Builder Test PASSED!")
    print("=" * 60)


def test_deepencoder_modes():
    """测试不同的DeepEncoder模式"""
    print("\n" + "=" * 60)
    print("Testing DeepEncoder Modes")
    print("=" * 60)
    
    # 直接导入以避免依赖问题
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deepencoder_tower", 
        os.path.join(os.path.dirname(__file__), "llava/model/multimodal_encoder/deepencoder_tower.py")
    )
    deepencoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepencoder_module)
    DeepEncoderVisionTower = deepencoder_module.DeepEncoderVisionTower
    
    class Args:
        mm_vision_select_layer = -2
        mm_vision_select_feature = 'patch'
        unfreeze_mm_vision_tower = False
    
    args = Args()
    
    # 测试所有预定义模式
    modes = {
        "tiny":   {"base_size": 512,  "image_size": 512,  "crop_mode": False},
        "small":  {"base_size": 640,  "image_size": 640,  "crop_mode": False},
        "base":   {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large":  {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640,  "crop_mode": True},
    }
    
    print("\n📋 测试预定义模式:")
    for mode_name, expected in modes.items():
        tower = DeepEncoderVisionTower(
            vision_tower=None, 
            args=args, 
            delay_load=True,
            mode=mode_name
        )
        
        assert tower.sam_image_size == expected["base_size"], \
            f"Mode {mode_name}: Expected base_size {expected['base_size']}, got {tower.sam_image_size}"
        assert tower.local_image_size == expected["image_size"], \
            f"Mode {mode_name}: Expected image_size {expected['image_size']}, got {tower.local_image_size}"
        assert tower.crop_mode == expected["crop_mode"], \
            f"Mode {mode_name}: Expected crop_mode {expected['crop_mode']}, got {tower.crop_mode}"
        
        print(f"   ✅ {mode_name}: base_size={tower.sam_image_size}, image_size={tower.local_image_size}, crop_mode={tower.crop_mode}")
    
    # 测试自定义覆盖
    print("\n📋 测试自定义参数覆盖:")
    tower = DeepEncoderVisionTower(
        vision_tower=None,
        args=args,
        delay_load=True,
        mode="base",
        base_size=800,
        image_size=400,
        crop_mode=True
    )
    
    assert tower.sam_image_size == 800, f"Expected custom base_size 800, got {tower.sam_image_size}"
    assert tower.local_image_size == 400, f"Expected custom image_size 400, got {tower.local_image_size}"
    assert tower.crop_mode == True, f"Expected custom crop_mode True, got {tower.crop_mode}"
    print(f"   ✅ 自定义: base_size={tower.sam_image_size}, image_size={tower.local_image_size}, crop_mode={tower.crop_mode}")
    
    print("\n" + "=" * 60)
    print("Mode Test PASSED!")
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
    parser.add_argument(
        "--compare_structure",
        action="store_true",
        help="Compare model structure between original DeepEncoder and integrated version"
    )
    args = parser.parse_args()
    
    print("\n" + "#" * 60)
    print("# GEM + DeepEncoder Integration Test")
    print("#" * 60)
    
    # 如果指定了 --compare_structure，只运行结构对比
    if args.compare_structure:
        compare_model_structure()
        return
    
    # 基本功能测试
    test_deepencoder_tower()
    
    # 模式测试
    test_deepencoder_modes()
    
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

