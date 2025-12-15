from PIL import Image
import math


# 复制自 modeling_deepseekocr.py 的核心逻辑
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def analyze_image_patches(image_path):
    # 模拟配置
    min_num = 2
    max_num = 9  # 最多切9块
    image_size = 640  # Local View 尺寸

    try:
        image = Image.open(image_path)
    except:
        print(f"无法打开图片: {image_path}")
        return

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算候选网格
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)

    # 找到最佳网格
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    num_patches = target_aspect_ratio[0] * target_aspect_ratio[1]

    print(f"\n--- 图片分析: {image_path} ---")
    print(f"原始尺寸: {orig_width} x {orig_height}")
    print(f"长宽比: {aspect_ratio:.2f}")
    print(f"匹配网格: {target_aspect_ratio[0]} (宽) x {target_aspect_ratio[1]} (高)")
    print(f"切片数量 (N): {num_patches}")
    print(f"预期日志输出: PATCHES: torch.Size([{num_patches}, 100, 1280])")


# 替换成你的一张心电图路径
image_path = "/Users/zhangyf/PycharmProjects/cfel/GEM/41328635-0.png"
analyze_image_patches(image_path)
