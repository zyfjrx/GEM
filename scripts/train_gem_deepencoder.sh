#!/bin/bash

###############################################################################
# GEM + DeepEncoder Training Script
###############################################################################
#
# 使用 DeepSeek-OCR 的 DeepEncoder 作为视觉编码器训练 GEM 模型
#
# 模式选择：
#   - Base Mode: 适合初期预训练，形状固定，batch size可以较大
#   - Gundam Mode: 适合大图/长条图，动态切分，需要自定义collator
#
# 建议训练策略：
#   Stage 1: Base Mode + 冻结视觉编码器 + 只训练Projector (1 epoch)
#   Stage 2: Gundam Mode (可选) + 全模型微调 (2-3 epochs)
#
###############################################################################

# ==================== 环境配置 ====================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ==================== 模型和数据路径 ====================
# 基础 LLM（如 Llama-2）
MODEL_BASE="meta-llama/Llama-2-7b-hf"

# DeepEncoder 预训练权重路径（必须修改！）
DEEPENCODER_WEIGHTS="/path/to/deepencoder/weights"  # TODO: 修改为实际路径

# 数据路径
DATA_PATH="./dataset/ecg_qa_train.json"
IMAGE_FOLDER="./dataset/ecg_images"
ECG_FOLDER="./dataset/ecg_data"  # ECG 数据文件夹

# ECG Tower 配置（CoCa 模型）
ECG_TOWER="/path/to/ecg_tower/weights"  # TODO: 修改为实际路径
OPEN_CLIP_CONFIG="coca_ViT-B-32"  # ECG 编码器配置

# 输出目录
OUTPUT_DIR="./checkpoints/gem-deepencoder-base"

# ==================== DeepEncoder 配置 ====================
# 是否使用 DeepEncoder（True 表示使用 DeepSeek-OCR 的双编码器）
USE_DEEPENCODER=True

# 是否使用 Gundam Mode（动态切分）
# False: Base Mode（推荐初期训练，1024x1024固定尺寸）
# True: Gundam Mode（大图/长条图，动态切分，需要自定义collator）
USE_GUNDAM_MODE=False

# ==================== 训练超参数 ====================
# 学习率设置
LR_MM_PROJECTOR=1e-3    # Projector 学习率（预训练阶段）
LR_LLM=2e-5             # LLM 学习率（全模型微调阶段）

# Batch Size（根据显存调整）
# Base Mode 推荐: 16-32
# Gundam Mode 推荐: 4-8
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2

# 训练轮数
NUM_TRAIN_EPOCHS=1

# ==================== 模型配置 ====================
# Projector 类型（mlp2x_gelu 表示两层MLP+GELU）
MM_PROJECTOR_TYPE="mlp2x_gelu"

# 使用哪一层的视觉特征（-2 表示倒数第二层）
MM_VISION_SELECT_LAYER=-2

# 是否冻结视觉编码器（True 表示只训练 Projector）
FREEZE_VISION_TOWER=True

# 是否训练 Projector
TUNE_MM_MLP_ADAPTER=True

# ==================== DeepSpeed 配置 ====================
DEEPSPEED_CONFIG="./scripts/zero2.json"

###############################################################################
# 开始训练
###############################################################################

echo "=========================================="
echo "GEM + DeepEncoder Training"
echo "=========================================="
echo "DeepEncoder Weights: $DEEPENCODER_WEIGHTS"
echo "Use Gundam Mode: $USE_GUNDAM_MODE"
echo "Freeze Vision Tower: $FREEZE_VISION_TOWER"
echo "Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS"
echo "=========================================="

# 检查 DeepEncoder 权重是否存在
if [ ! -e "$DEEPENCODER_WEIGHTS" ]; then
    echo "❌ Error: DeepEncoder weights not found at $DEEPENCODER_WEIGHTS"
    echo "Please download/prepare the weights first!"
    exit 1
fi

# 运行训练
deepspeed llava/train/train_mem.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_BASE \
    --version plain \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --ecg_folder $ECG_FOLDER \
    --ecg_tower $ECG_TOWER \
    --open_clip_config $OPEN_CLIP_CONFIG \
    --vision_tower $DEEPENCODER_WEIGHTS \
    --use_deepencoder $USE_DEEPENCODER \
    --use_gundam_mode $USE_GUNDAM_MODE \
    --mm_projector_type $MM_PROJECTOR_TYPE \
    --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --freeze_vision_tower $FREEZE_VISION_TOWER \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate $LR_MM_PROJECTOR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo "=========================================="
echo "Training completed!"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
