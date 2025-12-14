#!/bin/bash

# =============================================================================
# GEM Training Script with DeepEncoder (DeepSeek-OCR's SAM+CLIP-L)
# 使用DeepSeek-OCR的双编码器(SAM ViT-B + CLIP-L)替代原有CLIP视觉编码器
# =============================================================================

# distributed training configurations
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="1234"
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# your huggingface configurations
export HF_HOME=""

LLM_VERSION=""  # e.g., "PULSE-ECG/PULSE-7B" or local path
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
DATA_VERSION=""
BASE_RUN_NAME="GEM-DeepEncoder-${LLM_VERSION_CLEAN}-${DATA_VERSION}-finetune"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

version=llava_v1

# =============================================================================
# Data paths
# =============================================================================
data_path=""          # JSON文件路径
image_folder=""       # ECG图像文件夹
ecg_folder=""         # ECG时间序列文件夹
ecg_tower=""          # ECG-CoCa checkpoint路径

# =============================================================================
# DeepEncoder Configuration (DeepSeek-OCR)
# =============================================================================
# DeepSeek-OCR模型路径 (用于提取DeepEncoder权重)
# 可以是:
#   - HuggingFace模型ID: "deepseek-ai/DeepSeek-OCR"
#   - 本地下载的完整模型目录 (包含safetensors/bin文件)
#   - 你微调后合并的模型路径
deepseek_ocr_path=""  # e.g., "/path/to/deepseek-ocr-model"

# DeepEncoder 模式配置
# 可选模式:
#   - tiny:   base_size=512,  image_size=512,  crop_mode=false (64 tokens)
#   - small:  base_size=640,  image_size=640,  crop_mode=false (100 tokens)
#   - base:   base_size=1024, image_size=1024, crop_mode=false (256 tokens) [默认，推荐用于ECG]
#   - large:  base_size=1280, image_size=1280, crop_mode=false (400 tokens)
#   - gundam: base_size=1024, image_size=640,  crop_mode=true  (多分辨率裁剪)
DEEPENCODER_MODE="base"

# 可选：自定义覆盖模式默认值 (留空则使用模式默认值)
# DEEPENCODER_BASE_SIZE=""    # 全局视图尺寸 (SAM输入)
# DEEPENCODER_IMAGE_SIZE=""   # 局部裁剪尺寸 (动态预处理时使用)
# DEEPENCODER_CROP_MODE=""    # true/false 是否启用动态裁剪

# =============================================================================
# Training hyperparameters
# =============================================================================
num_epochs=1
GRAD_ACC_STEP=2
BATCH_PER_GPU=4  # DeepEncoder较大 (SAM+CLIP ~400M)，建议减小batch size
TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU * $GRAD_ACC_STEP))
echo "TOTAL_BATCH_SIZE: ${TOTAL_BATCH_SIZE}"

# =============================================================================
# Run training
# =============================================================================
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${version} \
    --data_path ${data_path} \
    --ecg_folder ${ecg_folder} \
    --ecg_tower ${ecg_tower} \
    --open_clip_config coca_ViT-B-32 \
    --image_folder $image_folder \
    --vision_tower ${deepseek_ocr_path} \
    --use_deepencoder True \
    --deepencoder_mode ${DEEPENCODER_MODE} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio ori \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/${BASE_RUN_NAME} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
