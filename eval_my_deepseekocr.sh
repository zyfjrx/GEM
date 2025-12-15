#!/bin/bash

# 你的模型路径
MODEL_PATH="path/to/your/trained/deepseekocr-llava-checkpoint"
# GEM 提供的测试图片路径
IMAGE_FOLDER="path/to/gem/data/ecg_images"
# GEM 提供的测试问题文件 (例如 ECG-Grounding)
QUESTION_FILE="path/to/gem/data/ecg-grounding-test-mimiciv.json"
# 结果保存路径
OUTPUT_FILE="eval_outputs/deepseekocr_grounding_result.jsonl"

mkdir -p eval_outputs

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_deepseekocr.py \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --question-file $QUESTION_FILE \
    --answers-file $OUTPUT_FILE \
    --conv-mode "llava_v1" \
    --temperature 0
