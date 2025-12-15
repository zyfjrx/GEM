import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import transformers
from typing import Dict, Optional, Sequence, List
import re
from PIL import Image
# import wfdb  <-- 移除：不需要读取波形库
import math


# ... (保留 split_list, get_chunk, preprocess_qwen 函数不变) ...

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)

    # 加载模型 (这里会自动加载你集成好的 DeepEncoderVisionTower)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = []
    with open(args.question_file, "r") as f:
        json_data = json.load(f)
        for line in json_data:
            questions.append({
                "question_id": line["id"],
                "image": line["image"],
                # 兼容不同格式的问题字段
                "text": line["conversations"][0]["value"].replace("<image>\n", ""),
                # "ecg": line["ecg"], <-- 移除：不需要 ECG 文件路径
            })

    # 断点续跑逻辑 (保留)
    existing_question_ids = set()
    if os.path.exists(args.answers_file):
        with open(args.answers_file, "r") as ans_file:
            for line in ans_file:
                existing_data = json.loads(line)
                existing_question_ids.add(existing_data["question_id"])

    output_file = open(args.answers_file, "a")  # 使用 "a" (append) 模式更安全

    for line in tqdm(questions):
        idx = line["question_id"]
        if idx in existing_question_ids:
            continue

        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        # 构造 Prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 处理图像
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        # 这里会调用你写的 DeepEncoderImageProcessor
        image_tensor = process_images([image], image_processor, model.config)[0]

        # === 关键修改：移除了加载 ECG 波形的代码 ===
        # 原代码中 wfdb.rdsamp ... 全部删掉

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # ecgs = ecg, <-- 移除：不传入 ecg 参数
                images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda'),  # 确保类型匹配
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        new_answer = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }

        output_file.write(json.dumps(new_answer) + "\n")
        output_file.flush()

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,default="/Users/zhangyf/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True, default="")
    # parser.add_argument("--ecg-folder", type=str, default="") <-- 移除
    parser.add_argument("--question-file", type=str, required=True, default="question.json")
    parser.add_argument("--answers-file", type=str, required=True, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)  # 建议稍微调低温度以获得稳定测量值
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    args = parser.parse_args()

    eval_model(args)
