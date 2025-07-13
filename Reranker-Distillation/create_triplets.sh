#!/bin/bash

# 将原始 logits 数据转换为 MarginMSE 格式
python create_triplets.py \
  --input_files train_distill_qwen3_8b_vLLMlogit.jsonl test_distill_qwen3_8b_vLLMlogit.jsonl \
  --output_suffix _margin_sampled \
  --top_k 8 \
  --num_negatives 4