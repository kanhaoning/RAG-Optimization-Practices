#!/bin/bash

# 使用 vLLM + Qwen3-Reranker-8B 生成训练/测试数据的 logits 分数
python generate_logits.py \
  --model_path BAAI/Qwen3-Reranker-8B \
  --input_files train.jsonl test.jsonl \
  --output_suffix _distill_qwen3_8b_vLLMlogit \
  --batch_size 8 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.9 \
  --task_instruction "Given a web search query, retrieve relevant passages that answer the query"