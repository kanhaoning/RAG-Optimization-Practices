#!/bin/bash

# 评估模型在测试集上的性能
python evaluation.py \
  --model_before_path /root/autodl-tmp/bge-reranker-v2-m3 \
  --model_after_path output/checkpoint-1217 \
  --dataset_path test.jsonl