# 使用 vLLM + Qwen3-Embedding-8B 为 scidocs 数据集生成蒸馏分数
# --input_files: 输入的原始数据，可以有多个，用空格隔开
# --output_suffix: 输出文件名的后缀
# --batch_size: vLLM编码时的批处理大小，根据显存调整
# --tensor_parallel_size: GPU卡数
python generate_distillation_data.py \
  --teacher_model_path "/root/autodl-tmp/Qwen3-Embedding-8B" \
  --input_files scidocs-reranking/validation.jsonl \
  --output_suffix _kldiv_distill \
  --batch_size 4 \
  --tensor_parallel_size 1
