# --model_before:       蒸馏前的原始学生模型路径 (e.g., "BAAI/bge-m3")
# --model_after:        蒸馏后保存的模型 checkpoint 路径 (e.g., "output/checkpoint-953")
# --in_domain_dataset:  领域内测试集路径，用于评估模型在目标任务上的性能
# --out_domain_dataset: 领域外测试集路径，用于评估模型的泛化能力和是否发生灾难性遗忘
python evaluation.py \
  --model_before BAAI/bge-m3 \
  --model_after output/checkpoint-XXX \
  --in_domain_dataset dataset_scidocs/test.jsonl \
  --out_domain_dataset dataset_stackoverflowdupquestions/test.jsonl