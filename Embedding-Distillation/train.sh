# --nproc_per_node:             指定在当前机器上使用多少个GPU。
# --student_model_name_or_path: 学生模型的路径，例如你下载的 'BAAI/bge-m3'
# --train_dataset_path:         经过步骤1生成的蒸馏数据集的路径
# --output_dir:                 模型训练的输出目录，用于保存 checkpoint 和最终模型
# --num_train_epochs:           训练的总轮数
# --per_device_train_batch_size:每个 GPU 上的训练批次大小
# --gradient_accumulation_steps:梯度累积的步数，(batch_size * gradient_accumulation_steps) 才是有效批次大小
# --learning_rate:              学习率
# --warmup_ratio:               学习率预热的比例
# --logging_steps:              每隔多少步打印一次日志
# --save_strategy:              模型保存策略，"steps" 表示按步数保存
# --save_only_model:            只保存模型权重，不保存优化器状态等，可以节省空间
# --save_steps:                 每隔多少步保存一次 checkpoint
# --save_total_limit:           最多保存多少个 checkpoint
# --temperature:                损失函数DistillKLDivLoss的温度超参数
# --bf16:                       启用 bf16 混合精度训练，可以加速训练并减少显存占用
# --eval_strategy:              评估策略，"no" 表示训练期间不进行评估
torchrun --nproc_per_node 1 train.py \
    --student_model_name_or_path your_path_to/bge-m3 \
    --train_dataset_path dataset_scidocs/validation_kldiv_distill.jsonl \
    --output_dir output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --learning_rate 6e-5 \
    --warmup_ratio 0.02 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_only_model true \
    --save_steps 1000 \
    --save_total_limit 2 \
    --temperature 1 \
    --bf16 \
    --eval_strategy "no"