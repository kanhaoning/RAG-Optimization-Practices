import logging
import os
from dataclasses import dataclass, field

from datasets import Value, load_dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import MarginMSELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from transformers import HfArgumentParser

# 设置日志记录
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 禁用 W&B 日志记录
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    与模型和数据相关的参数
    """

    model_name_or_path: str = field(metadata={"help": "预训练模型的路径或 Hugging Face Hub 上的名称"})
    train_data: str = field(metadata={"help": "训练数据文件路径 (.jsonl)"})
    eval_data: str = field(metadata={"help": "评估数据文件路径 (.jsonl)"})
    max_length: int = field(
        default=512, metadata={"help": "模型处理的最大序列长度 (query + passage)"}
    )


def main():
    # 1. 解析命令行参数
    parser = HfArgumentParser((ModelArguments, CrossEncoderTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(f"模型参数: {model_args}")
    logger.info(f"训练参数: {training_args}")
    logger.info(f"训练脚本已适配 MarginMSELoss，将使用 (query, positive, negative) 三元组进行训练。")


    # 2. 初始化 CrossEncoder 模型
    # 对于回归/蒸馏任务，num_labels 设置为 1
    model = CrossEncoder(
        model_args.model_name_or_path,
        num_labels=1,
        max_length=model_args.max_length,
    )

    # 3. 加载和预处理数据集
    logger.info("正在加载数据集...")
    # 数据集应包含 'query', 'positive', 'negative', 'score' 列
    train_dataset = load_dataset(
        "json", data_files=model_args.train_data
    )["train"]
    eval_dataset = load_dataset(
        "json", data_files=model_args.eval_data
    )["train"]

    # 确保 'score' 列是 float 类型
    train_dataset = train_dataset.cast_column("score", Value("float32"))
    eval_dataset = eval_dataset.cast_column("score", Value("float32"))

    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"评估集样本数: {len(eval_dataset)}")
    logger.info(f"训练集的一个样本: {train_dataset[0]}")


    # 4. 定义损失函数
    # MarginMSELoss 用于知识蒸馏，它处理 (query, positive, negative) 三元组
    loss = MarginMSELoss(model)

    # 5. 初始化 Trainer
    # 注意：我们没有提供自定义的 evaluator，因为 CECorrelationEvaluator 不适用。
    # Trainer 将默认在评估集上计算损失（loss）作为评估指标。
    # 因此，'metric_for_best_model' 应设置为 'eval_loss'。
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    # 6. 开始训练
    logger.info("开始模型训练...")
    trainer.train()

    # 7. 保存最终模型
    logger.info("训练完成，保存最终模型...")
    model.save_pretrained(training_args.output_dir)
    logger.info(f"模型已保存至: {training_args.output_dir}")


if __name__ == "__main__":
    main()