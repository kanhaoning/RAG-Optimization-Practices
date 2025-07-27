"""
本脚本使用 DistillKLDivLoss 进行知识蒸馏，以微调学生模型。

它使用一个预先计算好的蒸馏数据集。在这个数据集中，教师模型已经为每个
(query, positive) 和 (query, negative) 对生成了相似度分数。

学生模型通过训练来最小化其输出的相似度分布与教师模型输出的相似度分布之间的
KL散度 (Kullback-Leibler Divergence)。这有助于学生模型模仿教师模型的排序行为。
"""
import logging
import os
from dataclasses import dataclass, field

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import HfArgumentParser

# 设置日志记录，以监控训练过程
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 不使用 Weights & Biases
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    与模型和数据相关的参数。
    """

    student_model_name_or_path: str = field(
        metadata={"help": "预训练学生模型的路径或其在 Hugging Face Hub 上的名称。"}
    )
    train_dataset_path: str = field(metadata={"help": "训练数据文件（.jsonl）的路径。"})
    temperature: float = field(
        default=2.0,
        metadata={"help": "用于 DistillKLDivLoss 的温度参数。较高的温度会产生更软的概率分布。"}
    )


def main():
    """
    执行训练过程的主函数。
    """
    # 1. 解析命令行参数
    parser = HfArgumentParser((ModelArguments, SentenceTransformerTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("--- 脚本开始 ---")
    logger.info(f"模型参数: {model_args}")
    logger.info(f"训练参数: {training_args}")

    # 2. 加载学生模型
    logger.info(f"从以下位置加载学生模型: {model_args.student_model_name_or_path}")
    student_model = SentenceTransformer(model_args.student_model_name_or_path)

    # 3. 加载训练数据集
    # 数据集应为 .jsonl 文件，每行是一个字典。
    # DistillKLDivLoss 要求输入格式为：
    # {"query": str, "positive": str, "negative": str, ..., "label": List[float]}
    logger.info(f"从以下位置加载训练数据: {model_args.train_dataset_path}")
    train_dataset = load_dataset("json", data_files=model_args.train_dataset_path)["train"]
    logger.info(f"成功加载 {len(train_dataset)} 条训练样本。")
    logger.info(f"一条训练样本示例: {train_dataset[0]}")

    # 4. 定义DistillKLDivLoss损失函数
    train_loss = losses.DistillKLDivLoss(
        model=student_model,
        temperature=model_args.temperature
    )
    logger.info(f"使用 DistillKLDivLoss 进行知识蒸馏，温度(temperature)设置为: {model_args.temperature}。")

    # 5. 初始化 Trainer
    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    # 6. 开始训练模型
    logger.info("开始模型训练...")
    trainer.train()

    # 7. 保存最终模型
    logger.info("训练完成。正在保存最终模型...")
    # 模型被保存到 `output_dir` 参数指定的目录中。
    trainer.save_model(training_args.output_dir)
    logger.info(f"模型成功保存至: {training_args.output_dir}")
    logger.info("--- 脚本结束 ---")


if __name__ == "__main__":
    main()