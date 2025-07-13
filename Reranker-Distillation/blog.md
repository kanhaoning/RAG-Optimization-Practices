# 将重排序大模型Qwen3-Reranker-8B的知识蒸馏到小模型BGE-reranker-m3-v2上
BGE-reranker-m3-v2是一个非常好用的重排序模型，在RAG（检索增强生成）中用于进一步优化检索出的文档。但是有一个痛点是用大模型合成、甚至人工标注query、positive、negative三元组数据用于训练微调会比较麻烦且成本较高。最近阿里云发布了Qwen3-reranker系列SOTA重排序模型，本文尝试一下用最强的8B大模型Qwen3-reranker-8B来知识蒸馏0.6B的BGE-reranker-m3-v2小模型，取得了XX%的提升。

## 工具：
### 学生模型：BGE-reranker-m3-v2
教师模型: Qwen3-reranker-8B  
训练/评测框架：sentence-transformer  
数据集：MTEB/stackoverflowdupquestions-reranking  
推理框架：vLLM

## 训练方法：
目标是让学生模型计算（查询，更相关文档）的相关性分数和（查询，更不相关文档）的相关性分数的差值与教师模型给这两个pair计算的分数的差值接近。
loss 函数是Margin-MSE。公式如下：
```
L (𝑄, 𝑃+, 𝑃−) = MSE(𝑀𝑠 (𝑄, 𝑃+) − 𝑀𝑠 (𝑄, 𝑃−), 𝑀𝑡 (𝑄, 𝑃+) − 𝑀𝑡 (𝑄, 𝑃−))
```
𝑄是查询，𝑃+是相关性更高的文档，𝑃−是相关性更低的文档。𝑀𝑡()、 𝑀𝑠()分别是教师模型计算的相关性分数。MSE()是均方误差。

## 环境：
以下是我的环境中相关的几个库的版本，如果其中你没有安装，可以直接使用pip install XXX安装。
```
Package                                  Version                  
---------------------------------------- ------------------------ 
torch                                    2.6.0
sentence-transformers                    5.0.0
transformers                             4.53.1
vllm                                     0.8.4
```

## 实现：
### 第一步：下载模型和数据集
#### 1.下载bge-reranker-v2-m3模型
```
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-v2-m3')
```
#### 2.下载Qwen3-8B-reranker模型
```
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-Reranker-8B')
```
#### 3.下载stackoverflowdupquestions-reranking数据集
进入https://www.modelscope.cn/datasets/MTEB/stackoverflowdupquestions-reranking/files  
手动点击下载train.jsonl.gz、test.jsonl.gz数据到你的实验环境（直接拉取数据集可能会有BUG）  

执行以下命令解压：
```
gunzip train.jsonl.gz
gunzip test.jsonl.gz
```

### 第二步：大模型计算Logits

完整代码如下：
```
import os
import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from tqdm import tqdm
import math
from vllm.inputs.data import TokensPrompt

# --- 1. 路径和参数配置 ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 请将模型路径更改为您实际使用的 8B 模型路径
model_path = '/root/autodl-tmp/Qwen3-Reranker-8B'

# 可以在这里添加任意数量的文件路径
input_dataset_paths = [
    'train.jsonl',
    'test.jsonl'
]

# 【自动生成】输出文件路径将根据输入文件自动生成，无需手动配置
# 例如 'train.jsonl' -> 'train_distill_qwen3_8b_vLLMlogit.jsonl'
output_suffix = '_distill_qwen3_8b_vLLMlogit'

batch_size = 8  # 使用 vllm 可以显著增大 batch_size，请根据显存调整

# --- 2. VLLM 模型加载和相关配置 ---

print(f"Using vLLM for inference.")
number_of_gpu = torch.cuda.device_count()
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 使用 vLLM 加载模型
# enable_prefix_caching=True 对 rerank 这种固定模板的任务有很好的加速效果
print("Loading model with vLLM...")
model = LLM(
    model=model_path,
    tensor_parallel_size=number_of_gpu,
    max_model_len=8192,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9 # 根据您的显存情况调整
)
print("Model loaded successfully.")

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
max_length=8192
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
sampling_params = SamplingParams(temperature=0,
    max_tokens=1,
    logprobs=20,
    allowed_token_ids=[true_token, false_token],
)

# 使用Qwen3-reranker官方默认的指令 (Instruction)
task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'


# 生成大模型messages
def format_and_tokenize_inputs(queries, docs, instruction):
    """使用 apply_chat_template 格式化并 tokenize 输入"""
    messages = []
    for query, doc in zip(queries, docs):
        # 构建符合 Qwen3 聊天格式的输入
        message = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"}
        ]
        messages.append(message)
    templated_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False
    )
    processed_messages = [ele[:max_length] + suffix_tokens for ele in templated_messages]
    final_messages = [TokensPrompt(prompt_token_ids=ele) for ele in processed_messages]
    return final_messages

# 【逻辑不变】函数定义完全保留
def compute_scores_vllm(batch_queries, batch_docs):
    """计算分数的函数"""
    tokenized_batch = format_and_tokenize_inputs(batch_queries, batch_docs, task_instruction)
    outputs = model.generate(tokenized_batch, sampling_params=sampling_params, use_tqdm=False)

    scores = []
    for i in range(len(outputs)):
        final_logprobs = outputs[i].outputs[0].logprobs[-1]

        # 处理 "yes" token
        if true_token not in final_logprobs:
            true_logprob = -10.0 # 使用浮点数
        else:
            true_logprob = final_logprobs[true_token].logprob

        # 处理 "no" token
        if false_token not in final_logprobs:
            false_logprob = -10.0 # 使用浮点数
        else:
            false_logprob = final_logprobs[false_token].logprob

        # 计算logit空间的差值 (更稳定的计算方式)
        # logit = log(prob_yes) - log(prob_no)
        # 由于logprob是log_softmax的结果，直接相减即可
        logit_diff = true_logprob - false_logprob
        scores.append(logit_diff)

    return scores


# --- 3. 数据集生成主循环 ---
# 遍历输入文件列表，对每个文件进行处理
for input_dataset_path in input_dataset_paths:
    # 自动构建输出文件路径
    base_name, extension = os.path.splitext(input_dataset_path)
    output_distill_path = f"{base_name}{output_suffix}{extension}"

    print("\n" + "="*80)
    print(f"Start processing file: {input_dataset_path}")
    print(f"Output will be saved to: {output_distill_path}")
    print("="*80 + "\n")

    # 加载并重构数据集
    print(f"Loading and reformatting data from {input_dataset_path}...")
    try:
        with open(input_dataset_path, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_dataset_path}. Skipping this file.")
        continue # 如果文件不存在，跳过当前循环

    # 将原始数据格式转换为扁平的 (query, passage) 对列表
    reformatted_data = []
    for item in tqdm(original_data[:], desc="Reformatting data"):
        query = item['query']
        # 添加 positive passages
        for passage in item.get('positive', []):
            reformatted_data.append({'query': query, 'passage': passage})
        # 添加 negative passages
        for passage in item.get('negative', []):
            reformatted_data.append({'query': query, 'passage': passage})

    print(f"Original items: {len(original_data)}, Reformatted to {len(reformatted_data)} query-passage pairs.")
    print(f"Starting to generate distillation data for {len(reformatted_data)} examples using vLLM...")

    # 打开输出文件准备写入
    with open(output_distill_path, 'w', encoding='utf-8') as out_f:
        # 使用 tqdm 创建进度条，分批处理【重构后】的数据
        for i in tqdm(range(0, len(reformatted_data), batch_size), desc=f"Generating Distill Data for {os.path.basename(input_dataset_path)}"):
            batch_data = reformatted_data[i:i+batch_size]
            # 从重构后的数据中提取 query 和 passage
            queries = [sample['query'] for sample in batch_data]
            documents = [sample['passage'] for sample in batch_data]

            # 计算教师模型的 logits (在此处是 logprobs 的差值)，此函数调用逻辑不变
            scores_batch = compute_scores_vllm(queries, documents)

            # 将结果写入新的 JSONL 文件
            for j in range(len(batch_data)):
                final_score = scores_batch[j]

                # 输出格式与原代码一致
                distill_sample = {
                    "query": queries[j],
                    "passage": documents[j],
                    "score": final_score
                }
                out_f.write(json.dumps(distill_sample, ensure_ascii=False) + '\n')
    
    print(f"\nDistillation for {input_dataset_path} successfully generated and saved to: {output_distill_path}")


# --- 4. 清理资源 ---
# 在所有文件处理完成后再清理模型资源
destroy_model_parallel()
print("\nAll files have been processed. Model resources destroyed.")
print("Script finished.")
```

正常情况会打印如下log，表明完成生成logit:
```
Model loaded successfully.

================================================================================
Start processing file: train.jsonl
Output will be saved to: train_distill_qwen3_8b_vLLMlogit.jsonl
================================================================================

Loading and reformatting data from train.jsonl...
Reformatting data: 100%|██████████| 19847/19847 [00:00<00:00, 99606.16it/s]
Original items: 19847, Reformatted to 593522 query-passage pairs.
Starting to generate distillation data for 593522 examples using vLLM...
Generating Distill Data for train.jsonl: 100%|██████████| 74191/74191 [1:38:53<00:00, 12.50it/s]

Distillation for train.jsonl successfully generated and saved to: train_distill_qwen3_8b_vLLMlogit.jsonl

================================================================================
Start processing file: test.jsonl
Output will be saved to: test_distill_qwen3_8b_vLLMlogit.jsonl
================================================================================

Loading and reformatting data from test.jsonl...
Reformatting data: 100%|██████████| 2992/2992 [00:00<00:00, 112759.63it/s]
Original items: 2992, Reformatted to 89470 query-passage pairs.
Starting to generate distillation data for 89470 examples using vLLM...
Generating Distill Data for test.jsonl: 100%|██████████| 11184/11184 [14:53<00:00, 12.51it/s]

Distillation for test.jsonl successfully generated and saved to: test_distill_qwen3_8b_vLLMlogit.jsonl

All files have been processed. Model resources destroyed.
Script finished.
```

这一步会生成两个jsonl文件
```
train_distill_qwen3_8b_vLLMlogit.jsonl
test_distill_qwen3_8b_vLLMlogit.jsonl
```
内容大致如下：
```
{"query": "String isNullOrEmpty in Java?", "passage": "Java equivalent of c# String.IsNullOrEmpty() and String.IsNullOrWhiteSpace()", "score": 0.7499999403953552}
......
```
### 第三步：生成query、positive、negative三元组数据集

完整代码如下：
```
import json
import itertools
import os
from collections import defaultdict
from tqdm import tqdm
import random # 引入 random 库

def convert_to_margin_mse_format_sampled(input_file_path: str,
                                         output_file_path: str,
                                         top_k: int = 5,
                                         num_negatives: int = 5):
    """
    将 (query, passage, score) 格式的数据集转换为 (query, positive, negative, score_diff) 格式，
    并使用智能采样策略控制数据集大小。

    Args:
        input_file_path (str): 输入的 .jsonl 文件路径。
        output_file_path (str): 输出转换后格式的 .jsonl 文件路径。
        top_k (int): 对于每个query，选择分数最高的k个passage作为候选正例。
        num_negatives (int): 为每个正例选择的负例数量。
    """
    if not os.path.exists(input_file_path):
        print(f"错误: 输入文件未找到: {input_file_path}")
        return

    print(f"▶️ 开始处理文件: {input_file_path}")
    print(f"   - 采样策略: top_k={top_k}, num_negatives={num_negatives}")

    # 1. 按 query 分组
    query_to_passages = defaultdict(list)
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc="  - 步骤1/3: 读取并分组数据"):
            try:
                data = json.loads(line)
                query_to_passages[data['query']].append({
                    "passage": data['passage'],
                    "score": data['score']
                })
            except json.JSONDecodeError:
                print(f"  - 警告: 跳过格式错误的行: {line.strip()}")
                continue

    print(f"  - 找到了 {len(query_to_passages)} 个独立的 query。")

    # 2. 排序、采样并生成三元组
    triplets_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for query, passages in tqdm(query_to_passages.items(), desc="  - 步骤2/3: 排序并生成采样三元组"):
            if len(passages) < 2:
                continue

            # 按分数从高到低排序
            passages.sort(key=lambda x: x['score'], reverse=True)

            # 选择 top_k 作为正例
            positive_candidates = passages[:top_k]

            for i, p_pos in enumerate(positive_candidates):
                # 负例池是当前正例之后的所有 passage
                negative_pool = passages[i+1:]
                if not negative_pool:
                    continue

                # 选择 "hard negatives" (分数最接近的)
                # 如果负例池大小超过 num_negatives，就从中选择最难的 num_negatives 个
                # 也可以随机选择: negative_samples = random.sample(negative_pool, min(len(negative_pool), num_negatives))
                negative_samples = negative_pool[:num_negatives]

                for p_neg in negative_samples:
                    # 确保 passage 内容不同
                    if p_pos['passage'] == p_neg['passage']:
                        continue

                    score_diff = p_pos['score'] - p_neg['score']
                    
                    # 只有当分数差为正时才是有意义的pair（排序保证了这一点，但以防万一）
                    if score_diff > 0:
                        new_record = {
                            "query": query,
                            "positive": p_pos['passage'],
                            "negative": p_neg['passage'],
                            "score": score_diff
                        }
                        f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        triplets_count += 1
    
    print(f"  - 步骤3/3: 转换完成！")
    print(f"✅ (采样后) 总共生成了 {triplets_count} 个三元组, 已保存到: {output_file_path}")


if __name__ == '__main__':
    files_to_process = [
        "train_distill_qwen3_8b_vLLMlogit.jsonl",
        "test_distill_qwen3_8b_vLLMlogit.jsonl"
    ]

    for input_file in files_to_process:
        base_name, extension = os.path.splitext(input_file)
        # 更新输出文件名以反映采样方法
        output_file = f"{base_name}_margin_sampled{extension}"
        
        # 运行带有采样逻辑的转换函数
        # 你可以调整 top_k 和 num_negatives 来控制数据集大小和质量
        # 例如，k=5, neg=5 -> 每个query最多生成 5*5=25 个样本
        # 例如，k=10, neg=3 -> 每个query最多生成 10*3=30 个样本
        convert_to_margin_mse_format_sampled(input_file, output_file, top_k=8, num_negatives=4)
        print("-" * 50)
```
完成后会输出如下log:
```
▶️ 开始处理文件: train_distill_qwen3_8b_vLLMlogit.jsonl
   - 采样策略: top_k=8, num_negatives=4
  - 步骤1/3: 读取并分组数据: 593522it [00:02, 259600.82it/s]
  - 找到了 19820 个独立的 query。
  - 步骤2/3: 排序并生成采样三元组: 100%|██████████| 19820/19820 [00:04<00:00, 4160.78it/s]
  - 步骤3/3: 转换完成！
✅ (采样后) 总共生成了 623271 个三元组, 已保存到: train_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl
--------------------------------------------------
▶️ 开始处理文件: test_distill_qwen3_8b_vLLMlogit.jsonl
   - 采样策略: top_k=8, num_negatives=4
  - 步骤1/3: 读取并分组数据: 89470it [00:00, 255129.85it/s]
  - 找到了 2992 个独立的 query。
  - 步骤2/3: 排序并生成采样三元组: 100%|██████████| 2992/2992 [00:00<00:00, 4176.65it/s]
  - 步骤3/3: 转换完成！
✅ (采样后) 总共生成了 94044 个三元组, 已保存到: test_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl
--------------------------------------------------
```
并生成如下两个文件：
```
train_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl
test_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl
```
数据集内容大致如下：
```
{"query": "String isNullOrEmpty in Java?", "positive": "Java equivalent of c# String.IsNullOrEmpty() and String.IsNullOrWhiteSpace()", "negative": "isLocalHost(String hostNameOrIpAddress) in Java", "score": 6.0777692794799805}
......
```

### 第四步：训练
训练脚本train_kd_margin.py完整代码如下
```
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
```

在终端执行以下命令开始训练：
```
torchrun --nproc_per_node 1 train_kd_margin.py \
    --model_name_or_path your_path_to/bge-reranker-v2-m3 \
    --train_data your_path_to/train_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl \
    --eval_data your_path_to/test_distill_qwen3_8b_vLLMlogit_margin_sampled.jsonl \
    --output_dir output \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --eval_strategy "steps" \
    --eval_steps 10000 \
    --save_total_limit 2 \
    --bf16 \
    --load_best_model_at_end \
    --save_only_mode true \
    --metric_for_best_model "eval_loss"
```
log如下：
```
2025-07-11 13:29:41 - 训练脚本已适配 MarginMSELoss，将使用 (query, positive, negative) 三元组进行训练。
2025-07-11 13:29:41 - Use pytorch device: cuda:0
2025-07-11 13:29:42 - 正在加载数据集...
2025-07-11 13:29:43 - 训练集样本数: 623271
2025-07-11 13:29:43 - 评估集样本数: 94044
2025-07-11 13:29:43 - 训练集的一个样本: {'query': 'Java launch error selection does not contain a main type', 'positive': 'Eclipse: "selection does not contain a main type" error when main function exists', 'negative': 'Eclipse Java Launch Error: Selection does not contain a main type', 'score': 0.2500000596046448}
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
[2025-07-11 13:29:43,663] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
2025-07-11 13:29:44 - 开始模型训练...
  0%|          | 1/1218 [00:03<1:16:36,  3.78s/it]
{'loss': 9.0142, 'grad_norm': 205.3222198486328, 'learning_rate': 0.0, 'epoch': 0.0}
......
100%|█████████▉| 1217/1218 [1:09:09<00:03,  3.41s/it]
{'loss': 0.733, 'grad_norm': 5.882925510406494, 'learning_rate': 3.4572169403630083e-08, 'epoch': 1.0}
100%|██████████| 1218/1218 [1:09:10<00:00,  2.75s/it]2025-07-11 14:38:56 - Saving model checkpoint to output/checkpoint-1218
2025-07-11 14:38:56 - Save model to output/checkpoint-1218
{'loss': 0.217, 'grad_norm': 2.9422049522399902, 'learning_rate': 1.7286084701815042e-08, 'epoch': 1.0}
100%|██████████| 1218/1218 [1:11:24<00:00,  3.52s/it]
2025-07-11 14:41:09 - 训练完成，保存最终模型...
2025-07-11 14:41:09 - Save model to output
{'train_runtime': 4284.348, 'train_samples_per_second': 145.476, 'train_steps_per_second': 0.284, 'train_loss': 0.7906847017494524, 'epoch': 1.0}
2025-07-11 14:41:12 - 模型已保存至: output
```

### 第五步：评测
评测代码如下：
```
import json
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# 1. 指定您的模型路径
model_path = 'your_path_to_checkpoint/checkpoint-XXXX'
model = CrossEncoder(model_path)

# 2. 从 test.jsonl 加载数据集
samples = []
with open('your_path_to/test.jsonl', 'r') as f:
    for line in f:
        samples.append(json.loads(line))

# 3. 初始化评估器
# at_k=10 表示评估指标（如 MRR 和 NDCG）将计算到前 10 个结果
evaluator = CrossEncoderRerankingEvaluator(
    samples, 
    at_k=10, 
    name='test-evaluation',
    show_progress_bar=True
)

# 4. 运行评估
# 评估器会计算模型对 "positive" 和 "negative" 文档进行重排后的性能
# 评估指标包括 MAP, MRR@10, 和 NDCG@10
results = evaluator(model)
print("\n--- Returned Results Dictionary ---")
print(results)
```

分别将model_path改为原始模型和蒸馏后的模型权重的路径，结果如下：  
蒸馏前：
```
--- Returned Results Dictionary ---
{'test-evaluation_map': 0.47206092285098944, 'test-evaluation_mrr@10': 0.4782342861386979, 'test-evaluation_ndcg@10': 0.5472842802832706}
```

蒸馏后：
```
--- Returned Results Dictionary ---
{'test-evaluation_map': 0.5654945694133273, 'test-evaluation_mrr@10': 0.5738247761225702, 'test-evaluation_ndcg@10': 0.6386302876724329}
```

root@autodl-container-63984a89ed-fcc49336:~/autodl-tmp/finetune_bgereranker/stackover# python eval-github.py \
  --model_before_path /root/autodl-tmp/bge-reranker-v2-m3 \
  --model_after_path /root/autodl-tmp/finetune_bgereranker/stackover/output4/checkpoint-1218 \
  --dataset_path /root/autodl-tmp/dataset/stackoverflowdupquestions-reranking/test.jsonl

正在从 /root/autodl-tmp/dataset/stackoverflowdupquestions-reranking/test.jsonl 加载数据集...
加载完成！共 2992 条样本。

--- 正在加载并评估模型: /root/autodl-tmp/bge-reranker-v2-m3 ---
                                                                                                                                        
--- 正在加载并评估模型: /root/autodl-tmp/finetune_bgereranker/stackover/output4/checkpoint-1218 ---
                                                                                                                                        

==================================================
✅ 最终评估结果汇总
==================================================

【蒸馏前】模型性能:
  - MAP: 0.472061
  - MRR@10: 0.478234
  - NDCG@10: 0.547284

【蒸馏后】模型性能:
  - MAP: 0.565495
  - MRR@10: 0.573825
  - NDCG@10: 0.638630

==================================================
🚀 性能变化分析 (蒸馏后 vs. 蒸馏前)
==================================================
指标 [MAP]:
  - 绝对提升: +0.093434
  - 相对提升: +19.79% ↑
指标 [MRR@10]:
  - 绝对提升: +0.095590
  - 相对提升: +19.99% ↑
指标 [NDCG@10]:
  - 绝对提升: +0.091346
  - 相对提升: +16.69% ↑

评估完成！✨