# 重排序模型知识蒸馏

<p align="center">
  <a href="https://pytorch.org/" target="_blank"> <img src="https://img.shields.io/badge/PyTorch-2.6-red.svg" alt="PyTorch Version"></a>
  <a href="https://www.sbert.net/" target="_blank"> <img src="https://img.shields.io/badge/Sentence--Transformers-5.0-blue.svg" alt="Sentence-Transformers Version"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-Reranker-8B" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3--reranker-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/BAAI/bge-reranker-v2-m3" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-BGE--Reranker-yellow" alt="Hugging Face Model"></a>
</p>

> **本项目配有详细教程：[知乎](https://zhuanlan.zhihu.com/p/1928223248396551046)**

## 1. 项目简介

本项目是 **RAG 优化实战系列** 的第一部分， **Reranker 模型的知识蒸馏**。

实践了如何将8B的重排序SOTA模型（教师模型：**Qwen3-Reranker-8B**）的知识，通过 `MarginMSE` 损失函数，高效地蒸馏到一个轻量级且广泛使用的小模型（学生模型：**BAAI/bge-reranker-v2-m3**）上。这种方法旨在不依赖成本更高的人工数据标注和大模型数据合成，显著提升小型 Reranker 模型在特定任务上的性能。

- **教师模型**: `Qwen/Qwen3-Reranker-8B`
- **学生模型**: `BAAI/bge-reranker-v2-m3`
- **数据集**: `MTEB/stackoverflowdupquestions-reranking`
- **核心技术**: `MarginMSE` 知识蒸馏

## 2. 性能表现

通过本项目的蒸馏流程，`bge-reranker-v2-m3` 模型在 `stackoverflowdupquestions-reranking` 测试集上的性能得到了全面且显著的提升。

| 指标 (Metric) | 蒸馏前 (Before) | 蒸馏后 (After) | 绝对提升 | **相对提升** |
| :--- | :---: | :---: | :---: | :---: |
| **MAP** | 0.472061 | **0.565317** | +0.093256 | **+19.76%** 🚀 |
| **MRR@10** | 0.478234 | **0.573779** | +0.095545 | **+19.98%** 🚀 |
| **NDCG@10** | 0.547284 | **0.639033** | +0.091748 | **+16.76%** 🚀 |

## 3. 复现步骤

### 3.1. 环境准备

首先，克隆本仓库并安装所需的依赖库。建议使用 `Python 3.9+`。

```bash
git clone https://github.com/kanhaoning/RAG-Optimization-Practices.git
cd RAG-Optimization-Practices/Reranker-Distillation
pip install -r requirements.txt 
# 安装核心依赖
pip install torch sentence-transformers==5.0.0 transformers==4.53.1 vllm==0.8.4
```

### 3.2. 下载模型与数据集

**模型下载:**
使用 `modelscope` 或 `huggingface-cli` 下载教师和学生模型。
```python
# bge-reranker-v2-m3
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='/path/to/your/models')

# Qwen3-Reranker-8B
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-Reranker-8B', cache_dir='/path/to/your/models')
```

**数据集下载:**
从 [MTEB/stackoverflowdupquestions-reranking](https://www.modelscope.cn/datasets/MTEB/stackoverflowdupquestions-reranking/files) 手动下载 `train.jsonl.gz` 和 `test.jsonl.gz`，然后执行以下命令解压（直接拉取可能有BUG），并粘贴到RAG-Optimization-Practices/Reranker-Distillation目录：
```bash
gunzip train.jsonl.gz
gunzip test.jsonl.gz
```

### 3.3. 执行蒸馏流程

整个流程分为4个步骤，每个步骤都由一个 `.sh` 脚本驱动。

**步骤 1: 生成教师模型 Logits**

使用 `vLLM` 加速 `Qwen3-Reranker-8B` 模型，为数据集中的每个 (query, passage) 对生成相关性分数（logits）。  
```bash
bash generate_logits.sh
```
- **关键参数**: 在 `generate_logits.sh` 中，你需要修改 `--model_path` 为你下载的 `Qwen3-Reranker-8B` 模型路径。可以根据你的 GPU 显存大小调整 `--batch_size`。

**步骤 2: 构建 MarginMSE 训练样本**

将上一步生成的分数文件转换为 `(query, positive, negative, score_diff)` 格式的三元组，用于 `MarginMSELoss` 训练。

```bash
bash create_triplets.sh
```
- 此脚本将上一步的输出转换为蒸馏所需格式，通常无需修改。

**步骤 3: 训练学生模型**

使用处理好的三元组数据，对 `bge-reranker-v2-m3` 学生模型进行微调。

```bash
bash train.sh
```
- **关键参数**: 在 `train.sh` 中，检查 `--model_name_or_path` 是否为 `bge-reranker-v2-m3` 的路径。同时，可以根据你的GPU显存调整 `--per_device_train_batch_size` ，根据卡数调整 `--nproc_per_node`。
**步骤 4: 性能评测与对比**

评估蒸馏前后的模型性能，并计算提升。

```bash
bash evaluation.sh
```
- **关键参数**: 在 evaluation.sh 中，确保 `--model_before_path` 指向原始的 `bge-reranker-v2-m3` 模型，而 `--model_after_path` 指向上一步训练输出的模型checkpoint的路径（默认为 output/checkpoint-XXX）。
## 4. 代码文件说明

- `generate_logits.py`: 使用 vLLM 和教师模型（Qwen3-Reranker-8B）为数据集打分。
- `create_triplets.py`: 将打好分的数据转换为 MarginMSE 损失函数所需的三元组格式。
- `train.py`: 使用 `sentence-transformers` 框架和 `MarginMSELoss` 进行模型蒸馏训练。
- `evaluation.py`: 使用 `CrossEncoderRerankingEvaluator` 评测并对比模型蒸馏前后的性能。
- `*.sh`: 上述 Python 脚本的执行脚本，封装了所有命令行参数。

## 5. 参考与致谢

- [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [sentence-transformers](https://www.sbert.net)
- [mteb/stackoverflowdupquestions-reranking](https://huggingface.co/datasets/mteb/stackoverflowdupquestions-reranking)
- [vLLM](https://github.com/vllm-project/vllm)

