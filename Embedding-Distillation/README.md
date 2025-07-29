# Embedding模型知识蒸馏 (Embedding Model Knowledge Distillation)

<p align="center">
  <a href="https://pytorch.org/" target="_blank"> <img src="https://img.shields.io/badge/PyTorch-2.6-red.svg" alt="PyTorch Version"></a>
  <a href="https://www.sbert.net/" target="_blank"> <img src="https://img.shields.io/badge/Sentence--Transformers-5.0-blue.svg" alt="Sentence-Transformers Version"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-Embedding-8B" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3--embedding-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/BAAI/bge-m3" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-BGE--m3-yellow" alt="Hugging Face Model"></a>
</p>

> **本项目配有详细教程：[将向量大模型Qwen3-Embedding-8B的知识蒸馏到小模型BGE-m3上](https://www.google.com/search?q=https://your-blog-post-link-here)**
## 1\. 项目简介

本项目是 **RAG 优化实战系列** 的第二部分， **Embedding 模型的知识蒸馏**。

实践了如何将8B的SOTA向量模型（教师模型：**`Qwen/Qwen3-Embedding-8B`**）的知识，通过 **`DistillKLDivLoss`** 损失函数，高效地蒸馏到一个轻量级且广泛使用的小模型（学生模型：**`BAAI/bge-m3`**）上。这种“软标签”的蒸馏方法，旨在通过学习教师模型的完整输出分布，来提升学生模型在特定领域的排序能力，同时有效避免灾难性遗忘。

  - **教师模型**: `Qwen/Qwen3-Embedding-8B` (8B)
  - **学生模型**: `BAAI/bge-m3` (0.6B)
  - **数据集**:
      - **领域内**: `MTEB/scidocs-reranking`
      - **领域外**: `MTEB/stackoverflowdupquestions-reranking`
  - **核心技术**: `DistillKLDivLoss` 知识蒸馏
  - **推理加速**: `vLLM`

## 2\. 性能表现

通过本项目的蒸馏流程，`bge-m3` 模型在领域内数据集 `scidocs` 上的性能获得了显著提升，同时在领域外数据集上保持了较好的泛化能力。

| 指标 (Metric) | 蒸馏前 (Before) | 蒸馏后 (After) | 绝对提升 | **相对提升** |
| :--- | :---: | :---: | :---: | :---: |
| **领域内 (scidocs)** | | | | |
| **MAP** | 0.7744 | **0.8534** | +0.0790 | **+10.20%** 🚀 |
| **MRR@10** | 0.9321 | **0.9554** | +0.0233 | **+2.50%** 🚀 |
| **NDCG@10** | 0.8296 | **0.8973** | +0.0676 | **+8.15%** 🚀 |
| **领域外 (stackoverflowdupquestions)** | | | | |
| **MAP** | 0.5168 | 0.5040 | -0.0129 | -2.49% |
| **MRR@10** | 0.5240 | 0.5116 | -0.0124 | -2.37% |
| **NDCG@10** | 0.5904 | 0.5774 | -0.0129 | -2.19% |

## 3\. 复现步骤

### 3.1. 环境准备

首先，克隆本仓库并安装所需的依赖库。建议使用 `Python 3.9+`。

```bash
git clone https://github.com/kanhaoning/RAG-Optimization-Practices.git
cd RAG-Optimization-Practices/Embedding-Distillation
# 建议创建一个虚拟环境
# pip install virtualenv
# python -m venv venv
# source venv/bin/activate
pip install torch sentence-transformers==5.0.0 transformers==4.53.1 vllm==0.8.4 datasets tqdm modelscope
```

### 3.2. 下载模型与数据集

**模型下载:**
使用 `modelscope` 或 `huggingface-cli` 下载教师和学生模型。

```python
# 学生模型: BAAI/bge-m3
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='/path/to/your/models')

# 教师模型: Qwen/Qwen3-Embedding-8B
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-Embedding-8B', cache_dir='/path/to/your/models')
```

**数据集下载:**

1.  **Scidocs (领域内)**: 访问 [MTEB/scidocs-reranking](https://www.modelscope.cn/datasets/MTEB/scidocs-reranking/files)，手动下载`validation.jsonl.gz`和`test.jsonl.gz`到 `Embedding-Distillation/dataset_scidocs` 目录（直接拉取可能报错），然后解压。

2.  **Stackoverflow (领域外)**: 访问 [MTEB/stackoverflowdupquestions-reranking](https://www.modelscope.cn/datasets/MTEB/stackoverflowdupquestions-reranking/files)，手动下载`test.jsonl.gz`到 `Embedding-Distillation/dataset_stackoverflowdupquestions` 目录，然后解压。

执行以下命令解压所有数据集：

```bash
# 进入 scidocs 目录并解压
cd dataset_scidocs
gunzip validation.jsonl.gz
gunzip test.jsonl.gz
cd ..

# 进入 stackoverflowdupquestions 目录并解压
cd dataset_stackoverflowdupquestions
gunzip test.jsonl.gz
cd ..
```

### 3.3. 执行蒸馏流程

整个流程分为3个步骤，每个步骤都由一个 `.sh` 脚本驱动。请按顺序执行。

**步骤 1: 生成教师模型分数 (Soft Labels)**

使用 `vLLM` 加速 `Qwen3-Embedding-8B` 模型，为 `scidocs` 的`validation`集计算`(query, positive)`和`(query, negative)`的相似度分数，作为蒸馏用的软标签。

```bash
bash generate_distillation_data.sh
```

  - **关键参数**: 在 `generate_distillation_data.sh` 中，你需要修改 `--teacher_model_path` 为你下载的 `Qwen3-Embedding-8B` 模型路径。可以根据你的 GPU 显存大小调整 `--batch_size`。

**步骤 2: 训练学生模型**

使用上一步生成的蒸馏数据集（`validation_kldiv_distill.jsonl`），对 `bge-m3` 学生模型进行微调。

```bash
bash train.sh
```

  - **关键参数**: 在 `train.sh` 中，修改 `--student_model_name_or_path` 为你本地的 `bge-m3` 模型路径。同时，可以根据你的GPU显存和数量调整 `--per_device_train_batch_size` 和 `--nproc_per_node`。

**步骤 3: 性能评测与对比**

评估蒸馏前后的模型性能，包括领域内和领域外数据集，并生成对比表格。

```bash
bash evaluation.sh
```

  - **关键参数**: 在 `evaluation.sh` 中，确保 `--model_before` 指向原始的 `bge-m3` 模型，而 `--model_after` 指向上一步训练输出的模型checkpoint的路径（例如: `output/checkpoint-953`）。

## 4\. 代码文件说明

  - `generate_distillation_data.py`: 使用 vLLM 和教师模型（Qwen3-Embedding-8B）为数据集计算相似度分数，生成 `DistillKLDivLoss` 需要的软标签文件。
  - `train.py`: 使用 `sentence-transformers` 框架和 `DistillKLDivLoss` 进行模型蒸馏训练。
  - `evaluation.py`: 使用 `RerankingEvaluator` 评测并对比模型蒸馏前后的性能。
  - `*.sh`: 上述 Python 脚本的执行脚本，封装了所有命令行参数，方便一键执行。

## 5\. 参考与致谢

  - **论文**: [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176v3), [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers](https://arxiv.org/pdf/2010.11386)
  - **模型**: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
  - **框架**: [sentence-transformers](https://www.sbert.net), [vLLM](https://github.com/vllm-project/vllm)
  - **数据集**: [MTEB/scidocs-reranking](https://huggingface.co/datasets/mteb/scidocs-reranking), [MTEB/stackoverflowdupquestions-reranking](https://huggingface.co/datasets/mteb/stackoverflowdupquestions-reranking)