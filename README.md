# RAG 优化实践 (RAG Optimization Practices)
<p align="center">
  <a href="https://pytorch.org/" target="_blank"> <img src="https://img.shields.io/badge/PyTorch-2.6-red.svg" alt="PyTorch Version"></a>
  <a href="https://www.sbert.net/" target="_blank"> <img src="https://img.shields.io/badge/Sentence--Transformers-5.0-blue.svg" alt="Sentence-Transformers Version"></a>
</p>

## 🚀 项目简介

本项目旨在分享 **RAG (检索增强生成) 优化** 的实践经验。

## 📂 项目结构

```
/RAG-Optimization-Practices
├── 📄 README.md                         <-- 你正在看的主页
│      
├── 📁 Reranker-Distillation/             (✅ 已完成)
│   ├── README.md                       # Reranker 蒸馏模块的详细说明
│   ├── generate_logits.sh              # 步骤1: 教师模型生成Logit分数
│   ├── create_triplets.sh              # 步骤2: 构建训练样本
│   ├── train.sh                        # 步骤3: 训练学生模型
│   └── evaluation.sh                   # 步骤4: 评测性能
│
├── 📁 Embedding-Distillation/            (✅ 已完成)
│   ├── README.md                       # Embedding 蒸馏模块的详细说明
│   ├── generate_distillation_data.sh   # 步骤1: 教师模型生成蒸馏分数
│   ├── train.sh                        # 步骤2: 训练学生模型
│   └── evaluation.sh                   # 步骤3: 评测性能
│
├── 📁 Query-Expansion-RL/                (⏳ 规划中)
└── 📁 Milvus-Optimization/               (⏳ 规划中)
```

-----

## ✅ 已完成模块

### 模块一：Reranker 知识蒸馏

本项目的第一部分，实践如何将SOTA重排序模型（教师模型：`Qwen3-Reranker-8B`）的知识地蒸馏到一个 0.6B 的轻量级模型（学生模型：`BAAI/bge-reranker-v2-m3`）上。

#### 核心成果：性能显著提升

不依赖人工标注，通过知识蒸馏，学生模型在 `stackoverflowdupquestions-reranking` 数据集上的性能获得了**近 20% 的相对提升**。

| 指标 (Metric) | 蒸馏前 (原始) | 蒸馏后 (优化) | 绝对提升 | **相对提升** |
| :--- | :---: | :---: | :---: | :---: |
| **MAP** | 0.4721 | **0.5653** | +0.0932 | **+19.76%** 🚀 |
| **MRR@10** | 0.4782 | **0.5738** | +0.0956 | **+19.98%** 🚀 |
| **NDCG@10** | 0.5473 | **0.6390** | +0.0917 | **+16.76%** 🚀 |

👉 **点击查看详细实现、代码和复现步骤: [./Reranker-Distillation/README.md](https://github.com/kanhaoning/RAG-Optimization-Practices/tree/main/Reranker-Distillation/README.md)**


### 模块二：Embedding 知识蒸馏

本项目的第二部分，实践如何将SOTA向量模型（教师模型：`Qwen3-Embedding-8B`）的知识蒸馏到一个 0.6B 的轻量级模型（学生模型：`BAAI/bge-m3`）上，旨在提升其在特定领域的排序能力。

#### 核心成果：领域内性能提升10%，且未发生灾难性遗忘

通过基于KL散度的知识蒸馏，学生模型在 `scidocs` 这一科学文档数据集上的 **MAP 指标相对提升了 10.20%**，同时在领域外数据集上的性能下降幅度控制在 2.5% 以内，缓解了灾难性遗忘。

| 指标 (Metric) | 蒸馏前 (原始) | 蒸馏后 (优化) | 绝对提升 | **相对提升** |
| :--- | :---: | :---: | :---: | :---: |
| **MAP** | 0.7744 | **0.8534** | +0.0790 | **+10.20%** 🚀 |
| **MRR@10** | 0.9321 | **0.9554** | +0.0233 | +2.50% |
| **NDCG@10** | 0.8296 | **0.8973** | +0.0676 | +8.15% |

👉 **点击查看详细实现、代码和复现步骤: [./Embedding-Distillation/README.md](https://github.com/kanhaoning/RAG-Optimization-Practices/tree/main/Embedding-Distillation/README.md)**
