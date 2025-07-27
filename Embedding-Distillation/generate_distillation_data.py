# generate_distillation_data.py

import json
import torch
from vllm import LLM
import logging
from tqdm import tqdm
from itertools import product
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# --- 1. 日志和参数配置 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用 Qwen3-Embedding-8B 模型为 DistillKLDivLoss 生成蒸馏数据。")
    parser.add_argument('--teacher_model_path', type=str, required=True,
                        help='教师模型 (Qwen3-Embedding-8B) 所在的路径。')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='需要处理的原始数据文件路径列表 (例如: validation.jsonl test.jsonl)。')
    parser.add_argument('--output_suffix', type=str, default='_kldiv_distill',
                        help='添加到输出文件名中的后缀。')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='vLLM 编码时使用的批处理大小，请根据您的 GPU 显存进行调整。')
    parser.add_argument('--tensor_parallel_size', type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", 1)),
                    help='vLLM 使用的张量并行大小。')
    return parser.parse_args()


# --- 2. 模型加载和工具函数 ---

def initialize_model(args: argparse.Namespace) -> LLM:
    """根据参数加载 vLLM 模型"""
    logging.info(f"正在从以下路径加载教师模型: {args.teacher_model_path}")
    try:
        model = LLM(
            model=args.teacher_model_path,
            trust_remote_code=True,
            task="embed",
            tensor_parallel_size=args.tensor_parallel_size
        )
        logging.info("教师模型加载成功。")
        return model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise

def similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """计算两个向量的余弦相似度"""
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# --- 3. 数据集生成主流程 ---

def process_file(
    input_path: str,
    output_path: str,
    model: LLM,
    args: argparse.Namespace
):
    """处理单个输入文件，并生成对应的蒸馏数据文件"""
    logging.info(f"开始处理文件: {input_path}")
    logging.info(f"输出将保存到: {output_path}")

    # 1. 收集所有不重复的句子和三元组
    logging.info(f"正在从 {input_path} 读取数据并收集不重复的句子...")
    unique_texts = set()
    triplets = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query = data.get('query')
                positives = data.get('positive')
                negatives = data.get('negative')

                if not all([query, positives, negatives]):
                    continue
                
                unique_texts.add(query)
                # DistillKLDivLoss 可以处理多个负样本，但为保持与原逻辑一致，此处仍使用笛卡尔积
                # 更高效的做法是每个 query 对应1个 positive 和所有 negatives
                for pos_item, neg_item in product(positives, negatives):
                    unique_texts.add(pos_item)
                    unique_texts.add(neg_item)
                    triplets.append({'query': query, 'positive': pos_item, 'negative': neg_item})
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {input_path}")
        return

    input_texts = list(unique_texts)
    logging.info(f"共找到 {len(input_texts)} 个不重复的句子需要编码。")
    logging.info(f"共生成 {len(triplets)} 个 (query, pos, neg) 三元组。")

    # 2. 使用教师模型批量生成向量
    logging.info("正在使用教师模型生成向量...")
    all_embeddings = []
    # 使用 model.embed() 批量处理
    outputs = model.embed(input_texts)
    all_embeddings = [torch.tensor(o.outputs.embedding, dtype=torch.float32) for o in outputs]
    logging.info("向量生成完毕。")

    # 3. 创建 文本 -> 向量 的映射字典
    text_to_embedding = {text: emb for text, emb in zip(input_texts, all_embeddings)}

    # 4. 计算相似度并写入最终的蒸馏文件
    logging.info(f"正在计算相似度并写入蒸馏数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for triplet in tqdm(triplets, desc=f"为 {os.path.basename(input_path)} 计算标签"):
            q_text = triplet['query']
            p_text = triplet['positive']
            n_text = triplet['negative']
            
            emb_q = text_to_embedding.get(q_text)
            emb_p = text_to_embedding.get(p_text)
            emb_n = text_to_embedding.get(n_text)

            if emb_q is None or emb_p is None or emb_n is None:
                logging.warning(f"跳过一个无法找到全部向量的三元组: {triplet}")
                continue

            sim_pos = similarity(emb_q, emb_p)
            sim_neg = similarity(emb_q, emb_n)
            
            # 为 DistillKLDivLoss 创建标签：[positive_score, negative_score]
            label = [sim_pos, sim_neg]
            
            record = {
                "query": q_text,
                "positive": p_text,
                "negative": n_text,
                "label": label  # 标签格式为分数列表
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    logging.info(f"DistillKLDivLoss 蒸馏数据为 {input_path} 生成完成！✨")


# --- 4. 主函数入口 ---

def main():
    """主执行函数"""
    args = parse_arguments()
    logging.info(f"脚本启动，参数: {vars(args)}")

    model = initialize_model(args)

    for input_file in args.input_files:
        print("\n" + "="*80)
        base_name, extension = os.path.splitext(input_file)
        output_file = f"{base_name}{args.output_suffix}{extension}"
        
        process_file(input_file, output_file, model, args)
        print("="*80)

    logging.info("\n所有文件处理完毕。脚本执行结束。")


if __name__ == "__main__":
    main()