# -*- coding: utf-8 -*-
"""
数据格式转换脚本

本脚本用于将包含 (query, passage, score) 的 .jsonl 数据集，
转换为适用于 MarginMSE 损失函数的 (query, positive, negative, score_diff) 格式。

核心功能：
1. 按 query 对数据进行分组。
2. 对每个 query 下的 passages 按 score 降序排序。
3. 使用智能采样策略生成训练/测试样本：
   - 为每个 query 选择分数最高的 `top_k` 个 passage 作为正例 (positive)。
   - 为每个选定的正例，从其后分数较低的 passage 中选择 `num_negatives` 个作为负例 (negative)。
4. 计算正例和负例的分数差 (score_diff)，并保存为新的 .jsonl 文件。

用法:
  - 使用默认参数运行 (与原始脚本等价):
    python script.py

  - 指定输入文件和自定义参数运行:
    python script.py --input_files file1.jsonl file2.jsonl --top_k 10 --num_negatives 5

  - 查看帮助:
    python script.py --help
"""
import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import logging

# --- 配置日志 ---
# 使用日志模块替代 print，更适合开源项目
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def convert_to_margin_mse_format(
    input_file_path: str,
    output_file_path: str,
    top_k: int,
    num_negatives: int
) -> int:
    """
    将 (query, passage, score) 格式的数据集转换为 (query, positive, negative, score_diff) 格式。

    通过采样策略，为每个 query 下的高分 passage（正例）匹配若干个低分 passage（负例）。

    Args:
        input_file_path (str): 输入的 .jsonl 文件路径。
        output_file_path (str): 输出转换后格式的 .jsonl 文件路径。
        top_k (int): 对于每个query，选择分数最高的k个passage作为候选正例。
        num_negatives (int): 为每个正例选择的负例数量。

    Returns:
        int: 成功生成的样本数量。
    """
    if not os.path.exists(input_file_path):
        logging.error(f"输入文件未找到: {input_file_path}")
        return 0

    logging.info(f"▶️ 开始处理文件: {input_file_path}")
    logging.info(f"  - 采样策略: top_k={top_k}, num_negatives={num_negatives}")

    # 1. 按 query 分组
    query_to_passages = defaultdict(list)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="  - 步骤 1/3: 读取并分组数据"):
                try:
                    data = json.loads(line)
                    query_to_passages[data['query']].append({
                        "passage": data['passage'],
                        "score": data['score']
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"跳过格式错误的行: {line.strip()} | 错误: {e}")
                    continue
    except Exception as e:
        logging.error(f"读取文件时发生错误 {input_file_path}: {e}")
        return 0

    logging.info(f"  - 找到了 {len(query_to_passages)} 个独立的 query。")

    # 2. 排序、采样并生成样本
    triplets_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        progress_bar = tqdm(
            query_to_passages.items(),
            desc="  - 步骤 2/3: 排序并生成样本"
        )
        for query, passages in progress_bar:
            if len(passages) < 2:
                continue

            # 按分数从高到低排序
            passages.sort(key=lambda x: x['score'], reverse=True)

            # 选择 top_k 作为正例候选
            positive_candidates = passages[:top_k]

            for i, p_pos in enumerate(positive_candidates):
                # 负例池是当前正例之后的所有 passage
                negative_pool = passages[i+1:]
                if not negative_pool:
                    continue

                # 选择 "hard negatives" (分数最接近的)
                # 您也可以在这里切换到随机采样策略:
                # import random
                # negative_samples = random.sample(negative_pool, min(len(negative_pool), num_negatives))
                negative_samples = negative_pool[:num_negatives]

                for p_neg in negative_samples:
                    # 确保 passage 内容不同 (虽然排序后通常不会相同，但作为安全检查)
                    if p_pos['passage'] == p_neg['passage']:
                        continue

                    # 分数差应为正数 (排序已保证)
                    score_diff = p_pos['score'] - p_neg['score']
                    if score_diff > 0:
                        new_record = {
                            "query": query,
                            "positive": p_pos['passage'],
                            "negative": p_neg['passage'],
                            "score": score_diff
                        }
                        f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        triplets_count += 1

    logging.info("  - 步骤 3/3: 转换完成！")
    return triplets_count


def main():
    """主函数，用于解析命令行参数并执行转换任务。"""
    parser = argparse.ArgumentParser(
        description="将 (query, passage, score) 格式的数据集转换为 Margin-MSE 格式。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息中的格式
    )

    parser.add_argument(
        '--input_files',
        nargs='+',  # 接受一个或多个输入文件
        default=[
            "train_distill_qwen3_8b_vLLMlogit.jsonl",
            "test_distill_qwen3_8b_vLLMlogit.jsonl"
        ],
        help="一个或多个待处理的 .jsonl 输入文件路径。\n默认值: %(default)s"
    )

    parser.add_argument(
        '--output_suffix',
        type=str,
        default="_margin_sampled",
        help="添加到输出文件名的后缀。\n例如 'input.jsonl' 将变为 'input_margin_sampled.jsonl'。\n默认值: %(default)s"
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=8,
        help="每个 query 选择分数最高的 K 个 passage作为正例。\n默认值: %(default)s"
    )

    parser.add_argument(
        '--num_negatives',
        type=int,
        default=4,
        help="为每个正例匹配的负例数量。\n默认值: %(default)s"
    )

    args = parser.parse_args()

    logging.info("--- 开始数据转换任务 ---")
    logging.info(f"输入文件: {args.input_files}")
    logging.info(f"参数: top_k={args.top_k}, num_negatives={args.num_negatives}")
    logging.info("-" * 50)

    total_generated = 0
    for input_file in args.input_files:
        base_name, extension = os.path.splitext(input_file)
        output_file = f"{base_name}{args.output_suffix}{extension}"

        count = convert_to_margin_mse_format(
            input_file_path=input_file,
            output_file_path=output_file,
            top_k=args.top_k,
            num_negatives=args.num_negatives
        )

        if count > 0:
            logging.info(f"✅ (采样后) 总共为 {input_file} 生成了 {count} 个样本, 已保存到: {output_file}")
            total_generated += count
        else:
            logging.warning(f"⚠️ 未能为 {input_file} 生成任何样本。请检查文件内容和格式。")
        logging.info("-" * 50)

    logging.info(f"--- 所有任务完成！总共生成了 {total_generated} 个样本。 ---")


if __name__ == '__main__':
    main()