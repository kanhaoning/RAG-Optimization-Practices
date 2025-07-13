# evaluation.py

import json
import argparse # 1. 导入 argparse 模块，用于处理命令行参数
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

def load_samples(dataset_path: str) -> list:
    """
    从 .jsonl 文件加载数据集。

    Args:
        dataset_path (str): 数据集文件的路径。

    Returns:
        list: 包含样本的列表。
    """
    samples = []
    print(f"正在从 {dataset_path} 加载数据集...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"加载完成！共 {len(samples)} 条样本。")
    return samples

def evaluate_model(model_path: str, samples: list) -> dict:
    """
    加载模型并执行评估。

    Args:
        model_path (str): CrossEncoder 模型的路径。
        samples (list): 用于评估的样本列表。

    Returns:
        dict: 包含评估结果（MAP, MRR, NDCG）的字典。
    """
    print(f"\n--- 正在加载并评估模型: {model_path} ---")
    model = CrossEncoder(model_path)
    
    evaluator = CrossEncoderRerankingEvaluator(
        samples,
        at_k=10,
        name='test-evaluation',
        show_progress_bar=True
    )
    
    return evaluator(model)

def print_results(model_name: str, results: dict):
    """
    格式化打印单个模型的评估结果。
    """
    print(f"\n【{model_name}】模型性能:")
    for key, value in results.items():
        metric_name = key.split('_')[-1] # 从 'test-evaluation_map' 中提取 'map'
        print(f"  - {metric_name.upper()}: {value:.6f}")

def compare_and_print_changes(results_before: dict, results_after: dict):
    """
    对比两个模型的结果并打印性能变化。
    """
    print("\n" + "="*50)
    print("🚀 性能变化分析 (蒸馏后 vs. 蒸馏前)")
    print("="*50)

    for key in results_before:
        metric_name = key.split('_')[-1]
        score_before = results_before[key]
        score_after = results_after[key]
        
        absolute_change = score_after - score_before
        relative_change = (absolute_change / score_before) * 100 if score_before != 0 else float('inf')
        
        change_sign = "↑" if absolute_change >= 0 else "↓"
        
        print(f"指标 [{metric_name.upper()}]:")
        print(f"  - 绝对提升: {absolute_change:+.6f}")
        print(f"  - 相对提升: {relative_change:+.2f}% {change_sign}")

# 2. 使用 if __name__ == "__main__": 结构，这是 Python 脚本的标准入口点
if __name__ == "__main__":
    # 3. 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="评估和对比模型蒸馏前后的 Reranker 性能。")
    parser.add_argument(
      '--model_before_path', 
      type=str, 
      default='/root/autodl-tmp/bge-reranker-v2-m3', 
      required=False, 
      help='蒸馏前（原始）模型的路径。'
    )
    parser.add_argument(
      '--model_after_path', 
      type=str, 
      default='output/checkpoint-1217', 
      required=False,
      help='蒸馏后（微调）模型的路径。'
    )
    parser.add_argument(
      '--dataset_path', 
      type=str, 
      default='test.jsonl', 
      required=False, 
      help='test.jsonl 数据集文件的路径。'
    )
    args = parser.parse_args()

    # 从文件中加载数据
    samples = load_samples(args.dataset_path)

    # 评估蒸馏前的模型
    results_before = evaluate_model(args.model_before_path, samples)
    
    # 评估蒸馏后的模型
    results_after = evaluate_model(args.model_after_path, samples)

    # 打印独立的结果
    print("\n\n" + "="*50)
    print("✅ 最终评估结果汇总")
    print("="*50)
    print_results("蒸馏前", results_before)
    print_results("蒸馏后", results_after)
    
    # 对比并打印性能变化
    compare_and_print_changes(results_before, results_after)

    print("\n评估完成！✨")