# evaluation.py
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator

def load_samples(file_path):
    """加载数据集样本"""
    print(f"正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    print(f"成功加载 {len(samples)} 条样本")
    return samples

def evaluate(model_path, dataset_path):
    """评估模型在指定数据集上的表现"""
    print(f"\n正在加载模型: {model_path}")
    model = SentenceTransformer(model_path)
    samples = load_samples(dataset_path)
    
    print("开始评估...")
    evaluator = RerankingEvaluator(samples, show_progress_bar=True, name='')
    results = evaluator(model)
    
    # 简化结果键名
    return {
        'map': results['map'],
        'mrr@10': results['mrr@10'],
        'ndcg@10': results['ndcg@10']
    }

def print_results_table(title, results_before, results_after):
    """打印结果对比表格"""
    print("\n" + "=" * 60)
    print(f"{title} 性能对比")
    print('=' * 60)
    print(f"{'指标':<10}{'蒸馏前':>12}{'蒸馏后':>12}{'绝对变化':>12}{'相对变化(%)':>12}")
    print("-" * 60)
    
    for metric in ['map', 'mrr@10', 'ndcg@10']:
        before = results_before[metric]
        after = results_after[metric]
        abs_change = after - before
        rel_change = (abs_change / before) * 100 if before != 0 else 0
        
        print(f"{metric:<10}{before:>12.4f}{after:>12.4f}{abs_change:>+12.4f}{rel_change:>+12.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比模型蒸馏前后的性能")
    parser.add_argument("--model_before", required=True, help="蒸馏前的模型路径")
    parser.add_argument("--model_after", required=True, help="蒸馏后的模型路径")
    parser.add_argument("--in_domain_dataset", required=True, help="领域内测试集路径")
    parser.add_argument("--out_domain_dataset", required=True, help="领域外测试集路径")
    
    args = parser.parse_args()

    print("=" * 60)
    print("模型性能对比评估")
    print("=" * 60)
    
    # 领域内评测
    print("\n" + "-" * 30)
    print("领域内数据集评估")
    print("-" * 30)
    in_before = evaluate(args.model_before, args.in_domain_dataset)
    in_after = evaluate(args.model_after, args.in_domain_dataset)
    
    # 领域外评测
    print("\n" + "-" * 30)
    print("领域外数据集评估")
    print("-" * 30)
    out_before = evaluate(args.model_before, args.out_domain_dataset)
    out_after = evaluate(args.model_after, args.out_domain_dataset)
    
    # 打印结果表格
    print_results_table("领域内数据集", in_before, in_after)
    print_results_table("领域外数据集", out_before, out_after)
    print("\n评估完成")