# -*- coding: utf-8 -*-
import os
import json
import argparse
import logging
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# --- 1. 日志和参数配置 ---

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用 Qwen3-Reranker-8B 模型和 vLLM 为数据集生成蒸馏分数。")
    
    # 将原脚本中的所有可配置项都改为命令行参数，并设置默认值
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Qwen3-Reranker-8B',
                        help='Qwen3-Reranker-8B 模型所在的路径。')
    parser.add_argument('--input_files', nargs='+', default=['train.jsonl', 'test.jsonl'],
                        help='需要处理的输入文件路径列表，可以提供一个或多个文件。')
    parser.add_argument('--output_suffix', type=str, default='_distill_qwen3_8b_vLLMlogit',
                        help='添加到输出文件名中的后缀。')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='推理时使用的批处理大小，请根据您的 GPU 显存进行调整。')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='模型支持的最大序列长度。')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='vLLM 使用的 GPU 显存比例。')
    parser.add_argument('--task_instruction', type=str, default='Given a web search query, retrieve relevant passages that answer the query',
                        help='用于 Reranker 模型的指令文本。')

    return parser.parse_args()


# --- 2. 模型加载和工具函数 (逻辑不变) ---

def initialize_model_and_tokenizer(args: argparse.Namespace):
    """根据参数加载 vLLM 模型和 Tokenizer"""
    logging.info("Initializing model and tokenizer...")
    number_of_gpu = torch.cuda.device_count()
    logging.info(f"Detected {number_of_gpu} GPUs.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading model with vLLM...")
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=number_of_gpu,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,  # 对 rerank 这种固定模板的任务有很好的加速效果
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    logging.info("Model loaded successfully.")

    # 定义固定的 token 和采样参数
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    return model, tokenizer, sampling_params


def format_and_tokenize_inputs(
    tokenizer: AutoTokenizer,
    queries: List[str],
    docs: List[str],
    instruction: str,
    max_length: int
) -> List[TokensPrompt]:
    """使用 apply_chat_template 格式化并 tokenize 输入 (逻辑不变)"""
    messages = []
    for query, doc in zip(queries, docs):
        message = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"}
        ]
        messages.append(message)
    
    # 固定的模板后缀
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    templated_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    processed_messages = [ele[:max_length] + suffix_tokens for ele in templated_messages]
    final_messages = [TokensPrompt(prompt_token_ids=ele) for ele in processed_messages]
    return final_messages


def compute_scores_vllm(
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    batch_queries: List[str],
    batch_docs: List[str],
    instruction: str,
    max_length: int
) -> List[float]:
    """计算分数的函数 (逻辑不变)"""
    tokenized_batch = format_and_tokenize_inputs(tokenizer, batch_queries, batch_docs, instruction, max_length)
    outputs = model.generate(tokenized_batch, sampling_params=sampling_params, use_tqdm=False)

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    scores = []
    for output in outputs:
        final_logprobs = output.outputs[0].logprobs[-1]

        true_logprob = final_logprobs.get(true_token, -10.0)
        if not isinstance(true_logprob, float): # vLLM 可能返回 Logprob 对象
            true_logprob = true_logprob.logprob

        false_logprob = final_logprobs.get(false_token, -10.0)
        if not isinstance(false_logprob, float): # vLLM 可能返回 Logprob 对象
            false_logprob = false_logprob.logprob

        # logit = log(prob_yes) - log(prob_no)
        logit_diff = true_logprob - false_logprob
        scores.append(logit_diff)
        
    return scores


# --- 3. 数据集生成主流程 ---

def process_file(
    input_path: str,
    output_path: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    args: argparse.Namespace
):
    """处理单个输入文件，并生成对应的输出文件"""
    logging.info(f"Start processing file: {input_path}")
    logging.info(f"Output will be saved to: {output_path}")

    # 加载并重构数据集
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}. Skipping this file.")
        return

    logging.info("Reformatting data...")
    reformatted_data = []
    for item in tqdm(original_data, desc="Reformatting data", leave=False):
        query = item['query']
        for passage in item.get('positive', []):
            reformatted_data.append({'query': query, 'passage': passage})
        for passage in item.get('negative', []):
            reformatted_data.append({'query': query, 'passage': passage})

    logging.info(f"Original items: {len(original_data)}, Reformatted to {len(reformatted_data)} query-passage pairs.")
    logging.info(f"Starting to generate distillation data for {len(reformatted_data)} examples...")

    # 分批处理并写入文件
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(reformatted_data), args.batch_size), desc=f"Generating for {os.path.basename(input_path)}"):
            batch_data = reformatted_data[i:i + args.batch_size]
            queries = [sample['query'] for sample in batch_data]
            documents = [sample['passage'] for sample in batch_data]

            scores_batch = compute_scores_vllm(
                model, tokenizer, sampling_params, queries, documents, args.task_instruction, args.max_model_len
            )

            for j in range(len(batch_data)):
                distill_sample = {
                    "query": queries[j],
                    "passage": documents[j],
                    "score": scores_batch[j]
                }
                out_f.write(json.dumps(distill_sample, ensure_ascii=False) + '\n')
    
    logging.info(f"Distillation for {input_path} successfully generated and saved to: {output_path}")


# --- 4. 主函数入口 ---

def main():
    """主执行函数"""
    args = parse_arguments()
    logging.info(f"Script started with the following arguments: {vars(args)}")

    model, tokenizer, sampling_params = initialize_model_and_tokenizer(args)

    for input_file in args.input_files:
        print("\n" + "="*80)
        base_name, extension = os.path.splitext(input_file)
        output_file = f"{base_name}{args.output_suffix}{extension}"
        
        process_file(input_file, output_file, model, tokenizer, sampling_params, args)
        print("="*80)

    # 清理资源
    destroy_model_parallel()
    logging.info("\nAll files have been processed. Model resources destroyed. Script finished.")


if __name__ == "__main__":
    main()