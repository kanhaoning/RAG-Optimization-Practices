python evaluation.py \
  --model_before /root/autodl-tmp/bge-m3 \
  --model_after /root/autodl-tmp/finetune_bgem3/scidocs-reranking/output_kldiv/checkpoint-953 \
  --in_domain_dataset /root/autodl-tmp/dataset/scidocs-reranking/test.jsonl \
  --out_domain_dataset /root/autodl-tmp/dataset/stackoverflowdupquestions-reranking/test.jsonl