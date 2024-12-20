
python opensora/eval/geneval/evaluate_images.py \
    opensora/eval/geneval/geneval_outputs \
    --outfile opensora/eval/geneval/results.jsonl \
    --model-path opensora/eval/geneval/detector

# python opensora/eval/geneval/summary_scores.py opensora/eval/geneval/results.jsonl
