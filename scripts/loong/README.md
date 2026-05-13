# LongRefiner on Loong with Qwen3.5-27B

This runner follows the zero-shot Loong flow, but inserts LongRefiner before
generation:

1. Read `/workspace/rag/Loong/loong_process.jsonl`.
2. Split each Loong `docs` field into LongRefiner `{"contents": "title\nbody"}` documents.
3. Refine documents with the README LoRA modules.
4. Put the refined text back into the original Loong `prompt_template`.
5. Generate the final answer with `Qwen/Qwen3.5-27B`.

Run a smoke test:

```bash
cd /workspace/rag/LongRefiner
bash scripts/loong/run_with_qwen35_vllm.sh \
  --env-file scripts/loong/qwen35_longrefiner.env \
  --max-samples 1
```

Run all samples:

```bash
cd /workspace/rag/LongRefiner
bash scripts/loong/run_with_qwen35_vllm.sh \
  --env-file scripts/loong/qwen35_longrefiner.env
```

If a Qwen3.5-27B OpenAI-compatible endpoint is already running, skip the vLLM
launcher and run the Python script directly:

```bash
cd /workspace/rag/LongRefiner
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/vllm-qwen35-cu128:${PYTHONPATH:-} \
/opt/conda/bin/python scripts/loong/run_loong_qwen35.py \
  --env-file scripts/loong/qwen35_longrefiner.env \
  --max-samples 1
```

Outputs are written to:

```text
outputs/loong/qwen35_27b_longrefiner/predictions.jsonl
```
