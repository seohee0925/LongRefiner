from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None


SCRIPT_DIR = Path(__file__).resolve().parent
LONGREFINER_ROOT = SCRIPT_DIR.parents[1]
RAG_ROOT = LONGREFINER_ROOT.parent
DEFAULT_INPUT = RAG_ROOT / "Loong" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = LONGREFINER_ROOT / "outputs" / "loong"

GEN_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's task directly. "
    "Follow the requested answer format exactly, and do not add unrelated commentary."
)


def load_env(path: Path | None) -> None:
    if load_dotenv is None:
        return
    if path and path.exists():
        load_dotenv(path)


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def env_int(*names: str, default: int) -> int:
    return int(env_first(*names, default=str(default)))


def env_float(*names: str, default: float) -> float:
    return float(env_first(*names, default=str(default)))


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, LONGREFINER_ROOT, RAG_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id") or row.get("id")
            if sample_id is not None:
                done.add(str(sample_id))
    return done


def parse_set_filter(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def slugify(value: str) -> str:
    value = value.strip().split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "run"


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def split_loong_docs(docs: Any) -> list[dict[str, str]]:
    def normalize_contents(value: Any, index: int) -> dict[str, str]:
        if isinstance(value, dict):
            contents = value.get("contents") or value.get("content") or stringify(value)
        else:
            contents = str(value)
        contents = contents.strip() or "(empty)"
        if "\n" not in contents:
            contents = f"Document {index + 1}\n{contents}"
        return {"contents": contents}

    if isinstance(docs, list):
        return [normalize_contents(doc, index) for index, doc in enumerate(docs)]

    text = str(docs or "")
    title_pattern = re.compile(r"<标题起始符>(.*?)<标题终止符>", re.DOTALL)
    matches = list(title_pattern.finditer(text))
    if not matches:
        return [normalize_contents(text, 0)]

    documents: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if title or body:
            documents.append({"contents": f"{title}\n{body}".strip()})
    return documents or [normalize_contents(text, 0)]


def build_refiner_question(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", ""))
    instruction = str(sample.get("instruction", ""))
    return question if not instruction else f"{question}\n\n[Instruction]\n{instruction}"


def build_loong_prompt(sample: dict[str, Any], docs: str) -> str:
    instruction = sample.get("instruction", "")
    question = sample.get("question", "")
    template = sample.get("prompt_template")
    if isinstance(template, str) and template:
        try:
            return template.format(docs=docs, instruction=instruction, question=question)
        except KeyError:
            pass
    return f"#Documents:\n{docs}\n\n#Instruction:\n{instruction}\n\n#Question:\n{question}"


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return dict(usage)


class PromptTruncator:
    def __init__(self, model_name: str, max_input_tokens: int, trust_remote_code: bool) -> None:
        self.max_input_tokens = max_input_tokens
        self.tokenizer = None
        if max_input_tokens > 0:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )

    def build_messages(self, prompt: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        messages = [
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if self.tokenizer is None or self.max_input_tokens <= 0:
            return messages, {}

        def rendered_len(user_prompt: str) -> int:
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return len(self.tokenizer.encode(rendered, add_special_tokens=False))

        original_tokens = rendered_len(prompt)
        if original_tokens <= self.max_input_tokens:
            return messages, {
                "original_prompt_tokens": original_tokens,
                "truncated_prompt_tokens": original_tokens,
                "openai_max_input_tokens": self.max_input_tokens,
            }

        user_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        overflow = original_tokens - self.max_input_tokens
        keep_tokens = max(1, len(user_token_ids) - overflow - 128)
        truncated_prompt = self.tokenizer.decode(user_token_ids[-keep_tokens:], skip_special_tokens=True)
        truncated_tokens = rendered_len(truncated_prompt)
        while truncated_tokens > self.max_input_tokens and keep_tokens > 1:
            keep_tokens = max(1, keep_tokens - (truncated_tokens - self.max_input_tokens) - 128)
            truncated_prompt = self.tokenizer.decode(user_token_ids[-keep_tokens:], skip_special_tokens=True)
            truncated_tokens = rendered_len(truncated_prompt)

        return [
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": truncated_prompt},
        ], {
            "original_prompt_tokens": original_tokens,
            "truncated_prompt_tokens": truncated_tokens,
            "openai_max_input_tokens": self.max_input_tokens,
        }


class LongRefinerRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.longrefiner_vllm_root.strip():
            vllm_root = resolve_path(Path(args.longrefiner_vllm_root))
            if vllm_root.exists() and str(vllm_root) not in sys.path:
                sys.path.insert(0, str(vllm_root))
        if str(LONGREFINER_ROOT) not in sys.path:
            sys.path.insert(0, str(LONGREFINER_ROOT))

        from longrefiner import LongRefiner

        self.args = args
        self.refiner = LongRefiner(
            base_model_path=args.longrefiner_base_model_path,
            query_analysis_module_lora_path=args.longrefiner_query_analysis_lora_path,
            doc_structuring_module_lora_path=args.longrefiner_doc_structuring_lora_path,
            global_selection_module_lora_path=args.longrefiner_global_selection_lora_path,
            score_model_name=args.longrefiner_score_model_name,
            score_model_path=args.longrefiner_score_model_path,
            max_model_len=args.longrefiner_max_model_len,
            tensor_parallel_size=args.longrefiner_tensor_parallel_size,
            gpu_memory_utilization=args.longrefiner_gpu_memory_utilization,
            dtype=args.longrefiner_dtype,
            enforce_eager=args.longrefiner_enforce_eager,
        )

    def refine(self, sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        question = build_refiner_question(sample)
        document_list = split_loong_docs(sample.get("docs", ""))
        start = time.perf_counter()
        if self.args.longrefiner_quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                refined = self.refiner.run(question, document_list, budget=self.args.longrefiner_budget)
        else:
            refined = self.refiner.run(question, document_list, budget=self.args.longrefiner_budget)
        elapsed = round(time.perf_counter() - start, 6)

        refined_chunks = [str(chunk) for chunk in refined if str(chunk).strip()]
        refined_docs = "\n\n".join(refined_chunks).strip() or str(sample.get("docs", ""))
        return refined_docs, {
            "refine_seconds": elapsed,
            "original_doc_count": len(document_list),
            "refined_chunk_count": len(refined_chunks),
            "budget": self.args.longrefiner_budget,
        }


def generate_one(
    *,
    client: OpenAI,
    truncator: PromptTruncator,
    refiner: LongRefinerRunner,
    sample: dict[str, Any],
    args: argparse.Namespace,
    selected_index: int,
) -> dict[str, Any]:
    refined_docs, refine_metadata = refiner.refine(sample)
    prompt = build_loong_prompt(sample, refined_docs)
    messages, prompt_usage = truncator.build_messages(prompt)

    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": args.llm_model,
        "messages": messages,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        kwargs["top_p"] = args.top_p
    if args.disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(**kwargs)
    elapsed = round(time.perf_counter() - start, 6)
    content = response.choices[0].message.content or ""

    row = {
        "id": sample.get("id"),
        "sample_id": sample.get("id"),
        "selected_index": selected_index,
        "set": sample.get("set"),
        "type": sample.get("type"),
        "level": sample.get("level"),
        "language": sample.get("language"),
        "length": sample.get("length"),
        "question": sample.get("question"),
        "instruction": sample.get("instruction"),
        "answer": sample.get("answer"),
        "doc": sample.get("doc"),
        "prompt_template": sample.get("prompt_template"),
        "prompt": prompt,
        "refined_docs": refined_docs,
        "longrefiner": True,
        "run_name": args.run_name,
        "generation_model": args.llm_model,
        "generation_base_url": args.llm_base_url,
        "generation_system_prompt": GEN_SYSTEM_PROMPT,
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p if args.temperature > 0 else None,
            "max_new_tokens": args.max_new_tokens,
            **prompt_usage,
        },
        "refiner_params": {
            "base_model_path": args.longrefiner_base_model_path,
            "query_analysis_module_lora_path": args.longrefiner_query_analysis_lora_path,
            "doc_structuring_module_lora_path": args.longrefiner_doc_structuring_lora_path,
            "global_selection_module_lora_path": args.longrefiner_global_selection_lora_path,
            "score_model_name": args.longrefiner_score_model_name,
            "score_model_path": args.longrefiner_score_model_path,
            "max_model_len": args.longrefiner_max_model_len,
            "tensor_parallel_size": args.longrefiner_tensor_parallel_size,
            "gpu_memory_utilization": args.longrefiner_gpu_memory_utilization,
            "dtype": args.longrefiner_dtype,
            "enforce_eager": args.longrefiner_enforce_eager,
            **refine_metadata,
        },
        "timing": {"generation_seconds": elapsed},
        "generate_response": content,
    }
    usage = usage_dict(response)
    if usage:
        row["generation_usage"] = usage
    return row


def run(args: argparse.Namespace) -> Path:
    samples = read_jsonl(args.input_path)
    if args.sets:
        samples = [sample for sample in samples if str(sample.get("set")) in args.sets]

    done = set() if args.force else load_done_ids(args.predictions_path)
    pending = [
        (index, sample)
        for index, sample in enumerate(samples)
        if str(sample.get("id")) not in done
    ]
    if args.max_samples is not None:
        pending = pending[: args.max_samples]

    client = OpenAI(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        timeout=args.request_timeout,
    )
    truncator = PromptTruncator(args.llm_model, args.openai_max_input_tokens, args.trust_remote_code)
    refiner = LongRefinerRunner(args)

    for index, sample in tqdm(pending, desc="longrefiner+qwen35"):
        row = generate_one(
            client=client,
            truncator=truncator,
            refiner=refiner,
            sample=sample,
            args=args,
            selected_index=index,
        )
        append_jsonl(args.predictions_path, row)
    return args.predictions_path


def parse_args() -> argparse.Namespace:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=SCRIPT_DIR / "qwen35_longrefiner.env")
    env_args, _ = env_parser.parse_known_args()
    load_env(env_args.env_file)

    parser = argparse.ArgumentParser(description="Run LongRefiner on Loong, then generate with Qwen3.5-27B.")
    parser.add_argument("--env-file", type=Path, default=env_args.env_file)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=env_first("RUN_NAME", default="qwen35_27b_longrefiner"))
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sets", default=env_first("LOONG_SETS", default=""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=env_float("REQUEST_TIMEOUT_SECONDS", default=3600.0))

    parser.add_argument("--llm-base-url", default=env_first("LLM_BASE_URL", default="http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm-api-key", default=env_first("LLM_API_KEY", default="EMPTY"))
    parser.add_argument("--llm-model", default=env_first("LLM_MODEL", default="Qwen/Qwen3.5-27B"))
    parser.add_argument("--temperature", type=float, default=env_float("LLM_TEMPERATURE", default=0.0))
    parser.add_argument("--top-p", type=float, default=env_float("LLM_TOP_P", default=0.95))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("LLM_MAX_NEW_TOKENS", default=4096))
    parser.add_argument("--openai-max-input-tokens", type=int, default=env_int("OPENAI_MAX_INPUT_TOKENS", default=32768))
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=env_first("DISABLE_THINKING", default="1").lower() in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=env_first("TRUST_REMOTE_CODE", default="1").lower() not in {"0", "false", "no"},
    )

    parser.add_argument("--longrefiner-vllm-root", default=env_first("LONGREFINER_VLLM_ROOT", default="/workspace/vllm-qwen35-cu128"))
    parser.add_argument("--longrefiner-base-model-path", default=env_first("LONGREFINER_BASE_MODEL_PATH", default="Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--longrefiner-query-analysis-lora-path", default=env_first("LONGREFINER_QUERY_ANALYSIS_LORA_PATH", default="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"))
    parser.add_argument("--longrefiner-doc-structuring-lora-path", default=env_first("LONGREFINER_DOC_STRUCTURING_LORA_PATH", default="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"))
    parser.add_argument("--longrefiner-global-selection-lora-path", default=env_first("LONGREFINER_GLOBAL_SELECTION_LORA_PATH", default="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"))
    parser.add_argument("--longrefiner-score-model-name", default=env_first("LONGREFINER_SCORE_MODEL_NAME", default="bge-reranker-v2-m3"))
    parser.add_argument("--longrefiner-score-model-path", default=env_first("LONGREFINER_SCORE_MODEL_PATH", default="BAAI/bge-reranker-v2-m3"))
    parser.add_argument("--longrefiner-max-model-len", type=int, default=env_int("LONGREFINER_MAX_MODEL_LEN", default=25000))
    parser.add_argument("--longrefiner-tensor-parallel-size", type=int, default=env_int("LONGREFINER_TENSOR_PARALLEL_SIZE", default=1))
    parser.add_argument("--longrefiner-gpu-memory-utilization", type=float, default=env_float("LONGREFINER_GPU_MEMORY_UTILIZATION", default=0.7))
    parser.add_argument("--longrefiner-dtype", default=env_first("LONGREFINER_DTYPE", default="auto"))
    parser.add_argument(
        "--longrefiner-enforce-eager",
        action="store_true",
        default=env_first("LONGREFINER_ENFORCE_EAGER", default="0").lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--longrefiner-budget", type=int, default=env_int("LONGREFINER_BUDGET", default=2048))
    parser.add_argument(
        "--longrefiner-quiet",
        action="store_true",
        default=env_first("LONGREFINER_QUIET", default="1").lower() in {"1", "true", "yes"},
    )

    args = parser.parse_args()
    args.input_path = resolve_path(args.input_path)
    args.output_dir = resolve_path(args.output_dir)
    args.sets = parse_set_filter(args.sets)
    args.run_dir = args.output_dir / args.run_name
    args.predictions_path = (
        resolve_path(args.predictions_path)
        if args.predictions_path
        else args.run_dir / "predictions.jsonl"
    )
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    return args


def main() -> None:
    args = parse_args()
    print(f"Input: {args.input_path}", flush=True)
    print(f"Predictions: {args.predictions_path}", flush=True)
    print(f"Run name: {args.run_name}", flush=True)
    print(f"Generation endpoint: {args.llm_base_url}", flush=True)
    print(f"Generation model: {args.llm_model}", flush=True)
    print(f"LongRefiner base: {args.longrefiner_base_model_path}", flush=True)
    print(f"LongRefiner budget: {args.longrefiner_budget}", flush=True)
    run(args)
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
