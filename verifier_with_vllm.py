# -*- coding: utf-8 -*-
"""
Verifier：读取 api_get_answer 输出的 reply / answer，用本地 vLLM 模型判分，并写入 score 字段。
- reply：学生输出（或从中 extract 的最终答案）作为 student_answer
- answer：标准答案作为 ground_truth
- 使用 vLLM 加载 /Users/molu/Tree-as-Environment/model，不做 batch_get_chat_api。
"""
import os

# 避免多份 OpenMP 运行时导致 vLLM 初始化失败 (OMP Error #15)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERIFIER_LOGGING_LEVEL", "INFO"))

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str | None:
    """从文本中提取最后一个 \\boxed{...} 的内容。"""
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str | None:
    """用多种候选模式尝试提取最终答案。"""
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[: -len(stop_word)].strip()
    return last_match


def extract_solution(solution_str: str) -> str | None:
    """优先从 \\boxed{} 取答案，否则用 Final Answer 等模式。"""
    if not solution_str or not isinstance(solution_str, str):
        return None
    boxed = extract_last_boxed(solution_str)
    if boxed:
        return boxed
    return extract_last_final_answer(solution_str)


def load_data(path: Path, load_type: str) -> List[Dict[str, Any]]:
    """从 parquet 或 json/jsonl 加载带 reply / answer 的数据。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if load_type == "parquet" or str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict("records")
    if load_type == "json" or str(path).endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if load_type == "jsonl" or str(path).endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported load_type or extension: {path}")


def save_data(rows: List[Dict[str, Any]], out_path: Path, save_type: str) -> None:
    """将带 score 的数据保存为 parquet 或 json/jsonl。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if save_type == "parquet" or str(out_path).endswith(".parquet"):
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
    elif save_type == "json" or str(out_path).endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    elif save_type == "jsonl" or str(out_path).endswith(".jsonl"):
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported save_type or extension: {out_path}")
    logger.info(f"Saved {len(rows)} rows to {out_path}")


def run_verifier(
    model_path: str,
    data_path: Path,
    load_type: str,
    save_path: Path,
    save_type: str,
    batch_size: int = 8,
    max_question_len: int = 2048,
    max_ground_truth_len: int = 1024,
    max_student_answer_len: int = 1024,
    gpu_memory_utilization: float = 0.5,
    use_extract_solution: bool = True,
) -> List[Dict[str, Any]]:
    """
    使用 vLLM 加载 model_path 的模型，对 data 中每条用 reply 与 answer 做验证，写回 score。
    """
    rows = load_data(data_path, load_type)
    if not rows:
        logger.warning("No rows loaded. Exit.")
        return []

    # 检查必要字段
    if not all("reply" in r and "answer" in r for r in rows):
        missing = [i for i, r in enumerate(rows) if "reply" not in r or "answer" not in r]
        raise KeyError(f"Rows {missing[:5]}{'...' if len(missing) > 5 else ''} missing 'reply' or 'answer'.")

    logger.info(f"Loaded {len(rows)} rows from {data_path}")
    logger.info(f"Loading vLLM model: {model_path}")
    # 说明：在 Mac（无 CUDA）上标准 vLLM 会使用 CPU，gpu_memory_utilization 无效。
    # 若要在 Apple Silicon 上用 GPU，需使用 vllm-metal：https://github.com/vllm-project/vllm-metal
    if os.uname().sysname == "Darwin":
        logger.warning(
            "Running on macOS: standard vLLM uses CPU only (no CUDA). "
            "For GPU acceleration, use vllm-metal (MLX/Metal backend)."
        )
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    # 准备 student_answer：可选从 reply 里 extract_solution
    def get_student_answer(r: Dict[str, Any]) -> str:
        reply = r.get("reply") or ""
        if not isinstance(reply, str):
            reply = str(reply)
        if use_extract_solution:
            sol = extract_solution(reply)
            if sol is not None:
                return sol[:max_student_answer_len]
            return (reply[-max_student_answer_len:] if len(reply) > max_student_answer_len else reply) or "No Answer"
        return reply[:max_student_answer_len] or "No Answer"

    all_scores: List[float] = []
    all_verifications: List[str] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        questions = [str(r.get("question", ""))[:max_question_len] for r in batch]
        ground_truths = [str(r.get("answer", ""))[:max_ground_truth_len] for r in batch]
        student_answers = [get_student_answer(r) for r in batch]

        prompts = [
            VERIFIER_PROMPT_TEMPLATE.format(
                question=q,
                ground_truth=gt,
                student_answer=sa,
            )
            for q, gt, sa in zip(questions, ground_truths, student_answers)
        ]

        outputs = llm.generate(prompts, sampling_params)
        for out, r in zip(outputs, batch):
            text = (out.outputs[0].text or "").strip()
            all_verifications.append(text)
            if VERIFIER_PASS_TAG in text:
                score = 1.0
            else:
                score = 0.0
            # 若启用 extract 且未提取到答案，可扣分（与参考实现一致）
            if use_extract_solution and extract_solution(str(r.get("reply") or "")) is None and r.get("reply"):
                score -= 0.5
            all_scores.append(max(0.0, score))

    for i, r in enumerate(rows):
        r["score"] = all_scores[i]
        r["verification"] = all_verifications[i]

    if save_path:
        save_data(rows, save_path, save_type)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Verifier: 读入带 reply/answer 的数据，用 vLLM 本地模型判分，写入 score 并保存"
    )
    parser.add_argument("--model_path", type=str, default="/Users/molu/Tree-as-Environment/model")
    parser.add_argument("--data_path", type=str, required=True, help="输入文件路径（parquet/json/jsonl）")
    parser.add_argument("--load_type", type=str, default="parquet", choices=["parquet", "json", "jsonl"])
    parser.add_argument("--save_path", type=str, default=None, help="输出路径，默认在输入同目录下加 _with_score")
    parser.add_argument("--save_type", type=str, default=None, choices=["parquet", "json", "jsonl"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_question_len", type=int, default=2048)
    parser.add_argument("--max_ground_truth_len", type=int, default=1024)
    parser.add_argument("--max_student_answer_len", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--no_extract_solution", action="store_true", help="不从 reply 中提取答案，直接用整段 reply 作为 student_answer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_path)
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        stem = data_path.stem
        ext = data_path.suffix
        save_path = data_path.parent / f"{stem}_with_score{ext}"
    save_type = args.save_type or (args.load_type if args.load_type else "parquet")

    rows = run_verifier(
        model_path=args.model_path,
        data_path=data_path,
        load_type=args.load_type,
        save_path=save_path,
        save_type=save_type,
        batch_size=args.batch_size,
        max_question_len=args.max_question_len,
        max_ground_truth_len=args.max_ground_truth_len,
        max_student_answer_len=args.max_student_answer_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_extract_solution=not args.no_extract_solution,
    )
    total = len(rows)
    passed = sum(1 for r in rows if r.get("score", 0) >= 1.0)
    logger.info(f"Done. Total={total}, Passed={passed}, PassRate={passed / total * 100:.1f}%" if total else "Done. No rows.")


if __name__ == "__main__":
    main()
