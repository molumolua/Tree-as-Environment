# -*- coding: utf-8 -*-
"""
将 problem（或 question 字段）用模型拆分为 question 与 conditions，
输出 JSON 结构，并校验格式后写入 split_score，保存为新 parquet。
默认输入：save/output_with_score_1k.parquet，仅使用 score==1 的样本。
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from api import batch_get_chat_api
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, prepare_examples, save_output_parquet
    from json_tools import _fix_invalid_json_escapes
except Exception as error:
    raise error


SPLIT_PROMPT = """You are a math/algorithm problem analysis assistant. Split the following problem statement into two parts:

1. **question**: The core question to be answered (only the "what to compute/find" question or goal).
2. **conditions**: All given conditions, constraints, definitions, or known information from the problem, each as a separate item.

Requirements:
- Output only one valid JSON object. Do not output any other explanation or markdown.
- The JSON format must be exactly:
```json
{{"question": "the question in the problem", "conditions": ["condition 1", "condition 2", ...]}}
```

Problem text:
---
{problem}
---
"""


def pre_fun(example: Dict[str, Any]) -> str:
    """用 problem 或 question 字段作为待拆分的原文。"""
    raw = example.get("problem") or example.get("question", "")
    return SPLIT_PROMPT.format(problem=raw.strip())




def _extract_json_from_reply(reply: str) -> Dict[str, Any] | None:
    """从模型回复中尝试解析出 JSON 对象。会先修复 LaTeX 等导致的非法 \\v \\l \\r 等转义。"""
    if not reply or not isinstance(reply, str):
        return None
    text = reply.strip()
    # 去掉可能的 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    # 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 修复字符串内非法反斜杠（如 \vec \langle \rangle）后再解析
    fixed = _fix_invalid_json_escapes(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # 尝试找 {...} 块再解析
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    segment = text[start : i + 1]
                    try:
                        return json.loads(segment)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(_fix_invalid_json_escapes(segment))
                        except json.JSONDecodeError:
                            break
    return None


def validate_question_and_condition(obj: Any) -> bool:
    """
    检查解析结果是否符合预期结构：
    - 存在 "question" 且为非空字符串
    - 存在 "conditions" 且为列表，且每个元素为非空字符串
    """
    if not isinstance(obj, dict):
        return False
    q = obj.get("question")
    if not isinstance(q, str) or not q.strip():
        return False
    conds = obj.get("conditions")
    if not isinstance(conds, list):
        return False
    for c in conds:
        if not isinstance(c, str) or not c.strip():
            return False
    return True


def post_fun(example: Dict[str, Any], reply: str) -> None:
    """解析回复为 JSON，写入 question_and_condition（JSON 字符串），并设置 split_score。"""
    parsed = _extract_json_from_reply(reply)
    example["split_raw_reply"] = reply
    if parsed is None:
        example["question_and_condition"] = None
        example["split_score"] = 0
        return
    example["question_and_condition"] = json.dumps(parsed, ensure_ascii=False)
    example["split_score"] = 1 if validate_question_and_condition(parsed) else 0


def main():
    parser = argparse.ArgumentParser(
        description="读取带 score 的 parquet，仅取 score==1，用模型将 problem/question 拆成 question+conditions，校验后写 split_score 并保存"
    )
    parser.add_argument("--load_type", type=str, default="parquet")
    parser.add_argument(
        "--load_dir",
        type=str,
        default="./save",
        help="数据所在目录",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        default="output_with_score_1k.parquet",
        help="输入文件名（或 glob），默认 output_with_score_1k.parquet",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"])
    parser.add_argument("--start_problem_idx", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name", type=str, default="output_with_split.parquet")
    parser.add_argument("--save_meta_name", type=str, default="output_with_split_meta.json")
    parser.add_argument("--model", type=str, default="glm-4.7")
    parser.add_argument("--n_processes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inner_max_try", type=int, default=1)

    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Args: {vars(args)}")

    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    dataset = load_and_prepare_dataset(
        load_type=args.load_type,
        load_dir=Path(args.load_dir),
        split=args.split,
        file_glob=args.file_glob,
        drop_list=[],
        logger=logger,
    )
    examples = prepare_examples(
        ds=dataset,
        start_idx=args.start_problem_idx,
        max_rows=args.max_rows,
        logger=logger,
        extract_code=False,
    )

    if not examples:
        logger.info("No examples loaded. Exit.")
        return

    # 只保留 score == 1 的样本
    examples = [ex for ex in examples if ex.get("score") == 1 or ex.get("score") == 1.0]
    if not examples:
        logger.warning("No examples with score==1. Exit.")
        return
    logger.info(f"Processing {len(examples)} examples with score==1")

    total = len(examples)
    n_batches = (total + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        start = b * args.batch_size
        end = min(start + args.batch_size, total)
        batch = examples[start:end]
        logger.info(f"Batch {b + 1}/{n_batches} | size={len(batch)}")
        batch_get_chat_api(
            examples=batch,
            eng=args.model,
            pre_fun=pre_fun,
            post_fun=post_fun,
            logger=logger,
            n_processes=args.n_processes,
            temperature=args.temperature,
            timeout=args.timeout,
            max_try=args.inner_max_try,
        )
        save_output_parquet(
            output_problems=examples,
            save_dir_path=save_dir_path,
            logger=logger,
            save_name=args.save_name,
            meta_name=args.save_meta_name,
        )

    n_ok = sum(1 for ex in examples if ex.get("split_score") == 1)
    logger.info(f"Split score valid: {n_ok}/{len(examples)}. Saved to {save_dir_path / args.save_name}")


if __name__ == "__main__":
    main()
