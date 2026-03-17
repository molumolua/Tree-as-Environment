# -*- coding: utf-8 -*-
"""
对 reply 做步骤化总结：输入为 output_with_split.parquet，仅使用 score==1 且 split_score==1 的样本。
调用模型将 reply 总结为「步骤列表」JSON，校验结构及 conditions 约束后写入 summary_score，保存为新 parquet。
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from api import batch_get_chat_api
    from json_tools import _fix_invalid_json_escapes
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, prepare_examples, save_output_parquet
except Exception as error:
    raise error


SUMMARY_PROMPT = """You are given a question, a set of initial conditions, and a "reply" that is a step-by-step derivation answering the question using those conditions.

Your task: summarize this reply by splitting it into a sequence of steps. Each step must be represented as a JSON object with exactly two keys:
- **conditions**: a list of strings. Each string is either one of the initial conditions given below, or the exact "conclusion" text from a previous step. No paraphrasing: every condition must be a verbatim copy.
- **conclusion**: a single string, the conclusion or result obtained in this step.

Rules:
1. Each step may use the given initial conditions and/or conclusions from previous steps as its "conditions".
2. Every string in "conditions" must be an exact, character-for-character copy of either (a) one of the initial conditions listed below, or (b) the "conclusion" of some earlier step in your output. Do not invent or rephrase any condition.
3. The conclusion of the **last** step must directly answer or solve the question posed above; it should be the final answer to the question.
4. Output only one JSON array of such step objects, wrapped in a fenced code block. Use this format:

```json
[
  {{"conditions": ["condition 1 used in this step", "..."], "conclusion": "conclusion of this step"}},
  {{"conditions": ["...", "..."], "conclusion": "..."}},
  ...
]
```

---
Question: {question}

Initial conditions (use these exact strings when referring to them in steps):
{conditions_block}

Reply to summarize (step-by-step derivation):
---
{reply}
---
"""


def _get_question_and_conditions_from_example(example: Dict[str, Any]) -> List[str]:
    """从 question_and_condition 解析出 conditions 列表。"""
    qc = example.get("question_and_condition")
    if qc is None:
        return "",[]
    if isinstance(qc, str):
        try:
            obj = json.loads(qc)
        except json.JSONDecodeError:
            return "",[]
    elif isinstance(qc, dict):
        obj = qc
    else:
        return "",[]
    question = obj.get("question")
    conds = obj.get("conditions")
    if not question or not isinstance(question, str):
        return "",[]
    if not conds or not isinstance(conds, list):
        return "",[]
    return question, [c for c in conds if isinstance(c, str)]


def pre_fun(example: Dict[str, Any]) -> str:
    """用 question、conditions、reply 拼出 prompt。"""
    question,conditions = _get_question_and_conditions_from_example(example)
    conditions_block = "\n".join(f"- {c}" for c in conditions) if conditions else "(none)"
    reply = example.get("reply", "")
    return SUMMARY_PROMPT.format(
        question=question,
        conditions_block=conditions_block,
        reply=reply,
    )


def _extract_json_list_from_reply(reply: str) -> List[Any] | None:
    """从模型回复中解析出 ```json ... ``` 中的 JSON，期望为 list。解析失败时用 json_tools 修复非法转义后再试。"""
    if not reply or not isinstance(reply, str):
        return None
    text = reply.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed = _fix_invalid_json_escapes(text)
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    return parsed


def validate_summary_steps(
    steps: List[Any],
    initial_conditions: List[str],
) -> bool:
    """
    检查步骤列表是否符合预期：
    - 结构：非空列表，每项为 dict，且含 "conditions"（list of str）与 "conclusion"（str）。
    - 约束：每个 step 的 conditions 中每一条，必须是 initial_conditions 或前面某步 conclusion 的完全复制（逐字一致）。
    """
    if not steps or not isinstance(steps, list):
        return False
    allowed: set[str] = set(s for s in initial_conditions if isinstance(s, str))
    for step in steps:
        if not isinstance(step, dict):
            return False
        conds = step.get("conditions")
        concl = step.get("conclusion")
        if not isinstance(conds, list):
            return False
        if not isinstance(concl, str) or not concl.strip():
            return False
        for c in conds:
            if not isinstance(c, str):
                return False
            # 必须与某条 allowed 完全一致（不做归一化，要求完全复制）
            if c not in allowed:
                return False
        allowed.add(concl)
    return True


def post_fun(
    example: Dict[str, Any],
    reply: str,
    initial_conditions: List[str],
) -> None:
    """将原始回复写入 summary_raw_reply，解析为步骤列表，校验后写入 summary_steps（JSON 字符串）和 summary_score。"""
    example["summary_raw_reply"] = reply
    example["summary_steps"] = None
    example["summary_score"] = 0
    steps = _extract_json_list_from_reply(reply)
    if steps is None:
        return
    if not validate_summary_steps(steps, initial_conditions):
        return
    example["summary_steps"] = json.dumps(steps, ensure_ascii=False)
    example["summary_score"] = 1


def main():
    parser = argparse.ArgumentParser(
        description="读取 output_with_split.parquet，仅取 score==1 且 split_score==1，对 reply 做步骤总结并校验后保存"
    )
    parser.add_argument("--load_type", type=str, default="parquet")
    parser.add_argument("--load_dir", type=str, default="./save", help="数据所在目录")
    parser.add_argument(
        "--file_glob",
        type=str,
        default="output_with_split.parquet",
        help="输入文件名（或 glob），默认 output_with_split.parquet",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"])
    parser.add_argument("--start_problem_idx", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name", type=str, default="output_with_summary.parquet")
    parser.add_argument("--save_meta_name", type=str, default="output_with_summary_meta.json")
    parser.add_argument("--model", type=str, default="glm-4.7")
    parser.add_argument("--n_processes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
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

    # 只保留 score==1 且 split_score==1 的样本
    examples = [
        ex
        for ex in examples
        if ex.get("score") == 1 and (ex.get("split_score") == 1 or ex.get("split_score") == 1.0)
    ]
    if not examples:
        logger.warning("No examples with score==1 and split_score==1. Exit.")
        return
    logger.info(f"Processing {len(examples)} examples with score==1 and split_score==1")

    total = len(examples)
    n_batches = (total + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        start = b * args.batch_size
        end = min(start + args.batch_size, total)
        batch = examples[start:end]
        logger.info(f"Batch {b + 1}/{n_batches} | size={len(batch)}")

        def post_fun_bind(example: Dict[str, Any], reply: str) -> None:
            _, initial_conditions = _get_question_and_conditions_from_example(example)
            post_fun(example, reply, initial_conditions)

        batch_get_chat_api(
            examples=batch,
            eng=args.model,
            pre_fun=pre_fun,
            post_fun=post_fun_bind,
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

    n_ok = sum(1 for ex in examples if ex.get("summary_score") == 1)
    logger.info(f"Summary score valid: {n_ok}/{len(examples)}. Saved to {save_dir_path / args.save_name}")


if __name__ == "__main__":
    main()
