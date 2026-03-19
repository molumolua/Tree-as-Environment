# -*- coding: utf-8 -*-
"""
基于 output_with_summary.parquet 为每个子结论生成子问题：仅使用 score==1 且 split_score==1 且 summary_score==1 的样本，
且过滤掉 answer_type 为 Multiple Choice、Boolean、Other 的题目。
对 summary_steps 中除最后一步外的每个子结论，列出其相关 conditions 与结论，调用 API 生成 (question, answer, answer_type) 的 JSON，
解析后写回 summary_steps；answer 可带单位（如 m、cal），answer_type 须为允许的枚举之一。
整题的 question、answer、answer_type 会填入 summary_steps 的最后一个 step。
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from api import batch_get_chat_api
    from json_tools import _fix_invalid_json_escapes
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, prepare_examples, save_output_parquet
except Exception as error:
    raise error


# 过滤掉这些 answer_type 的题目，不为其生成子问题
FILTER_OUT_ANSWER_TYPES = {"Multiple Choice", "Boolean", "Other"}

# 写回 steps 的 answer_type 必须是以下之一（答案可带单位，如 m、cal）
ALLOWED_ANSWER_TYPES = {
    "Float", "Integer", "Expression", "String", "List",
    "Fraction", "Percentage", "Matrix", "Boolean"
}

SUB_QUESTION_PROMPT = """You are given a set of conditions (facts or prior conclusions) that form the context, and a conclusion derived from them.

Conditions (all relevant conditions used to derive the conclusion):
{conditions_block}

Conclusion: {conclusion}

Your task: Turn this conclusion into a (question, answer, answer_type) structure.
- question: exactly one natural-language question such that the answer to that question is the conclusion given above. The question should be self-contained.
- answer: the answer to that question. **Keep the answer as short as possible**: use a number, a short phrase, or at most one brief sentence. Do NOT write long paragraphs or lengthy explanations. You may include units (e.g. m, cal) when appropriate.
- answer_type: must be exactly one of: Float, Integer, Expression, String, List, Fraction, Percentage, Matrix, Boolean.

Output a single JSON object in this format, with no other text:
```json
{{"question": "your question here", "answer": "answer here", "answer_type": "answer_type here"}}
```
"""


def _normalize_answer_type(raw: Any) -> str:
    """将 answer_type 规范为 ALLOWED_ANSWER_TYPES 之一，否则默认 String。"""
    if raw is None:
        return "String"
    s = str(raw).strip()
    if s in ALLOWED_ANSWER_TYPES:
        return s
    return "String"


def _parse_sub_qa_reply(reply: str) -> Optional[Dict[str, Any]]:
    """
    从 API 回复中解析出 {"question", "answer", "answer_type"}。
    支持裸 JSON 或 ```json ... ``` 代码块；解析前用 _fix_invalid_json_escapes 修复 LaTeX 等非法转义；answer_type 会规范到 ALLOWED_ANSWER_TYPES。
    """
    if not reply or not isinstance(reply, str):
        return None
    text = reply.strip()
    # 尝试提取 ```json ... ``` 或 ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1).strip()
    # 尝试找 {...}
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        text = obj_match.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = json.loads(_fix_invalid_json_escapes(text))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    question = obj.get("question")
    answer = obj.get("answer")
    answer_type = obj.get("answer_type")
    if question is None and answer is None:
        return None
    question = (str(question) or "").strip()
    answer = (str(answer) or "").strip() if answer is not None else ""
    answer_type = _normalize_answer_type(answer_type)
    return {"question": question, "answer": answer, "answer_type": answer_type}


def _get_question_and_conditions_from_example(example: Dict[str, Any]) -> Tuple[str, List[str]]:
    """从 question_and_condition 解析出 question 和 conditions 列表。"""
    qc = example.get("question_and_condition")
    if qc is None:
        return "", []
    if isinstance(qc, str):
        try:
            obj = json.loads(qc)
        except json.JSONDecodeError:
            return "", []
    elif isinstance(qc, dict):
        obj = qc
    else:
        return "", []
    question = obj.get("question")
    conds = obj.get("conditions")
    if not question or not isinstance(question, str):
        question = ""
    if not conds or not isinstance(conds, list):
        conds = []
    return question, [c for c in conds if isinstance(c, str)]


def _get_summary_steps_from_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 example 的 summary_steps 解析出步骤列表（返回可修改的 list of dict）。"""
    raw = example.get("summary_steps")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [dict(s) for s in raw]
    if isinstance(raw, str):
        try:
            steps = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(steps, list):
            return [dict(s) for s in steps]
    return []


def _build_tasks(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将每个 sample 展开为多条「子结论 → 子问题」任务（排除最后一步）。
    每条 task 含 _ex_idx, _step_idx, conditions, conclusion，供 pre_fun 与 post_fun 使用。
    """
    tasks: List[Dict[str, Any]] = []
    for ex_idx, example in enumerate(examples):
        steps = _get_summary_steps_from_example(example)
        _,now_conditions = _get_question_and_conditions_from_example(example)
        # 只对除最后一步外的子结论生成问题
        for step_idx in range(max(0, len(steps) - 1)):
            step = steps[step_idx]
            conds = step.get("conditions") or []

            for cond in conds:
                if cond not in now_conditions:
                    now_conditions.append(cond)
            conclusion = (step.get("conclusion") or "").strip()
            if not conclusion:
                continue
            task = {
                "_ex_idx": ex_idx,
                "_step_idx": step_idx,
                "conditions": now_conditions,
                "conclusion": conclusion,
            }
            tasks.append(task)
    return tasks


def pre_fun(task: Dict[str, Any]) -> str:
    """用 task 的 conditions 与 conclusion 拼出单条子问题的 prompt。"""
    conditions_block = "\n".join(f"- {c}" for c in task.get("conditions") or []) or "(none)"
    return SUB_QUESTION_PROMPT.format(
        conditions_block=conditions_block,
        conclusion=task.get("conclusion", ""),
    )


def _make_post_fun(examples: List[Dict[str, Any]]):
    """post_fun 根据 task 的 _ex_idx/_step_idx 把解析出的 question/answer/answer_type 写回 examples 的 summary_steps。"""

    def post_fun(task: Dict[str, Any], reply: str) -> None:
        ex_idx = task.get("_ex_idx", -1)
        step_idx = task.get("_step_idx", -1)
        if ex_idx < 0 or step_idx < 0 or ex_idx >= len(examples):
            return
        example = examples[ex_idx]
        steps = _get_summary_steps_from_example(example)
        if step_idx >= len(steps):
            return
        parsed = _parse_sub_qa_reply(reply)
        if parsed is None:
            steps[step_idx]["question"] = ""
            steps[step_idx]["answer"] = ""
            steps[step_idx]["answer_type"] = "String"
        else:
            steps[step_idx]["question"] = parsed["question"]
            steps[step_idx]["answer"] = parsed["answer"]
            steps[step_idx]["answer_type"] = parsed["answer_type"]
        example["summary_steps"] = json.dumps(steps, ensure_ascii=False)

    return post_fun


def _fill_last_step_with_full_qa(examples: List[Dict[str, Any]]) -> None:
    """把每个 example 的整题的 question、answer、answer_type 填入 summary_steps 的最后一个 step。"""
    for example in examples:
        steps = _get_summary_steps_from_example(example)
        if not steps:
            continue
        full_question, _ = _get_question_and_conditions_from_example(example)
        full_answer = example.get("answer")
        if full_answer is None:
            full_answer = ""
        full_answer = str(full_answer).strip() if full_answer else ""
        full_answer_type = _normalize_answer_type(example.get("answer_type"))
        last = steps[-1]
        last["question"] = full_question
        last["answer"] = full_answer
        last["answer_type"] = full_answer_type
        example["summary_steps"] = json.dumps(steps, ensure_ascii=False)


def _step_qa_valid(step: Dict[str, Any]) -> bool:
    """检查该步的 question、answer、answer_type 是否有效：非空且 answer_type 在允许列表中。"""
    q = (step.get("question") or "").strip()
    a = (step.get("answer") or "").strip()
    at = (step.get("answer_type") or "").strip()
    return bool(q) and bool(a) and at in ALLOWED_ANSWER_TYPES


def _set_questions_score(examples: List[Dict[str, Any]]) -> None:
    """对每个 example：若 summary_steps 中除最后一步外每步都有有效的 question、answer、answer_type，则 questions_score=1，否则 0。"""
    for example in examples:
        steps = _get_summary_steps_from_example(example)
        if not steps:
            example["questions_score"] = 0
            continue
        sub_steps = steps[:-1]
        if not sub_steps:
            example["questions_score"] = 1
            continue
        all_ok = all(_step_qa_valid(s) for s in sub_steps)
        example["questions_score"] = 1 if all_ok else 0


def main():
    parser = argparse.ArgumentParser(
        description="读取 output_with_summary.parquet，仅取 score==1 & split_score==1 & summary_score==1，为每个子结论生成子问题并写回 summary_steps"
    )
    parser.add_argument("--load_type", type=str, default="parquet")
    parser.add_argument("--load_dir", type=str, default="./save", help="数据所在目录")
    parser.add_argument(
        "--file_glob",
        type=str,
        default="output_with_summary.parquet",
        help="输入文件名（或 glob），默认 output_with_summary.parquet",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"])
    parser.add_argument("--start_problem_idx", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name", type=str, default="output_with_sub_question.parquet")
    parser.add_argument("--save_meta_name", type=str, default="output_with_sub_question_meta.json")
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

    def _score_ok(ex):
        s = ex.get("score")
        ss = ex.get("split_score")
        sum_s = ex.get("summary_score")
        return (s == 1 or s == 1.0) and (ss == 1 or ss == 1.0) and (sum_s == 1 or sum_s == 1.0)

    examples = [ex for ex in examples if _score_ok(ex)]
    if not examples:
        logger.warning("No examples with score==1 and split_score==1 and summary_score==1. Exit.")
        return

    # 过滤 answer_type 为 Multiple Choice、Boolean、Other 的题目
    def _answer_type_ok(ex):
        at = ex.get("answer_type")
        if at is None:
            return True
        at_str = str(at).strip()
        return at_str not in FILTER_OUT_ANSWER_TYPES

    n_before_filter = len(examples)
    examples = [ex for ex in examples if _answer_type_ok(ex)]
    logger.info(
        f"After filtering answer_type not in {FILTER_OUT_ANSWER_TYPES}: "
        f"{len(examples)}/{n_before_filter} examples. Processing."
    )
    if not examples:
        logger.warning("No examples left after answer_type filter. Exit.")
        return

    # 为每个子结论（除最后一步）生成一条 API 任务
    tasks = _build_tasks(examples)
    if not tasks:
        logger.warning("No sub-conclusion tasks (need at least 2 steps per example). Exit.")
        # _fill_last_step_with_full_qa(examples)
        _set_questions_score(examples)
        save_output_parquet(
            output_problems=examples,
            save_dir_path=save_dir_path,
            logger=logger,
            save_name=args.save_name,
            meta_name=args.save_meta_name,
        )
        return

    logger.info(f"Total tasks (one per sub-conclusion): {len(tasks)}")
    post_fun = _make_post_fun(examples)

    total = len(tasks)
    n_batches = (total + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        start = b * args.batch_size
        end = min(start + args.batch_size, total)
        batch = tasks[start:end]
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

    _fill_last_step_with_full_qa(examples)
    _set_questions_score(examples)
    save_output_parquet(
        output_problems=examples,
        save_dir_path=save_dir_path,
        logger=logger,
        save_name=args.save_name,
        meta_name=args.save_meta_name,
    )

    n_ok = sum(1 for ex in examples if ex.get("questions_score") == 1)
    logger.info(f"Questions score valid: {n_ok}/{len(examples)}. Saved to {save_dir_path / args.save_name}")


if __name__ == "__main__":
    main()
