# -*- coding: utf-8 -*-
"""
基于 output_with_summary.parquet 为每个子结论生成子问题：仅使用 score==1 且 split_score==1 且 summary_score==1 的样本。
对 summary_steps 中除最后一步外的每个子结论，列出其相关 conditions 与结论，调用 API 生成「答案为该结论」的问题（放在 \\boxed{} 中），
用 verifier 的 extract_solution 提取后写入 summary_steps 对应步骤的 question 字段；全部成功则 questions_score=1，否则为 0。
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from api import batch_get_chat_api
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, prepare_examples, save_output_parquet
    from verifier_with_vllm import extract_solution
except Exception as error:
    raise error


SUB_QUESTION_PROMPT = """You are given a set of conditions (facts or prior conclusions) that form the context, and a conclusion derived from them.

Conditions (all relevant conditions used to derive the conclusion):
{conditions_block}

Conclusion: {conclusion}

Your task: Generate exactly one natural-language question such that the answer to that question is exactly the conclusion given above. The question should be self-contained and clearly have the conclusion as its answer.

Put your generated question inside \\boxed{{}}."""


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
    """post_fun 根据 task 的 _ex_idx/_step_idx 把提取出的 question 写回 examples 的 summary_steps。"""

    def post_fun(task: Dict[str, Any], reply: str) -> None:
        ex_idx = task.get("_ex_idx", -1)
        step_idx = task.get("_step_idx", -1)
        if ex_idx < 0 or step_idx < 0 or ex_idx >= len(examples):
            return
        example = examples[ex_idx]
        steps = _get_summary_steps_from_example(example)
        if step_idx >= len(steps):
            return
        question = extract_solution(reply) if reply else None
        if question is None:
            question = ""
        else:
            question = question.strip()
        steps[step_idx]["question"] = question
        example["summary_steps"] = json.dumps(steps, ensure_ascii=False)

    return post_fun


def _set_questions_score(examples: List[Dict[str, Any]]) -> None:
    """对每个 example：若 summary_steps 中除最后一步外每步都有非空 question，则 questions_score=1，否则 0。"""
    for example in examples:
        steps = _get_summary_steps_from_example(example)
        if not steps:
            example["questions_score"] = 0
            continue
        sub_steps = steps[:-1]
        if not sub_steps:
            example["questions_score"] = 1
            continue
        all_ok = all((s.get("question") or "").strip() for s in sub_steps)
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

    def _score_ok(ex):
        s = ex.get("score")
        ss = ex.get("split_score")
        sum_s = ex.get("summary_score")
        return (s == 1 or s == 1.0) and (ss == 1 or ss == 1.0) and (sum_s == 1 or sum_s == 1.0)

    examples = [ex for ex in examples if _score_ok(ex)]
    if not examples:
        logger.warning("No examples with score==1 and split_score==1 and summary_score==1. Exit.")
        return
    logger.info(f"Processing {len(examples)} examples with score==1, split_score==1, summary_score==1")

    # 为每个子结论（除最后一步）生成一条 API 任务
    tasks = _build_tasks(examples)
    if not tasks:
        logger.warning("No sub-conclusion tasks (need at least 2 steps per example). Exit.")
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
