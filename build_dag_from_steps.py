# -*- coding: utf-8 -*-
"""
从 output_with_sub_question.parquet 的 summary_steps 构建有向无环图（DAG），
并找到「删除后仍保持只有一个顶点」的最小 step 集合（以这些 steps 包含的总边数衡量）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from datasets import Dataset

# 每个 step: {"conditions": [...], "conclusion": "..."}
# 边：每个 condition 指向该 step 的 conclusion → 该 step 包含的边数 = len(conditions)

GLOBAL_INDEX = 0
QUESTION_PROMPT_TEMPLATE='''
Given Conditions:
{conditions_block}

Question:
{question}
'''
def steps_to_dag(
    steps: List[Dict[str, Any]],
    initial_conditions: List[str],
    deleted: List[List[int]]
) -> Tuple[Set[int], Dict[int, Set[int]], Dict[int, int], Dict[int, int], Dict[int, Set[int]]]:
    """
    将 steps 组织成 DAG，并计算权重与前置节点。

    - 顶点：step 下标 0..n-1。
    - 边：若 step i 的 conclusion 出现在 step j 的 conditions 中，则边 (i, j)，即 i 是 j 的依赖，j 是 i 的子节点。
    - 权重：无子节点则 weight=1；有子节点则 weight = sum(子节点权重)。（按逆拓扑序计算）
    - 前置：无子节点则前置=空；否则 前置 = 子节点 ∪ 各子节点的前置。（即该点的所有后代集合）

    返回：
    - roots: 无入边的顶点（唯一根）
    - succ: succ[i] = step i 的子节点集合（谁用到了 i 的 conclusion）
    - edge_count: step i 的边数 = len(conditions_i)
    - weight: weight[i] 如上
    - pred_set: pred_set[i] = 节点 i 的前置节点集合（子节点 + 子节点的前置）
    """
    n = len(steps)
    flat_deleted = [item for sublist in deleted for item in sublist]
    concl_to_idx: Dict[str, int] = {}
    for i, s in enumerate(steps):
        c = (s.get("conclusion") or "").strip()
        if c and c not in concl_to_idx and i not in flat_deleted:
            concl_to_idx[c] = i

    init_set = set(initial_conditions) if initial_conditions else set()
    in_degree = [0] * n
    succ: Dict[int, Set[int]] = {i: set() for i in range(n)}


    weight: Dict[int, int] = {}
    pred_set: Dict[int, Set[int]] = {}

    for i, s in enumerate(steps):
        if i in flat_deleted:
            continue
        if i not in weight:
            weight[i] = 1
        if i not in pred_set:
            pred_set[i] = set[int]()
        
        conds = s.get("conditions") or []
        for c in conds:
            if c in init_set:
                weight[i] +=1
                continue
            if c in concl_to_idx:
                j = concl_to_idx[c]
                if j < i and j not in flat_deleted:
                    weight[i] += weight[j]
                    pred_set[i].add(j)
                    pred_set[i] = pred_set[i] | pred_set[j]
                    succ[i].add(j)
                    in_degree[j] += 1

    roots = {i for i in range(n) if in_degree[i] == 0 and i not in flat_deleted}


    return roots, succ, weight, pred_set


def min_deleted_steps_one_vertex(
    steps: List[Dict[str, Any]],
    initial_conditions: List[str],
    deleted: List[List[int]] = [],
) -> Tuple[List[int], int | None]:
    """
    按给定规则决定删除哪些 step，使图变为「只有一个顶点」。

    1. 找满足 权重=子节点数量 的节点；若有多个，取边数最少的那一个（记为步骤2节点）。
    2. 找最终唯一父节点（根节点 root）。
    3. 在根的儿节点中找权重最大的那个（记为步骤4节点）。
    4. 比较： (根权重 - 步骤2节点权重) vs 步骤4节点权重。
    5. 若前者更大：只删除步骤2节点；否则只保留步骤4及其前置节点，其余全删。
    返回：(被删除的 step 下标列表, 删除后剩下的「最后一个父节点」的下标，无法确定时为 None)。
    """
    if not steps:
        return [], None
    if len(steps) == 1:
        return [], None
    flat_deleted = [item for sublist in deleted for item in sublist]

    roots, succ, weight, pred_set = steps_to_dag(steps, initial_conditions,deleted)

    # 1. 无前序节点的点
    cands = [i for i in range(len(steps)) if i not in flat_deleted and len(pred_set[i]) == 0]
    if not cands:
        return [], None
    

    # 2. 边数最少的那一个
    step2_node = min(cands, key=lambda i: weight[i])
    # 3. 唯一根（最终父节点）
    if len(roots) != 1:
        return [], None
    root = next(iter(roots))

    # 4. 根的儿子中权重最大的
    children_of_root = succ[root]
    if not children_of_root:
        return [], None
    step4_node = max(children_of_root, key=lambda i: weight[i])

    # 5. 比较
    val_parent_minus_step2 = weight[root] - weight[step2_node]
    val_step4 = weight[step4_node]

    # 6. 更新：决定删除集合与删除后的最后一个父节点
    if val_parent_minus_step2 >= val_step4:
        # 只删步骤2节点，剩下图中根仍是唯一父节点
        return [step2_node], root
    else:
        keep = {step4_node} | pred_set[step4_node]
        to_delete = [i for i in range(len(steps)) if i not in flat_deleted and i not in keep]
        # 保留的是 step4 及其前置，其中 step4 为新的唯一父节点（根）
        return to_delete, step4_node


def _get_summary_steps_from_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _get_initial_conditions_from_example(example: Dict[str, Any]) -> List[str]:
    qc = example.get("question_and_condition")
    if qc is None:
        return []
    if isinstance(qc, str):
        try:
            obj = json.loads(qc)
        except json.JSONDecodeError:
            return []
    elif isinstance(qc, dict):
        obj = qc
    else:
        return []
    conds = obj.get("conditions")
    if not isinstance(conds, list):
        return []
    return [c for c in conds if isinstance(c, str)]



def get_problem_from_example(example):
    global GLOBAL_INDEX
    deleted = example.get("deleted", [])
    flat_deleted = [item for sublist in deleted for item in sublist]

    steps = _get_summary_steps_from_example(example)
    conditions = _get_initial_conditions_from_example(example)
    question = ""
    ground_truth =""
    n = len(steps)
    end_index = -1
    for idx in range(n-1,-1,-1):
        if idx not in flat_deleted:
            question = str(steps[idx].get("question")) or ""
            ground_truth = str(steps[idx].get("answer")) or ""
            end_index = idx
            break

    for idx in range(end_index):
        if idx in flat_deleted:
            conditions.append(str(steps[idx].get("conclusion")) or "")
    
    problem_prompt = QUESTION_PROMPT_TEMPLATE.format(conditions_block= "\n".join(f"- {c}" for c in conditions) if conditions else "(none)",\
                    question=question if len(question) > 0 else "(none)")
    GLOBAL_INDEX += 1
    if len(question) > 0 and len(ground_truth) > 0 :
        return {
            "id":example['id'],
            "data_source": "dag",
            "prompt": [{
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },{
                "role": "user",
                "content": problem_prompt,
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': "train",
                'index': GLOBAL_INDEX,
                'answer': ground_truth,
                "question": problem_prompt
            }
        }
    else:
        # raise ValueError(f"ERROR! in example:{example}")
        return None
def delete_and_update_example(example):
    steps = _get_summary_steps_from_example(example)
    initial_conditions = _get_initial_conditions_from_example(example)
    deleted =  list(example.get("deleted",[]))
    to_delete, last_parent_idx = min_deleted_steps_one_vertex(steps,initial_conditions,deleted)

    if last_parent_idx == None:
        return None
    
    deleted.append(to_delete)
    example["deleted"] = deleted
    return example

def recover_and_update_example(example):
    deleted =  list(example.get("deleted",[]))
    if len(deleted) == 0:
        return None
    deleted.pop()
    example["deleted"] = deleted
    return example

def run_on_parquet(
    parquet_path: Path,
    output_parquet_path: Path | None = None,
) -> Dataset:
    """
    对 parquet 中每一行，用其 summary_steps 和 question_and_condition 的 conditions
    调用 min_deleted_steps_one_vertex，把 to_delete_indices、last_parent_index 等写入该行，
    最后将 ds 存到 output_parquet_path。若未指定则不写入磁盘。

    新增列：
    - to_delete_indices: List[int] 的 JSON 字符串
    - last_parent_index: int，无则为 -1
    - last_parent_step: dict 的 JSON 字符串，无则为 None
    - deleted_edge_count: int
    - total_edges: int
    """
    try:
        from datasets import load_dataset, Dataset
    except ImportError:
        raise ImportError("需要 datasets 库来读取 parquet，请 pip install datasets")

    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    n = len(ds)

    for row_idx in range(n):
        row = ds[row_idx]
        example = dict(row) if not isinstance(row, dict) else row
        deleted = example.get("deleted", [])
        steps = _get_summary_steps_from_example(example)
        round_num = 0
        while deleted is not None and len(deleted) < len(steps):
            round_num += 1
            example = delete_and_update_example(example)
            if example:
                problem = get_problem_from_example(example)
                print(len(steps))
                if problem:
                    prompt_content = problem["prompt"][1]["content"]
                    ground_truth = problem["reward_model"]["ground_truth"]
                    print(f"--- 样本 {row_idx + 1}/{n}  第 {round_num} 轮 ---")
                    print(prompt_content)
                    print(f"[ground_truth] {ground_truth}")
                    print()
                    deleted = example.get("deleted", [])
                else:
                    break
            else:
                break
        # break
    return ds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 parquet 的 steps 找最小删除 step 集合（保持单顶点），结果写回 parquet")
    parser.add_argument("--parquet", type=str, default="save2/filtered_train.parquet")
    parser.add_argument("--output", type=str, default=None, help="输出 parquet 路径（不指定则不写盘）")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    out_path = Path(args.output) if args.output else None
    ds = run_on_parquet(path, output_parquet_path=out_path)
    if out_path:
        print(f"已写入 {out_path}，共 {len(ds)} 条")
    else:
        print(f"处理完成，共 {len(ds)} 条（未指定 --output 故未写盘）")
