# -*- coding: utf-8 -*-
"""
从 output_with_sub_question.parquet 中筛选训练样本：
- score、split_score、summary_score、questions_score 均为 1
- answer_type 不是 "Multiple Choice"
并增加一列 deleted，值全为 []。
"""
from pathlib import Path
import argparse
import pandas as pd


def _score_equals_one(val) -> bool:
    """兼容 int 1 与 float 1.0。"""
    if val is None:
        return False
    return val == 1 or val == 1.0


def main():
    parser = argparse.ArgumentParser(
        description="从 output_with_sub_question.parquet 筛选 score/split_score/summary_score/questions_score 均为 1 且 answer_type 非 Multiple Choice，并添加 deleted 列"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="save/output_with_sub_question.parquet",
        help="输入 parquet 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="save/filtered_train.parquet",
        help="输出 parquet 路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    df = pd.read_parquet(input_path)
    n_before = len(df)

    # 四个 score 均为 1
    mask_score = (
        df["score"].map(_score_equals_one)
        & df["split_score"].map(_score_equals_one)
        & df["summary_score"].map(_score_equals_one)
        & df["questions_score"].map(_score_equals_one)
    )
    # answer_type 不是 "Multiple Choice"（大小写按需可再放宽）
    mask_not_mc = df["answer_type"].astype(str).str.strip() != "Multiple Choice"
    df = df.loc[mask_score & mask_not_mc].copy()

    # 新增列 deleted，每行均为 []
    df["deleted"] = [[] for _ in range(len(df))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"筛选前: {n_before} 条，筛选后: {len(df)} 条，已保存: {output_path}")


if __name__ == "__main__":
    main()
