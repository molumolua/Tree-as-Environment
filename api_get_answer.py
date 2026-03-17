# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from api import batch_get_chat_api
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, prepare_examples, save_output_parquet
except Exception as error:
    raise error


def pre_fun(example: Dict[str, Any]) -> str:
    """用 question 字段作为 prompt。"""
    return example.get("question", "")


def post_fun(example: Dict[str, Any], reply: str) -> None:
    """将 API 返回写入 answer 字段。"""
    example["answer"] = reply


def main():
    parser = argparse.ArgumentParser(description="读取数据，用 question 作为 prompt 调用 API，将 answer 写入并保存为 parquet")
    # 加载参数
    parser.add_argument("--load_type", type=str, default="parquet", help="json 或 parquet")
    parser.add_argument("--load_dir", type=str, default="./dataset",
                        help="数据所在目录（parquet/json 文件所在路径）")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "valid", "validation"],
                        help="加载的 split，与文件名前缀匹配，如 train-*.parquet")
    parser.add_argument("--file_glob", type=str, default=None,
                        help="可选，自定义文件名匹配，如 '*.parquet'")
    parser.add_argument("--start_problem_idx", type=int, default=0, help="起始样本下标")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="最多加载行数（None 表示全部）")
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name", type=str, default="output_with_answer.parquet")
    parser.add_argument("--save_meta_name", type=str, default="output_with_answer_meta.json")
    # 推理与并行
    parser.add_argument("--model", type=str, default="gpt-4", help="batch_get_chat_api 使用的模型")
    parser.add_argument("--n_processes", type=int, default=16, help="API 并行进程数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--timeout", type=int, default=60, help="单次请求超时（秒）")
    parser.add_argument("--think", action="store_true", default=False, help="是否开启 think 模式")
    # 批次与重试
    parser.add_argument("--batch_size", type=int, default=256, help="每批样本数")
    parser.add_argument("--inner_max_try", type=int, default=3, help="单条请求最大重试次数")

    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Args: {vars(args)}")

    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {save_dir_path}")

    # 加载数据
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

    # 只保留带 question 的样本
    examples = [ex for ex in examples if ex.get("question")]
    if not examples:
        logger.warning("No examples with 'question' field. Exit.")
        return
    logger.info(f"Processing {len(examples)} examples with 'question'")

    # 按批调用 API，post_fun 会原地写入 answer
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
            think=args.think,
        )

    # 保存为 parquet（含 answer 的完整 examples）
    save_output_parquet(
        output_problems=examples,
        save_dir_path=save_dir_path,
        logger=logger,
        save_name=args.save_name,
        meta_name=args.save_meta_name,
    )
    logger.info(f"Saved to {save_dir_path / args.save_name}")


if __name__ == "__main__":
    main()
