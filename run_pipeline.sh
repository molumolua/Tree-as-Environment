#!/usr/bin/env bash
# =============================================================================
# 全流程 Pipeline：按序执行
#   api_get_answer → verifier_with_vllm → api_split_question_and_condition
#   → api_summary_reply → api_get_sub_question → filter_train_data
# 所有参数集中在脚本顶部，前序 output 作为后续 file_glob/input。
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 一、路径与数据（可修改）
# -----------------------------------------------------------------------------
# 初始数据目录（api_get_answer 的输入）
LOAD_DIR="${LOAD_DIR:-./dataset}"
# 初始数据匹配（api_get_answer 的 file_glob，如 train-*.parquet 或 *.parquet）
FILE_GLOB_INITIAL="${FILE_GLOB_INITIAL:-webinstruct_1k.parquet}"
# 所有中间结果与最终结果保存目录
SAVE_DIR="${SAVE_DIR:-./save2}"
# split 名称，与文件名前缀一致
SPLIT="${SPLIT:-train}"

# 各步骤输出文件名（前一步的 output 即后一步的 input）
OUTPUT_ANSWER="${OUTPUT_ANSWER:-output_with_answer.parquet}"
OUTPUT_SCORE="${OUTPUT_SCORE:-output_with_score.parquet}"
OUTPUT_SPLIT="${OUTPUT_SPLIT:-output_with_split.parquet}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-output_with_summary.parquet}"
OUTPUT_SUB_QUESTION="${OUTPUT_SUB_QUESTION:-output_with_sub_question.parquet}"
OUTPUT_FILTERED="${OUTPUT_FILTERED:-filtered_train.parquet}"

# -----------------------------------------------------------------------------
# 二、通用 API / 推理参数（各 Python 脚本共用或部分共用）
# -----------------------------------------------------------------------------
LOAD_TYPE="${LOAD_TYPE:-parquet}"
START_PROBLEM_IDX="${START_PROBLEM_IDX:-0}"
MAX_ROWS="${MAX_ROWS:--1}"
MODEL="${MODEL:-glm-4.7}"
N_PROCESSES="${N_PROCESSES:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TIMEOUT="${TIMEOUT:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INNER_MAX_TRY="${INNER_MAX_TRY:-1}"

# verifier_with_vllm 专用
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/Users/molu/Tree-as-Environment/model}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

# Python 解释器
# 默认：molu 环境；Step2：vllm 环境
PYTHON_DEFAULT="${PYTHON_DEFAULT:-/Users/molu/miniconda3/envs/molu/bin/python}"
PYTHON_VLLM="${PYTHON_VLLM:-/Users/molu/.venv-vllm-metal/bin/python}"

# -----------------------------------------------------------------------------
# 三、由上面变量推导出的路径（一般无需改）
# -----------------------------------------------------------------------------
PATH_ANSWER="${SAVE_DIR}/${OUTPUT_ANSWER}"
PATH_SCORE="${SAVE_DIR}/${OUTPUT_SCORE}"
PATH_SPLIT="${SAVE_DIR}/${OUTPUT_SPLIT}"
PATH_SUMMARY="${SAVE_DIR}/${OUTPUT_SUMMARY}"
PATH_SUB_QUESTION="${SAVE_DIR}/${OUTPUT_SUB_QUESTION}"
PATH_FILTERED="${SAVE_DIR}/${OUTPUT_FILTERED}"

# -----------------------------------------------------------------------------
# 执行
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Pipeline 配置"
echo "=============================================="
echo "LOAD_DIR=${LOAD_DIR}"
echo "FILE_GLOB_INITIAL=${FILE_GLOB_INITIAL}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_ANSWER=${OUTPUT_ANSWER}"
echo "OUTPUT_SCORE=${OUTPUT_SCORE}"
echo "OUTPUT_SPLIT=${OUTPUT_SPLIT}"
echo "OUTPUT_SUMMARY=${OUTPUT_SUMMARY}"
echo "OUTPUT_SUB_QUESTION=${OUTPUT_SUB_QUESTION}"
echo "OUTPUT_FILTERED=${OUTPUT_FILTERED}"
echo "MODEL=${MODEL} N_PROCESSES=${N_PROCESSES} BATCH_SIZE=${BATCH_SIZE}"
echo "PYTHON_DEFAULT=${PYTHON_DEFAULT}"
echo "PYTHON_VLLM=${PYTHON_VLLM}"
echo "=============================================="

mkdir -p "${SAVE_DIR}"

# ----- 1. api_get_answer -----
echo ""
echo "[1/6] api_get_answer: 从 ${LOAD_DIR}/${FILE_GLOB_INITIAL} 生成 answer，写入 ${PATH_ANSWER}"
"${PYTHON_DEFAULT}" -m api_get_answer \
  --load_type "${LOAD_TYPE}" \
  --load_dir "${LOAD_DIR}" \
  --split "${SPLIT}" \
  --file_glob "${FILE_GLOB_INITIAL}" \
  --start_problem_idx "${START_PROBLEM_IDX}" \
  --save_dir "${SAVE_DIR}" \
  --save_name "${OUTPUT_ANSWER}" \
  --model "${MODEL}" \
  --n_processes "${N_PROCESSES}" \
  --temperature "${TEMPERATURE}" \
  --timeout "${TIMEOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --inner_max_try "${INNER_MAX_TRY}" \
  $([ -n "${MAX_ROWS}" ] && echo "--max_rows ${MAX_ROWS}")

# ----- 2. verifier_with_vllm -----
echo ""
echo "[2/6] verifier_with_vllm: 对 ${PATH_ANSWER} 判分，写入 ${PATH_SCORE}"
"${PYTHON_VLLM}" -m verifier_with_vllm \
  --model_path "${VLLM_MODEL_PATH}" \
  --data_path "${PATH_ANSWER}" \
  --load_type "${LOAD_TYPE}" \
  --save_path "${PATH_SCORE}" \
  --batch_size "${VLLM_BATCH_SIZE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"

# ----- 3. api_split_question_and_condition -----
echo ""
echo "[3/6] api_split_question_and_condition: 从 ${OUTPUT_SCORE} 拆分为 question+conditions，写入 ${PATH_SPLIT}"
"${PYTHON_DEFAULT}" -m api_split_question_and_condition \
  --load_type "${LOAD_TYPE}" \
  --load_dir "${SAVE_DIR}" \
  --file_glob "${OUTPUT_SCORE}" \
  --split "${SPLIT}" \
  --start_problem_idx "${START_PROBLEM_IDX}" \
  --save_dir "${SAVE_DIR}" \
  --save_name "${OUTPUT_SPLIT}" \
  --model "${MODEL}" \
  --n_processes "${N_PROCESSES}" \
  --temperature "${TEMPERATURE}" \
  --timeout "${TIMEOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --inner_max_try "${INNER_MAX_TRY}" \
  $([ -n "${MAX_ROWS}" ] && echo "--max_rows ${MAX_ROWS}")

# ----- 4. api_summary_reply -----
echo ""
echo "[4/6] api_summary_reply: 从 ${OUTPUT_SPLIT} 总结步骤，写入 ${PATH_SUMMARY}"
"${PYTHON_DEFAULT}" -m api_summary_reply \
  --load_type "${LOAD_TYPE}" \
  --load_dir "${SAVE_DIR}" \
  --file_glob "${OUTPUT_SPLIT}" \
  --split "${SPLIT}" \
  --start_problem_idx "${START_PROBLEM_IDX}" \
  --save_dir "${SAVE_DIR}" \
  --save_name "${OUTPUT_SUMMARY}" \
  --model "${MODEL}" \
  --n_processes "${N_PROCESSES}" \
  --temperature "${TEMPERATURE}" \
  --timeout "${TIMEOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --inner_max_try "${INNER_MAX_TRY}" \
  $([ -n "${MAX_ROWS}" ] && echo "--max_rows ${MAX_ROWS}")

# ----- 5. api_get_sub_question -----
echo ""
echo "[5/6] api_get_sub_question: 从 ${OUTPUT_SUMMARY} 生成子问题，写入 ${PATH_SUB_QUESTION}"
"${PYTHON_DEFAULT}" -m api_get_sub_question \
  --load_type "${LOAD_TYPE}" \
  --load_dir "${SAVE_DIR}" \
  --file_glob "${OUTPUT_SUMMARY}" \
  --split "${SPLIT}" \
  --start_problem_idx "${START_PROBLEM_IDX}" \
  --save_dir "${SAVE_DIR}" \
  --save_name "${OUTPUT_SUB_QUESTION}" \
  --model "${MODEL}" \
  --n_processes "${N_PROCESSES}" \
  --temperature "${TEMPERATURE}" \
  --timeout "${TIMEOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --inner_max_try "${INNER_MAX_TRY}" \
  $([ -n "${MAX_ROWS}" ] && echo "--max_rows ${MAX_ROWS}")

# ----- 6. filter_train_data -----
echo ""
echo "[6/6] filter_train_data: 从 ${PATH_SUB_QUESTION} 筛选训练集，写入 ${PATH_FILTERED}"
"${PYTHON_DEFAULT}" -m filter_train_data \
  --input "${PATH_SUB_QUESTION}" \
  --output "${PATH_FILTERED}"

echo ""
echo "=============================================="
echo "Pipeline 全部完成。最终结果: ${PATH_FILTERED}"
echo "=============================================="
