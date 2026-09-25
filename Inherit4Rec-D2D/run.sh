#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${GPU_ID:-}" ]]; then
    [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be non-negative" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpy, pandas, torch' >/dev/null || {
    echo "Missing dependencies; run ${PROJECT_ROOT}/install_dependencies.sh" >&2; exit 1;
}

UNIFORMER_DATA_DIR="${UNIFORMER_DATA_DIR:-${PROJECT_ROOT}/data/kuairand_1k}"
UNIFORMER_OUTPUT_DIR="${UNIFORMER_OUTPUT_DIR:-${SCRIPT_DIR}}"
required_files=(
    user_features_mapped.csv user_statistic_info.json
    item_features_mapped.csv item_statistic_info.json
    standard_interactions.csv user_positive_sequences.pkl
)
for filename in "${required_files[@]}"; do
    if [[ ! -f "${UNIFORMER_DATA_DIR}/${filename}" ]]; then
        echo "Missing data file: ${UNIFORMER_DATA_DIR}/${filename}" >&2
        echo "See ${PROJECT_ROOT}/DATA.md for the expected preprocessed-data format." >&2
        exit 1
    fi
done

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/train.py" \
    --data_dir "${UNIFORMER_DATA_DIR}" \
    --train_start 0 --valid_data 2 --test_data 2 \
    --seq_config "seq_a:256,seq_b:256,seq_c:256" \
    --batch_size 2048 --grad_accum_steps 1 --eval_batch_size 256 \
    --num_workers 8 --prefetch_factor 2 --static_chunksize 500000 \
    --seed 2026 --device auto \
    --d_model 400 --emb_dim 128 --user_tokens 25 --item_tokens 8 \
    --num_fim_layers 2 --num_tim_layers 1 --num_heads 5 \
    --s_hidden_multi 2 --source_hidden_dim 600 --target_hidden_dim 1800 \
    --t_hidden_multi 1 --dropout 0.01 --ns_shuffle_seed 42 \
    --growth_seed 20260716 \
    --lr 1e-4 --growth_lr 1.5e-4 --growth_final_lr 1e-4 \
    --growth_warmup_steps 200 --growth_cosine_steps 1000 \
    --sparse_lr 1e-3 --weight_decay 1e-3 --grad_clip 1.0 \
    --bf16 --reinit_sparse_every_epoch \
    --ckpt_dir "${UNIFORMER_OUTPUT_DIR}/checkpoints/growth_01b_to_03b" \
    --log_file "${UNIFORMER_OUTPUT_DIR}/logs/growth_01b_to_03b.log" \
    "$@"
