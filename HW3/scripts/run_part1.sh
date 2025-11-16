#!/usr/bin/env bash
set -Eeuo pipefail

: "${CUDA_VISIBLE_DEVICES:=3}"
: "${ROUNDS:=300}"
: "${BATCH_SIZE:=32}"
: "${CF_LIST:=0.10}"
: "${E_LIST:=7}"
: "${LR_LIST:=0.005}"
: "${PAR:=1}"

TRAIN=${TRAIN:-"$HOME/train_data.npy"}
TEST=${TEST:-"$HOME/test_data.npy"}
OUTROOT=${OUTROOT:-"./outputs_torch"}

mkdir -p "$OUTROOT"

run_one () {
  local cf="$1" e="$2" lr="$3"
  local tag="part1_cf${cf}_e${e}_lr${lr}"
  local out="${OUTROOT}/${tag}"
  [[ -f "${out}/history.json" ]] && { echo ">>> SKIP ${out}"; return; }
  echo "[$(date +%H:%M:%S)] >>> RUN cf=${cf} e=${e} lr=${lr} -> ${out}"
  mkdir -p "${out}"
  # Unbuffered stdout + line-buffered stdbuf for safety
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTHONUNBUFFERED=1 stdbuf -oL -eL \
  python -u train_serial_torch.py \
    --train_data "${TRAIN}" --test_data "${TEST}" \
    --rounds "${ROUNDS}" --client_frac "${cf}" --local_epochs "${e}" \
    --batch_size "${BATCH_SIZE}" --lr "${lr}" \
    --out_dir "${out}" | tee "${out}/run.log"
}

jobs=0
for cf in ${CF_LIST}; do
  for e in ${E_LIST}; do
    for lr in ${LR_LIST}; do
      if (( PAR > 1 )); then
        run_one "$cf" "$e" "$lr" &
        (( ++jobs ))
        (( jobs % PAR == 0 )) && wait
      else
        run_one "$cf" "$e" "$lr"
      fi
    done
  done
done
wait || true
echo "DONE."