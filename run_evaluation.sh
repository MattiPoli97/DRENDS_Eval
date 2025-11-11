#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
DATA_ROOT="./ExVivo_data"         
OUT_ROOT="./ExVivo_results"    
PY="python3"                        
SCRIPT="__main__.py"                
FOLDER_GLOB="*"                     
FORCE=0                             
# -------------------

MODELS=(midas damv2 damv2_metric zoedepth monodepth2 depthpro)

EXTRA_midas=(--midas_type DPT_Hybrid)
EXTRA_damv2=(--damv2_size small)
EXTRA_damv2_metric=(--damv2m_size Base --damv2m_domain Indoor --damv2m_max_depth 20.0)
EXTRA_zoedepth=(--zoe_max_depth 0.6)
EXTRA_depthpro=()
EXTRA_monodepth2=(--monodepth2_ckpt mono_640x192)

mkdir -p "$OUT_ROOT"
while IFS= read -r -d '' SCENE_DIR; do
  [ -d "$SCENE_DIR" ] || continue
  SCENE_NAME="$(basename "$SCENE_DIR")"

  for model in "${MODELS[@]}"; do
    varname="EXTRA_${model}"

    if [ "$model" = "depthpro" ]; then
      extra_args=()
    else
      eval "extra_args=(\"\${${varname}[@]}\")"
    fi

    OUT_DIR="${OUT_ROOT}/${SCENE_NAME}/${model}"
    mkdir -p "$OUT_DIR"

    if [ "$FORCE" -eq 0 ] && { [ -s "${OUT_DIR}/metrics_summary_raw.json" ] || [ -s "${OUT_DIR}/metrics_summary_metricized.json" ]; }; then
      echo "[SKIP] ${SCENE_NAME} / ${model} (summary exists; set FORCE=1 to rerun)"
      continue
    fi

    echo "=== Running ${model} on ${SCENE_NAME} ==="
    echo "Data -> $SCENE_DIR"
    echo "Out  -> $OUT_DIR"

    LOG="${OUT_DIR}/run.log"
    (
      set -x
      if [ "$model" = "depthpro" ]; then
        "$PY" "$SCRIPT" \
          --data_path "$SCENE_DIR" \
          --model "$model" \
          --output_path "$OUT_DIR" \
          --batch_size 1 \
          --num_workers 8 \
          --headless
      else
        "$PY" "$SCRIPT" \
          --data_path "$SCENE_DIR" \
          --model "$model" \
          --output_path "$OUT_DIR" \
          --batch_size 1 \
          --num_workers 0 \
          --headless \
          "${extra_args[@]}"
      fi
    ) 2>&1 | tee "$LOG"

    echo "=== Done ${model} on ${SCENE_NAME} ==="
  done
done < <(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$FOLDER_GLOB" -print0 | sort -z)

echo "All scenes & models finished. Results under: $OUT_ROOT"
