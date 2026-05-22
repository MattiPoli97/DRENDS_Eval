#!/usr/bin/env bash
set -euo pipefail

DATA_ROOTS=(
  #"/Volumes/One Touch/DRENDS/DRENDS_ExVivo_Sequences"
  "/Volumes/One Touch/DRENDS/DRENDS_Phantom_Sequences"
)

OUT_ROOTS=(
  #"/Volumes/One Touch/DRENDS/17_03_26/EXvivo_results"
  "/Volumes/One Touch/DRENDS/17_03_26/Phantom_results"
)

SCENES=(
  #"Seq00_Colon_Ext"
  #"Seq04_Intestine_Med"
  #"Seq08_Stomach_High"
  #"Seq09_Liver_Ext"
  "Seq13_Pancreas_Med"
)

PY="python3"
SCRIPT="__main__.py"
FOLDER_GLOB="*"
FORCE=0

MODELS=(foundationstereo)

for i in "${!DATA_ROOTS[@]}"; do
  DATA_ROOT="${DATA_ROOTS[$i]}"
  OUT_ROOT="${OUT_ROOTS[$i]}"

  mkdir -p "$OUT_ROOT"

  while IFS= read -r -d '' SCENE_DIR; do
    [ -d "$SCENE_DIR" ] || continue
    SCENE_NAME="$(basename "$SCENE_DIR")"
    # check if scene is in the provied list of scenes to run (if SCENES variable is set)
    if [ -n "${SCENES:-}" ] && [[ ! " ${SCENES[*]} " =~ " ${SCENE_NAME} " ]]; then
      echo "[SKIP] ${SCENE_NAME} (not in SCENES list)"
      continue
    fi

    for model in "${MODELS[@]}"; do

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

      cmd=(
        "$PY" "$SCRIPT"
        --data_path "$SCENE_DIR"
        --model "$model"
        --output_path "$OUT_DIR"
        --batch_size 1
        --headless
        --debug
      )

      case "$model" in
        depthpro)
          cmd+=(--num_workers 8
                --store_pngs
              )
          ;;
        damv2)
          cmd+=(
            --num_workers 0
            --store_pngs

          )
          ;;
        raft)
          cmd+=(
            --num_workers 0
            --store_pngs
      
            --raft_ckpt "models/raftstereo-middlebury.pth"
          )
          ;;
        foundationstereo)
          cmd+=(
            --num_workers 0
            --store_pngs

            --fs_ckpt "pretrained_models/23-51-11/model_best_bp2.pth"
            --fs_root "./FoundationStereo"
            --fs_iters 32
            --fs_scale 1.0
            --debug
          )
          ;;
        *)
          cmd+=(
            --num_workers 0
            --store_pngs
          )
          ;;
      esac

      (
        set -x
        "${cmd[@]}"
      ) 2>&1 | tee "$LOG"

      echo "=== Done ${model} on ${SCENE_NAME} ==="
    done
  done < <(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$FOLDER_GLOB" -print0 | sort -z)

  echo "All scenes & models finished. Results under: $OUT_ROOT"
done