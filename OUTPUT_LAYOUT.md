# Output Layout

New runs use the following layout under `outputs/`:

```text
outputs/
  data/
    refavs/                         # Generated JSONL datasets for training/eval
    refavs_hwnas/                   # Datasets generated on the HWNAS mount
  train/
    <model>_<gpu>gpu_<frames>f/     # One ms-swift run root, including logs
      vN-YYYYMMDD-HHMMSS/
        checkpoint-<step>/
  eval/
    <model>/<checkpoint-or-run>/    # Standalone evaluation outputs
      *.jsonl
      summary*.json
      masks/                         # Optional saved mask visualizations
        <split>/<sample-id>/frame_000.png
```

Naming rules:

- `data/` contains only generated dataset files (`refavs_*.jsonl`) and no model checkpoints.
- `train/` contains training logs, TensorBoard files, arguments, optimizer state, and checkpoints.
- `eval/` contains independent test output, summaries, and optional mask images; it must not be used as a training `output_dir`.
- Qwen REF-AVS evaluation saves thresholded predicted masks by default under
  `masks/<split>/<sample-id>/frame_XXX.png`. Each successful SAM-scored JSONL
  record includes `mask_dir` and `mask_paths`; pass `--no-save-masks` to keep
  IoU/Dice scoring while disabling image files.
- Use lowercase model identifiers: `videollama2`, `qwen2_5_omni_3b`, or `qwen2_5_omni_7b`.
- Include the GPU count and sampled frame count in wrapper-created run roots, for example `videollama2_2gpu_10f`.
- Keep the checkpoint name (`checkpoint-500`) supplied by ms-swift unchanged below the run root.
- Use `AURORA_DATA_JSONL`, `AURORA_VAL_JSONL`, `OUTPUT_DIR`, `LOG_FILE`, `--data-dir`, and `--output-dir` to override these defaults for legacy or special runs.

Existing flat directories under `outputs/` are intentionally left in place. They are historical artifacts or may belong to active runs and should be migrated manually only after the associated process has stopped.
