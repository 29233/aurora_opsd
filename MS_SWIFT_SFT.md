# AURORA SFT on ms-swift

This first migration milestone supports Qwen2.5-Omni-3B with the native
ms-swift multimodal processor and an in-process AURORA SAM head. It is limited
to SFT; VideoLLaMA2 registration is intentionally a later milestone.

The same registration now accepts the local Qwen2.5-Omni-7B checkpoint. The
current 7B memory boundary is documented below; the 3B smoke command remains
the default because it is faster to iterate.

## What is trained

`[SEG]` is added through ms-swift's `--new_special_tokens` API. The SFT
objective is:

```text
language CE + 2.0 * mask BCE + 0.5 * mask Dice
```

The language model uses LoRA on `thinker.model` Q/V projections. The complete
`embed_tokens`, `lm_head`, `text_hidden_fcs`, and SAM `mask_decoder` modules are
saved. The SAM image encoder, Qwen vision tower, Qwen audio tower, and
multimodal projector remain frozen.

## REF-AVS prompt contract

The Qwen2.5-Omni Stage-1 data path intentionally has no `system` turn. Each
user message is rendered as:

```text
<video><audio>The reference is: {exp} Please segment the corresponding object in the video.
```

`{exp}` is copied verbatim from `REFAVS/metadata.csv`. It can describe visual,
temporal, spatial, or audio-related properties; the instruction therefore says
`corresponding object`, not `corresponding sounding object`. The older
VideoLLaMA2/LLaVA dataset helpers use a different safety `system_message` and
the wording `...in the images.`; those templates are not used by this Omni
migration.

## Precision and loss scaling

The smoke command loads the model with `--torch_dtype float16` and enables
ms-swift/Transformers FP16 mixed precision (`fp16=true`, `bf16=false`). The
test machine uses Tesla V100 GPUs, which do not provide native BF16 support.
Attention uses `sdpa`; Flash Attention is not enabled. Trainable parameters are
converted to FP32 by ms-swift/PEFT where applicable, while the frozen Qwen
media towers and SAM image encoder run in their loaded FP16 dtype. SAM mask
losses are accumulated in FP32 for numerical stability.

DeepSpeed ZeRO-2 uses the repository configuration
`configs/deepspeed_zero2_fp16_v100.json`, with dynamic FP16 loss scaling and an
initial scale of `2^8` (256). The default DeepSpeed initial scale (`2^16`) was
observed to overflow on this joint loss: five steps were skipped, with
`overflow=true` and no parameter changes. The reduced initial scale produces
non-zero gradient norms and optimizer updates on V100.

## Smoke test

The supplied test uses one REF-AVS example and two frames:

```bash
AURORA_SAM_CHECKPOINT=/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth \
AURORA_SOURCE_DIR=/mnt/tbo/lvyf/AURORA/AURORA-main \
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
bash scripts/smoke_qwen2_5_omni_3b_swift_sft.sh
```

For normal training set `AURORA_NUM_SAM_FRAMES=10`, remove the smoke-test
limits from the script, and use the available number of GPUs. The data
converter writes standard `messages/videos/audios` fields plus the custom
`sam_frame_paths/mask_paths` supervision fields. `images` is deliberately not
used for ground-truth masks.

The converter accepts `--split train`, `--split val`, and the REF-AVS test
splits (`test_s`, `test_u`, `test_n`). Training reasoning is used only for the
train split. Validation and test records use the deterministic teacher-forced
answer `It is [SEG].`, while retaining split-specific reference text and SAM
mask paths.

To observe a short optimization trend, run for five steps in a separate output
directory:

```bash
MAX_STEPS=5 SAVE_STEPS=5 \
OUTPUT_DIR=/mnt/tbo/lvyf/AURORA_from_scratch/outputs/train/qwen2_5_omni_3b_sam_5step_fp16stable \
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
bash scripts/smoke_qwen2_5_omni_3b_swift_sft.sh
```

On the one-sample smoke data, the stable run reduced total loss from `2.92858`
to `2.57279` over five updates. CE reduced from `2.25399` to `1.90878`;
mask BCE varied between `0.08215` and `0.10099`, and Dice loss remained near
`1.0`, so this is only an optimization-path check, not a quality benchmark.
The checkpoint contains updated SAM decoder, projector, embedding/lm-head, and
LoRA weights. A one-time compatibility shim for the installed older Accelerate
is loaded by the plugin because Transformers 4.57 passes
`keep_torch_compile` to `unwrap_model`.

## 7B resource boundary

With the current modules-to-save policy, the 7B Omni model has about `10.69B`
parameters in total and `1.11B` trainable parameters (the full embedding and
language-head copies dominate the trainable memory). A single 32GB V100 loads
the model but runs out of memory when ZeRO-2 initializes the optimizer: it uses
about `27.72 GiB` and needs another `4.14 GiB`. An 8x32GB V100 run completed a
real five-step joint SFT probe with eight samples and `AURORA_NUM_SAM_FRAMES=2`,
at about `27.9 GiB` peak per rank. Rank-0 loss changed from `4.52839` to
`2.78311`, all five gradient norms were non-zero, and all eight optimizer
shards reported `overflow=false`. The 3B verification that preceded this used
one V100; the 7B run should use the multi-GPU configuration.

The same 7B/8-GPU setup does not currently fit with 10 SAM frames: model and
optimizer initialization succeeds, but backward reaches about `29.7 GiB` and
fails on an additional `2.26-2.27 GiB` allocation. The failed probe is recorded
under `outputs/qwen2_5_omni_7b_8gpu_10frame_probe`; reducing activation or
trainable-parameter memory is required before production-length 10-frame runs.

For an 8-GPU probe, provide at least one training sample per rank, for example
by setting `MAX_SAMPLES=8`, and use the local 7B path through `MODEL_PATH`.

## Full training entry point and checkpoints

Use `scripts/train_qwen2_5_omni_swift_sft.sh` for a full-data run. It defaults
to `MAX_SAMPLES=-1`, `MAX_STEPS=-1` (the configured epoch count), 10 SAM frames,
and saves/evaluates every 500 optimizer steps:

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_qwen2_5_omni_swift_sft.sh
```

Override `SAVE_STEPS` to change the interval. The underlying smoke script uses
`SAVE_STEPS` directly; when unset there it falls back to `MAX_STEPS`, which is
why bounded smoke runs saved at their final step.

The full entry point generates `outputs/data/refavs/refavs_val.jsonl` from the REF-AVS
`val` split and passes it as `--val_dataset`. `eval_strategy=steps` and
`eval_steps` follow `SAVE_STEPS` by default, so ms-swift evaluates at each
interval before saving the matching checkpoint. With
`predict_with_generate=True`, the validation pass is generation-based and the
generated collection is written to `predict.jsonl`; set `EVAL_STEPS` to use a
different evaluation cadence. The smoke entry point uses one validation sample
by default (`VAL_MAX_SAMPLES=1`).

TensorBoard logging is enabled by default with `--report_to tensorboard`.
Logs are written below the run's `runs/` directory. Use `REPORT_TO=none` to
disable it, or run `tensorboard --logdir /path/to/output/runs`.

Evaluation uses ms-swift's native `predict_with_generate=True` path. The
`Seq2SeqTrainer` uses the official `PtEngine` generation implementation,
partitions evaluation across all ranks, gathers the generated predictions, and
appends them to `predict.jsonl` in the run directory. Set
`AURORA_FREE_EVAL_MAX_TOKENS` to change the generation cap (the default is
128). The validation converter's `VAL_MAX_SAMPLES` can bound the collection
for a short probe. This evaluation is LLM-only: it records the generated CoT,
whether `[SEG]` was emitted, and the reference labels, but does not rerun SAM
or compute mask IoU/Dice. SAM remains in the training forward path and in the
saved checkpoint, so full segmentation testing can be run independently on
another machine after the checkpoint is copied.

This uses the documented ms-swift evaluation implementation directly rather
than a rank-0 custom callback. Consequently generation is distributed and the
trainer's normal evaluation/saving lifecycle remains in control.

Every ms-swift checkpoint's `adapter_model.safetensors` already includes the
trainable SAM mask decoder because `mask_decoder` is listed in
`--modules_to_save`. The external callback additionally writes a standalone
`sam_mask_decoder.safetensors` (120 tensors) into the same checkpoint directory.
The SAM image encoder and prompt encoder are frozen and are intentionally not
duplicated in this sidecar; the original full SAM checkpoint remains the source
for those frozen components.
