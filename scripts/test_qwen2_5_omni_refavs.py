#!/usr/bin/env python3
"""Evaluate an AURORA Qwen2.5-Omni checkpoint on REF-AVS test splits.

The language generation uses ms-swift's official PtEngine.  When enabled,
the generated response is then encoded through the AURORA training forward to
obtain SAM masks and frame-level IoU/Dice scores.  This keeps deployment and
evaluation on the same model/template path as SFT while avoiding vLLM-specific
assumptions about the custom SAM head.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_import_paths() -> None:
    root = _repo_root()
    swift_root = os.environ.get("MS_SWIFT_ROOT", "/mnt/tbo/lvyf/cate-pred-embedding")
    for path in (root, Path(swift_root)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _unwrap_model(model):
    current = model
    for _ in range(8):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            next_model = current.get_base_model()
            if next_model is current:
                break
            current = next_model
            continue
        break
    return current


def _frame_metrics(logits: torch.Tensor, target: torch.Tensor):
    prediction = logits.sigmoid() >= 0.5
    target = target >= 0.5
    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()
    pred_sum = prediction.sum().item()
    target_sum = target.sum().item()
    iou = 1.0 if union == 0 else intersection / union
    denominator = pred_sum + target_sum
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    return iou, dice


def _safe_sample_id(value: Any, case_index: int) -> str:
    """Convert a dataset id into a path component without allowing traversal."""
    text = str(value).strip() if value is not None else ""
    safe = "".join(
        char if (char.isascii() and (char.isalnum() or char in "._-")) else "_"
        for char in text
    )
    safe = safe.strip(".")
    return safe or f"case-{case_index:06d}"


def _save_masks(predictions: Iterable[torch.Tensor], mask_dir: Path, sample_id: Any, case_index: int) -> List[str]:
    """Save thresholded per-frame logits as portable grayscale PNG files."""
    from PIL import Image

    sample_dir = mask_dir / _safe_sample_id(sample_id, case_index)
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for frame_index, prediction in enumerate(predictions):
        mask = (prediction.detach().float().sigmoid() >= 0.5).cpu().numpy()
        # ``aurora_last_pred_masks`` is normally [frames, height, width].
        # Accept a singleton channel/batch dimension from older checkpoints.
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2-D frame mask, got shape {mask.shape}")
        path = sample_dir / f"frame_{frame_index:03d}.png"
        Image.fromarray((mask.astype("uint8") * 255), mode="L").save(path)
        paths.append(str(path))
    return paths


def _load_rows(path: Path, max_samples: int, rank: int, world_size: int) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if max_samples > 0:
        rows = rows[:max_samples]
    if world_size > 1:
        rows = rows[rank::world_size]
    return rows


def _dist_barrier(world_size: int, backend: str) -> None:
    """Synchronize ranks without coupling evaluation to model collectives."""
    if world_size <= 1:
        return
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()


def _extract_referring(row: Dict[str, Any]) -> str:
    """Return the source referring expression preserved in a dataset row."""
    for key in ("referring", "referring_expression", "exp"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    messages = row.get("messages") or []
    user_content = next(
        (str(message.get("content", "")) for message in messages
         if message.get("role") == "user"),
        "",
    )
    marker = "The reference is:"
    if marker in user_content:
        value = user_content.split(marker, 1)[1]
        value = value.split(" Please segment", 1)[0]
        return value.strip()
    return ""


def _aggregate_split(
    output_dir: Path,
    split: str,
    world_size: int,
    source_path: Path | None = None,
    max_samples: int = -1,
) -> Dict[str, Any]:
    """Merge rank summaries and case rollouts after all ranks finish a split."""
    rank_summaries = []
    case_records = []
    source_rows: List[Dict[str, Any]] = []
    if source_path is not None and source_path.is_file():
        with source_path.open(encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        if max_samples > 0:
            source_rows = source_rows[:max_samples]
    for rank in range(world_size):
        summary_path = output_dir / f"summary_{split}.rank{rank}.json"
        records_path = output_dir / f"{split}.rank{rank}.jsonl"
        if not summary_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(
                f"Missing rank output for {split}: {summary_path} or {records_path}"
            )
        rank_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        rank_records = []
        with records_path.open(encoding="utf-8") as handle:
            for local_index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                # Backfill fields for rank files produced by the previous
                # evaluator version, so existing probes can be aggregated
                # without rerunning inference.
                record.setdefault("rank", rank)
                record.setdefault("case_index", rank + local_index * world_size)
                record.setdefault("rollout", record.get("response"))
                record.setdefault("status", "error" if record.get("error") else "ok")
                source_index = record["case_index"]
                if 0 <= source_index < len(source_rows):
                    source_row = source_rows[source_index]
                    record.setdefault("referring", _extract_referring(source_row))
                    source_messages = source_row.get("messages") or []
                    record.setdefault(
                        "prompt",
                        source_messages[0].get("content", "") if source_messages else "",
                    )
                    record.setdefault(
                        "class_name",
                        source_row.get("class_name", ""),
                    )
                rank_records.append(record)
                case_records.append(record)
        # Normalize old rank files in place so both the per-rank and merged
        # JSONL outputs expose the same rollout/referring schema.
        with records_path.open("w", encoding="utf-8") as handle:
            for record in rank_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    case_records.sort(key=lambda item: (item.get("case_index", -1), item.get("rank", -1)))
    merged_path = output_dir / f"{split}.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for record in case_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    totals = {
        "samples": sum(item.get("samples", 0) for item in rank_summaries),
        "errors": sum(item.get("errors", 0) for item in rank_summaries),
        "seg_emitted": sum(item.get("seg_emitted", 0) for item in rank_summaries),
        "seg_scored": sum(item.get("seg_scored", 0) for item in rank_summaries),
        "iou_sum": sum(item.get("iou_sum", 0.0) for item in rank_summaries),
        "dice_sum": sum(item.get("dice_sum", 0.0) for item in rank_summaries),
    }
    aggregate = {
        "split": split,
        "world_size": world_size,
        "rank_summaries": [
            str(output_dir / f"summary_{split}.rank{rank}.json")
            for rank in range(world_size)
        ],
        "case_output": str(merged_path),
        **totals,
        "seg_rate": totals["seg_emitted"] / max(totals["samples"], 1),
        # IoU/Dice are averaged over samples for which the model emitted one
        # valid [SEG] token and SAM produced a mask.
        "mean_iou_scored": totals["iou_sum"] / max(totals["seg_scored"], 1),
        "mean_dice_scored": totals["dice_sum"] / max(totals["seg_scored"], 1),
        "rank_count": len(rank_summaries),
    }
    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    return aggregate


def _sam_scores(
    model,
    template,
    row: Dict[str, Any],
    response: str,
    device: torch.device,
    mask_dir: Path | None = None,
    case_index: int = 0,
):
    full_row = copy.deepcopy(row)
    full_row["messages"] = row["messages"][:1] + [{"role": "assistant", "content": response}]
    template.set_mode("train")
    try:
        encoded = template.encode(full_row)
        encoded.pop("_extra_kwargs", None)
        batch = _to_device(template.data_collator([encoded]), device)
        template.register_post_encode_hook([model])
        with torch.no_grad():
            model(**batch)
        base = _unwrap_model(model)
        predictions = getattr(base, "aurora_last_pred_masks", None)
        targets = batch.get("gt_masks")
        if not predictions or targets is None:
            raise RuntimeError("The generated [SEG] response did not produce AURORA masks")
        ious: List[float] = []
        dices: List[float] = []
        for prediction, target in zip(predictions, targets):
            target = target.to(device=prediction.device, dtype=prediction.dtype)
            for frame_prediction, frame_target in zip(prediction, target):
                iou, dice = _frame_metrics(frame_prediction, frame_target)
                ious.append(iou)
                dices.append(dice)
        scores = {
            "iou": sum(ious) / max(len(ious), 1),
            "dice": sum(dices) / max(len(dices), 1),
            "frames_scored": len(ious),
        }
        if mask_dir is not None:
            scores["mask_dir"] = str(mask_dir / _safe_sample_id(row.get("sample_id"), case_index))
            scores["mask_paths"] = _save_masks(
                predictions[0], mask_dir, row.get("sample_id"), case_index
            )
        return scores
    finally:
        template.set_mode("pt")


def _build_engine(args):
    # Importing the registration file is required before PtEngine resolves the
    # custom model/template names.
    import swift_plugin.aurora_qwen2_5_omni  # noqa: F401
    from safetensors.torch import load_file, save_file
    from swift.llm import PtEngine, get_template
    from swift.tuners import Swift

    device = torch.device(args.device)
    engine = PtEngine(
        args.model,
        torch_dtype=torch.float16,
        model_type="aurora_qwen2_5_omni",
        new_special_tokens=["[SEG]"],
        attn_impl=args.attn_impl,
        device_map={"": str(device)},
        download_model=False,
        max_batch_size=1,
    )
    # PEFT treats a broad ``modules_to_save=['embed_tokens']`` entry as
    # matching both Omni thinker and talker embeddings.  SFT only updates the
    # thinker embedding, so older ms-swift/PEFT versions fail if the untouched
    # talker key is absent.  Add that base tensor to a temporary adapter copy;
    # the user's checkpoint remains unchanged and all other loading is native.
    adapter_dir = Path(args.checkpoint)
    state_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not state_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint must contain adapter_model.safetensors and adapter_config.json: {adapter_dir}")
    state = load_file(str(state_path), device="cpu")
    talker_key = "base_model.model.talker.model.embed_tokens.weight"
    if talker_key not in state:
        talker = engine.model.talker.model.embed_tokens.weight.detach().cpu().contiguous()
        state[talker_key] = talker
        temp_dir = tempfile.TemporaryDirectory(prefix="aurora_adapter_")
        compat_dir = Path(temp_dir.name)
        shutil.copy2(config_path, compat_dir / "adapter_config.json")
        save_file(state, compat_dir / "adapter_model.safetensors", metadata={"format": "pt"})
        engine.model = Swift.from_pretrained(engine.model, str(compat_dir))
        # Keep the temporary directory alive for the lifetime of the engine.
        engine._aurora_adapter_tmp = temp_dir
    else:
        engine.model = Swift.from_pretrained(engine.model, str(adapter_dir))
    # PEFT may materialize ``modules_to_save`` on CPU even when the base Omni
    # model was loaded with a single-device map.  Keep all custom AURORA
    # modules colocated for the post-generation SAM forward.
    engine.model.to(device)
    template = get_template("aurora_qwen2_5_omni", engine.processor)
    # The standalone script does not run through Trainer, so bind the loaded
    # model explicitly.  The multimodal collator needs it to compute Omni
    # position ids instead of constructing a meta-device dummy model.
    template.model = engine.model
    template.set_mode("pt")
    return engine, template, device


def _evaluate_split(args, engine, template, device, split: str, output_dir: Path, rank: int, world_size: int):
    from swift.llm import InferRequest, RequestConfig

    path = Path(args.data_dir) / f"refavs_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test split: {path}")
    rows = _load_rows(path, args.max_samples, rank, world_size)
    split_mask_dir = None
    if args.run_sam and args.save_masks:
        split_mask_dir = Path(args.mask_dir) / split
    output_name = f"{split}.rank{rank}.jsonl" if world_size > 1 else f"{split}.jsonl"
    output_path = output_dir / output_name
    seg_id = template.tokenizer.convert_tokens_to_ids("[SEG]")
    totals = {
        "samples": 0,
        "errors": 0,
        "seg_emitted": 0,
        "seg_scored": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
    }
    started = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            # The source index is stable across ranks and lets rank 0 restore
            # the original dataset order in the merged rollout file.
            case_index = rank + (index - 1) * world_size
            request = InferRequest(
                messages=copy.deepcopy(row["messages"][:1]),
                videos=row.get("videos"),
                audios=row.get("audios"),
            )
            try:
                result = engine.infer(
                    [request],
                    RequestConfig(
                        max_tokens=args.max_new_tokens, temperature=0.0
                    ),
                    template=template,
                    use_tqdm=False,
                )[0]
                response = result.choices[0].message.content or ""
                generated_ids = template.tokenizer(response, add_special_tokens=False)["input_ids"]
                seg_count = sum(token == seg_id for token in generated_ids)
                record: Dict[str, Any] = {
                    "split": split,
                    "case_index": case_index,
                    "rank": rank,
                    "sample_id": row.get("sample_id"),
                    "referring": _extract_referring(row),
                    "prompt": (row.get("messages") or [{}])[0].get("content", ""),
                    "class_name": row.get("class_name", ""),
                    "rollout": response,
                    "response": response,
                    "labels": row.get("messages", [{}, {}])[-1].get("content", ""),
                    "seg_count": seg_count,
                    "status": "ok",
                }
                totals["samples"] += 1
                if seg_count == 1:
                    totals["seg_emitted"] += 1
                    if args.run_sam:
                        scores = _sam_scores(
                            engine.model,
                            template,
                            row,
                            response,
                            device,
                            mask_dir=split_mask_dir,
                            case_index=case_index,
                        )
                        record.update(scores)
                        totals["seg_scored"] += 1
                        totals["iou_sum"] += scores["iou"]
                        totals["dice_sum"] += scores["dice"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                totals["samples"] += 1
                totals["errors"] += 1
                handle.write(json.dumps({
                    "split": split,
                    "case_index": case_index,
                    "rank": rank,
                    "sample_id": row.get("sample_id"),
                    "referring": _extract_referring(row),
                    "prompt": (row.get("messages") or [{}])[0].get("content", ""),
                    "class_name": row.get("class_name", ""),
                    "rollout": None,
                    "response": None,
                    "status": "error",
                    "error": repr(exc),
                }, ensure_ascii=False) + "\n")
                if args.fail_fast:
                    raise
            if index % args.log_every == 0 or index == len(rows):
                print(f"[{split} rank={rank}] {index}/{len(rows)}", flush=True)
    summary = {
        "split": split,
        "rank": rank,
        "world_size": world_size,
        **totals,
        "seg_rate": totals["seg_emitted"] / max(totals["samples"], 1),
        "mean_iou_scored": totals["iou_sum"] / max(totals["seg_scored"], 1),
        "mean_dice_scored": totals["dice_sum"] / max(totals["seg_scored"], 1),
        "runtime_sec": time.time() - started,
        "output": str(output_path),
    }
    summary_path = output_dir / (f"summary_{split}.rank{rank}.json" if world_size > 1 else f"summary_{split}.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    _dist_barrier(world_size, args.dist_backend)
    if rank == 0:
        _aggregate_split(
            output_dir,
            split,
            world_size,
            source_path=path,
            max_samples=args.max_samples,
        )
    _dist_barrier(world_size, args.dist_backend)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-3B"))
    parser.add_argument("--checkpoint", required=True, help="ms-swift checkpoint directory containing adapter_model.safetensors")
    parser.add_argument("--data-dir", default=str(_repo_root() / "outputs/data/refavs"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation directory; defaults to outputs/eval/qwen2_5_omni/<checkpoint-name>",
    )
    parser.add_argument("--splits", nargs="+", default=["test_s", "test_u", "test_n"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--dist-backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--run-sam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directory for predicted mask PNGs; defaults to <output-dir>/masks",
    )
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save thresholded SAM masks as PNG files when --run-sam is enabled",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    default_data_dir = _repo_root() / "outputs/data/refavs"
    legacy_data_dir = _repo_root() / "outputs"
    if args.data_dir == str(default_data_dir) and not (default_data_dir / "refavs_test_s.jsonl").is_file():
        if (legacy_data_dir / "refavs_test_s.jsonl").is_file():
            print(f"Warning: new data directory is missing; using legacy directory {legacy_data_dir}", flush=True)
            args.data_dir = str(legacy_data_dir)
    if args.output_dir is None:
        checkpoint_name = _safe_sample_id(Path(args.checkpoint).name, 0)
        args.output_dir = str(_repo_root() / "outputs/eval/qwen2_5_omni" / checkpoint_name)
    _add_import_paths()
    os.environ.setdefault("AURORA_SAM_CHECKPOINT", "/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_dir is None:
        args.mask_dir = str(output_dir / "masks")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    engine, template, device = _build_engine(args)
    engine.model.eval()
    summaries = [_evaluate_split(args, engine, template, device, split, output_dir, rank, world_size) for split in args.splits]
    if rank == 0:
        aggregate_summaries = [
            json.loads((output_dir / f"summary_{split}.json").read_text(encoding="utf-8"))
            for split in args.splits
        ]
        (output_dir / "summary.json").write_text(
            json.dumps(aggregate_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    _dist_barrier(world_size, args.dist_backend)
    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
