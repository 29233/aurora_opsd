"""Legacy free-generation callback.

Evaluation is now handled by ms-swift's native ``predict_with_generate``
implementation in ``Seq2SeqTrainer``. This module is retained for backwards
compatibility with older experiments, but is deliberately not registered as a
Trainer callback so it cannot launch a rank-0 SAM evaluation during training.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, Iterable

import torch
from transformers import TrainerCallback

from swift.llm import InferRequest, PtEngine, RequestConfig, get_template
from swift.plugin import extra_callbacks


def _base_model(model):
    current = model
    for _ in range(6):
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


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


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


class AuroraFreeGenerationEvalCallback(TrainerCallback):
    """Run a bounded free-generation segmentation evaluation after eval steps."""

    def __init__(self):
        self._engine = None
        self._template = None
        self._writer = None
        self._busy = False

    @staticmethod
    def _enabled() -> bool:
        return os.environ.get("AURORA_FREE_EVAL", "1").lower() not in {"0", "false", "no"}

    def _log_scalars(self, args, state, values: Dict[str, float]) -> None:
        log_dir = getattr(args, "logging_dir", None)
        if not log_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            if self._writer is None:
                self._writer = SummaryWriter(log_dir=log_dir)
            for key, value in values.items():
                self._writer.add_scalar(key, value, state.global_step)
            self._writer.flush()
        except Exception as exc:
            print(f"[AURORA] free-eval TensorBoard logging unavailable: {exc}", flush=True)

    def _build_engine(self, model):
        base = _base_model(model)
        processor = getattr(base, "aurora_processor", None)
        if processor is None:
            raise RuntimeError("AURORA processor was not attached to the registered model")
        self._template = get_template("aurora_qwen2_5_omni", processor)
        self._engine = PtEngine.from_model_template(model, self._template, max_batch_size=1)

    @staticmethod
    def _dataset_rows(eval_dataloader) -> Iterable[Dict[str, Any]]:
        dataset = getattr(eval_dataloader, "dataset", None)
        if dataset is None:
            return []
        # With lazy tokenization the dataloader wraps the raw HF dataset in
        # LazyLLMDataset; its normal integer indexing returns encoded tensors,
        # while string indexing exposes raw records. Unwrap the common
        # ``.dataset`` chain so generation receives the original messages and
        # media paths.
        while hasattr(dataset, "dataset") and not isinstance(dataset, dict):
            inner = getattr(dataset, "dataset")
            if inner is dataset:
                break
            dataset = inner
        limit = int(os.environ.get("AURORA_FREE_EVAL_SAMPLES", "8"))
        if limit <= 0:
            limit = len(dataset)
        return (dataset[index] for index in range(min(limit, len(dataset))))

    def _evaluate_row(self, model, row: Dict[str, Any], device):
        base = _base_model(model)
        tokenizer = self._template.tokenizer
        user_messages = copy.deepcopy(row["messages"][:1])
        request = InferRequest(
            messages=user_messages,
            videos=row.get("videos"),
            audios=row.get("audios"),
        )
        max_tokens = int(os.environ.get("AURORA_FREE_EVAL_MAX_TOKENS", "128"))
        response = self._engine.infer(
            [request],
            RequestConfig(max_tokens=max_tokens, temperature=0.0),
            template=self._template,
            use_tqdm=False,
        )[0].choices[0].message.content
        generated_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
        seg_count = sum(token == seg_id for token in generated_ids)
        result = {"seg_emitted": float(seg_count == 1), "seg_valid": float(seg_count == 1)}
        if seg_count != 1:
            result.update({"iou": 0.0, "dice": 0.0})
            return result, response

        full_row = copy.deepcopy(row)
        full_row["messages"] = user_messages + [{"role": "assistant", "content": response}]
        self._template.set_mode("train")
        encoded = self._template.encode(full_row)
        encoded.pop("_extra_kwargs", None)
        batch = _to_device(self._template.data_collator([encoded]), device)
        # The registered template normally owns this hook. Re-registering is
        # idempotent and covers the generation context's temporary removal.
        self._template.register_post_encode_hook([model])
        with torch.no_grad():
            model(**batch)
        predictions = getattr(base, "aurora_last_pred_masks", None)
        targets = batch.get("gt_masks")
        if not predictions or targets is None:
            raise RuntimeError("Generated response did not produce AURORA mask outputs")
        ious = []
        dices = []
        for prediction, target in zip(predictions, targets):
            target = target.to(device=prediction.device, dtype=prediction.dtype)
            for frame_prediction, frame_target in zip(prediction, target):
                iou, dice = _frame_metrics(frame_prediction, frame_target)
                ious.append(iou)
                dices.append(dice)
        result.update({"iou": sum(ious) / max(len(ious), 1), "dice": sum(dices) / max(len(dices), 1)})
        return result, response

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        if (not self._enabled() or self._busy or model is None or eval_dataloader is None
                or not getattr(state, "is_world_process_zero", True)):
            return control
        self._busy = True
        was_training = model.training
        started = time.time()
        try:
            self._build_engine(model)
            device = next(model.parameters()).device
            totals = {"seg_emitted": 0.0, "seg_valid": 0.0, "iou": 0.0, "dice": 0.0}
            samples = 0
            responses = []
            for row in self._dataset_rows(eval_dataloader):
                values, response = self._evaluate_row(model, row, device)
                for key in totals:
                    totals[key] += values[key]
                responses.append(response)
                samples += 1
            if samples:
                metrics = {
                    "free_seg_rate": totals["seg_emitted"] / samples,
                    "free_valid_seg_rate": totals["seg_valid"] / samples,
                    "free_mean_iou": totals["iou"] / samples,
                    "free_mean_dice": totals["dice"] / samples,
                    "free_eval_samples": float(samples),
                    "free_eval_runtime": time.time() - started,
                }
                # ``Trainer.evaluate`` passes its mutable metrics dictionary to
                # callback handlers. Preserve the values for callback consumers;
                # TensorBoard and the JSONL record below are the durable logs.
                callback_metrics = kwargs.get("metrics")
                if isinstance(callback_metrics, dict):
                    callback_metrics.update(metrics)
                self._log_scalars(args, state, metrics)
                if getattr(state, "is_world_process_zero", True):
                    path = os.path.join(args.output_dir, "free_generation_eval.jsonl")
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": state.global_step, **metrics, "responses": responses}, ensure_ascii=False) + "\n")
                print(f"[AURORA] free eval step={state.global_step}: {metrics}", flush=True)
        except Exception as exc:
            # Do not silently turn a broken end-to-end evaluator into a false
            # training success. The first failure is actionable and should
            # stop the run before a long production job proceeds.
            raise RuntimeError(f"AURORA free-generation evaluation failed: {exc}") from exc
        finally:
            model.train(was_training)
            self._busy = False
        return control

