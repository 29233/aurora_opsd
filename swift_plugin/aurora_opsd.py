"""AURORA OPSD trainer: ms-swift 4.x GKD trainer + AURORA mask supervision.

Load this file via ``--external_plugins`` when running ``swift rlhf
--rlhf_type gkd``. It swaps the GKD trainer for a subclass whose
``compute_loss`` additionally runs the AURORA SAM mask branch (BCE + Dice) on
the student's on-policy rollout, so language distillation (JSD against the
privileged teacher view) and mask grounding are trained jointly — mirroring
the Stage-1 SFT contract (CE + 2.0*BCE + 0.5*Dice) with JSD replacing CE.

Design decisions (see the load.md conversation history for the analysis):

- ``lmbda=1`` (pure on-policy) is assumed: every supervised row carries the
  student's own rollout, and the [SEG] token(s) inside that rollout drive the
  mask branch. GT-label rows (``lmbda<1``) still work — the same forward
  finds [SEG] in the GT response instead.
- Rollouts WITHOUT [SEG] contribute the JSD loss only (skip the mask term);
  rollouts with MULTIPLE [SEG] use the FIRST occurrence (the probe showed the
  model emits [SEG] once; the tolerance here just avoids crashes on rare
  degenerate generations). This is the selection-bias trade-off we agreed on:
  the text signal still covers [SEG]-less failures.
- The teacher forward never runs the mask branch: swift's GKD path excludes
  ``labels`` from teacher inputs (``sft_alpha=0`` default), and the AURORA
  wrapper bypasses supervision when ``labels is None``.
- Trainer swap: ``TrainerFactory.TRAINER_MAPPING['gkd']`` is patched at plugin
  import time. Plugins are imported during argument parsing
  (``_import_external_plugins``), strictly before ``TrainerFactory.get_trainer_cls``
  runs in the training pipeline, so the patched entry wins. Guarded by
  ``AURORA_OPSD_MASK_LOSS`` (default on when this plugin is loaded) and only
  active for swift 4.x.

Loss weighting: the JSD term keeps the trainer's own scale; the mask term is
``AURORA_MASK_WEIGHT`` (default 1.0) * (2.0*BCE + 0.5*Dice) — the same BCE/Dice
weights as Stage-1 SFT (env-overridable via ``AURORA_BCE_WEIGHT`` /
``AURORA_DICE_WEIGHT``, read by ``attach_aurora_segmentation``). Initial run
should log both components; if JSD dominates (probe measured unclipped
KL~12.7), raise ``AURORA_MASK_WEIGHT``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    from swift.trainers import disable_gradient_checkpointing
    from swift.trainers.trainer_factory import TrainerFactory
except ImportError as exc:  # pragma: no cover - swift 3.x has no gkd trainer
    raise ImportError(
        "aurora_opsd requires ms-swift >= 4.x (GKD trainer with OPSD "
        "teacher_prompt support). The base env (swift 3.10.1) is not supported."
    ) from exc

# Local imports (plugin may be imported as part of the swift_plugin package or
# standalone via --external_plugins, so try both).
try:
    from .segmentation import SEG_TOKEN
except ImportError:
    from swift_plugin.segmentation import SEG_TOKEN

_MASK_LOSS_ENABLED = os.environ.get("AURORA_OPSD_MASK_LOSS", "1") == "1"
_MASK_WEIGHT = float(os.environ.get("AURORA_OPSD_MASK_WEIGHT", "1.0"))


def _patch_rollout_request_videos_type() -> None:
    """Relax ``RolloutInferRequest.videos`` typing to accept frame lists.

    AURORA rows represent each video as a list of frame image paths
    (``videos=[[f0..f9]]``) — the qwen-omni template deliberately supports
    this ("image list as video", template/templates/qwen.py handles
    ``isinstance(video, list)``), and the SFT path encodes it fine. The RL
    rollout path, however, serializes each sample with dacite into
    ``RolloutInferRequest``, whose ``videos: List[str]`` annotation rejects
    the nested structure before the template ever sees it.

    Widening the annotation to ``List[Any]`` lets the nested frame list flow
    through to the template unchanged. dacite reads types via
    ``get_type_hints`` (i.e. the class ``__annotations__``), so patching the
    dataclass ``Field.type`` attribute has no effect — the annotation itself
    must change. The ``videos`` field is declared on the PARENT class
    ``InferRequest`` (RolloutInferRequest only adds uuid etc.), and
    ``get_type_hints`` resolves inherited annotations from the defining
    class, so the patch must target the parent.
    """
    from swift.infer_engine.protocol import InferRequest, RolloutInferRequest

    from typing import Any, List

    for cls in (RolloutInferRequest, InferRequest):
        ann = cls.__annotations__
        if "videos" in ann and str(ann["videos"]) != "typing.List[typing.Any]":
            ann["videos"] = List[Any]


_patch_rollout_request_videos_type()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP/DeepSpeed/PEFT wrappers to reach the AURORA-annotated model."""
    current = model
    for _ in range(6):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        break
    return current


def _extract_seg_hidden(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], int]:
    """Gather [SEG]-position hidden states from a (possibly [SEG]-less) batch.

    Returns ``(seg_hidden, n_seg)`` where ``seg_hidden`` is ``[n_seg, D]``
    (``None`` when no sample emitted [SEG]) and ``n_seg`` counts total
    occurrences. Only the FIRST [SEG] per row is kept (rows with several
    occurrences are rare degenerate generations; crashing on them — as the
    SFT path deliberately does — would kill an on-policy run).
    """
    base = _unwrap(model)
    seg_token_id = base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if seg_token_id is None or seg_token_id == base.aurora_tokenizer.unk_token_id:
        raise ValueError("[SEG] is missing; pass --new_special_tokens '[SEG]'")
    seg_mask = labels.eq(seg_token_id)
    total = int(seg_mask.sum().item())
    if total == 0:
        return None, 0
    # first [SEG] per row: argmax finds the first True index
    first_idx = seg_mask.int().argmax(dim=-1)  # [B]
    has_seg = seg_mask.any(dim=-1)  # [B]
    rows = torch.nonzero(has_seg, as_tuple=False).squeeze(-1)
    positions = first_idx[rows]
    seg_hidden = hidden_states[rows, positions]  # [n_seg, D]
    return seg_hidden, total


def _mask_loss_from_seg_hidden(
    model: torch.nn.Module,
    seg_hidden: torch.Tensor,
    rows: torch.Tensor,
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run text_hidden_fcs -> SAM on [SEG] hidden states; return (loss, parts).

    ``rows`` maps each seg-hidden row back to its batch index so the right
    frames/GT masks are used. Frames are shared per row (AURORA contract), so
    only the rows that emitted [SEG] contribute — matching the
  skip-if-no-[SEG] decision.
    """
    base = _unwrap(model)
    projector_dtype = next(base.text_hidden_fcs.parameters()).dtype
    text_prompts = base.text_hidden_fcs(seg_hidden.to(dtype=projector_dtype)).float()

    images = batch["sam_pixel_values"].to(
        device=seg_hidden.device, dtype=next(base.sam.image_encoder.parameters()).dtype
    )
    batch_size, frame_count = images.shape[:2]
    with torch.no_grad():
        image_embeddings = base.sam.image_encoder(images.flatten(0, 1))
    decoder_dtype = next(base.sam.mask_decoder.parameters()).dtype
    image_embeddings = image_embeddings.view(batch_size, frame_count, *image_embeddings.shape[1:]).to(decoder_dtype)
    prompt_encoder_dtype = next(base.sam.prompt_encoder.parameters()).dtype

    bce = seg_hidden.new_zeros((), dtype=torch.float32)
    dice = seg_hidden.new_zeros((), dtype=torch.float32)
    for i, batch_index in enumerate(rows.tolist()):
        prompt = text_prompts[i].to(prompt_encoder_dtype).view(1, 1, -1)
        sparse, dense = base.sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=prompt)
        low_res_masks, _ = base.sam.mask_decoder(
            image_embeddings[batch_index],
            image_pe=base.sam.prompt_encoder.get_dense_pe().to(decoder_dtype),
            sparse_prompt_embeddings=sparse.to(decoder_dtype),
            dense_prompt_embeddings=dense.to(decoder_dtype),
            multimask_output=False,
        )
        pred_mask = base.sam.postprocess_masks(
            low_res_masks,
            input_size=tuple(batch["sam_resize_sizes"][batch_index][0]),
            original_size=tuple(batch["sam_original_sizes"][batch_index][0]),
        )[:, 0]
        target = batch["gt_masks"][batch_index].to(device=pred_mask.device, dtype=pred_mask.dtype)
        if target.shape != pred_mask.shape:
            raise ValueError(f"GT/predicted mask shape mismatch: {target.shape} != {pred_mask.shape}")
        from swift_plugin.segmentation import dice_loss

        bce = bce + torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, target)
        dice = dice + dice_loss(pred_mask, target)

    n = len(rows)
    bce = bce / n
    dice = dice / n
    loss = base.aurora_bce_weight * bce + base.aurora_dice_weight * dice
    return loss, {"mask_bce_loss": bce.detach(), "mask_dice_loss": dice.detach(), "mask_loss": loss.detach()}


class AuroraGKDTrainer(GKDTrainer):
    """GKD trainer with AURORA SAM mask supervision on the student rollout."""

    def _try_static_graph(self) -> None:
        """No-op: static graph is requested via ``--ddp_static_graph true``.

        The mask branch calls sam.mask_decoder / text_hidden_fcs directly
        (outside the DDP-wrapped forward), so those parameters fire their DDP
        gradient hooks a second time during the single backward pass.
        ``DistributedDataParallel(static_graph=True)`` makes DDP tolerate
        parameters that contribute to the loss graph outside forward(). The
        flag must be set when DDP is CONSTRUCTED (before any forward), so a
        lazy runtime call is too late — the training script passes
        ``--ddp_static_graph true`` instead.
        """

    def _mask_supervision(self, model, outputs, model_inputs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the AURORA mask loss from the student forward outputs.

        Requires ``output_hidden_states=True`` on the student forward (set in
        ``compute_loss``) and the AURORA supervision tensors to be present in
        ``model_inputs`` (our template's collator puts them there for
        segmentation rows).
        """
        labels = model_inputs["labels"]
        if "sam_pixel_values" not in model_inputs:
            raise ValueError(
                "AURORA OPSD mask supervision enabled but model_inputs lacks "
                "sam_pixel_values; check that rows carry sam_frame_paths/mask_paths "
                "and the aurora template is in use."
            )
        hidden = outputs.hidden_states[-1]
        seg_hidden, total = _extract_seg_hidden(model, hidden, labels)
        if seg_hidden is None:
            zero = hidden.new_zeros((), dtype=torch.float32)
            return zero * 0, {
                "mask_bce_loss": zero.detach(),
                "mask_dice_loss": zero.detach(),
                "mask_loss": zero.detach(),
                "seg_rows": torch.tensor(0.0),
            }
        base = _unwrap(model)
        seg_mask = labels.eq(base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN))
        first_idx = seg_mask.int().argmax(dim=-1)
        has_seg = seg_mask.any(dim=-1)
        rows = torch.nonzero(has_seg, as_tuple=False).squeeze(-1)
        loss, parts = _mask_loss_from_seg_hidden(model, seg_hidden, rows, model_inputs)
        parts["seg_rows"] = torch.tensor(float(len(rows)))
        return loss, parts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """JSD distillation loss (parent) + AURORA mask loss (this class).

        The parent's non-liger path forwards the student WITHOUT labels, so
        the AURORA wrapper (which keys mask supervision on ``labels``) stays
        inert and the parent's JSD loss sees clean logits. We re-run the mask
        branch here against the SAME forward's hidden states — one extra SAM
        pass, no extra LLM forward.
        """
        if not _MASK_LOSS_ENABLED:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        self._try_static_graph()
        model_inputs = inputs["model_inputs"]
        gkd_batch = inputs["gkd_batch"]

        # --- replicate the parent's student forward, but ask for hidden states
        # NOTE: kept in sync with GKDTrainer.compute_loss (non-liger, non-API
        # branch). The teacher side is delegated to the parent by NOT calling
        # it here — instead we call the parent for the JSD part only after
        # re-running the student forward would double the cost. To keep a
        # single student forward we reimplement the parent's loss path below.
        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(model_inputs)
            if not self.use_teacher_api:
                self.prepare_logits_to_keep(inputs["teacher_model_inputs"])
        teacher_labels = inputs["teacher_model_inputs"]["labels"]

        forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
        if self.use_liger_gkd_loss:
            raise ValueError("AURORA mask supervision is incompatible with --use_liger_kernel GKD loss")

        outputs_student = model(**forward_inputs, output_hidden_states=True)
        # The AURORA forward wrapper may or may not be active on this model
        # (it is, via attach_aurora_segmentation at load time): with
        # sam_pixel_values present but labels absent it returns early, so
        # hidden_states survive for our mask branch.

        # --- teacher output (mirrors parent logic)
        if self.use_teacher_api:
            teacher_topk_logprobs = gkd_batch.teacher_topk_logprobs
            teacher_topk_indices = gkd_batch.teacher_topk_indices
            teacher_out = type("TeacherOutputHolder", (), {})()  # placeholder, replaced below
            from swift.rlhf_trainers.gkd_helpers import TeacherOutput

            teacher_out = TeacherOutput(
                topk_logprobs=teacher_topk_logprobs, topk_indices=teacher_topk_indices, labels=teacher_labels
            )
        else:
            t_fwd = {k: v for k, v in inputs["teacher_model_inputs"].items() if k != "labels"}
            if self._is_self_distillation:
                adapter_ctx = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self._teacher_use_disable_adapter
                    else _null_ctx()
                )
                with torch.no_grad(), adapter_ctx:
                    outputs_teacher = model(**t_fwd)
            else:
                with torch.no_grad(), disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs
                ):
                    outputs_teacher = self.teacher_model(**t_fwd)
            from swift.rlhf_trainers.gkd_helpers import TeacherOutput

            teacher_out = TeacherOutput(full_logits=outputs_teacher.logits, labels=teacher_labels)

        jsd_loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, model_inputs["labels"])

        # --- AURORA mask loss on the same student forward
        if "sam_pixel_values" in model_inputs:
            mask_loss, parts = self._mask_supervision(model, outputs_student, model_inputs)
            loss = jsd_loss + _MASK_WEIGHT * mask_loss
            self._aurora_mask_parts = parts
        else:
            loss = jsd_loss
            self._aurora_mask_parts = None

        if return_outputs:
            return (loss, outputs_student)
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # surface the mask-loss components in the trainer's regular logging
        # (transformers Trainer.log is the canonical entry; GKDTrainer
        # overrides it, so we hook here rather than in _log)
        parts = getattr(self, "_aurora_mask_parts", None)
        if parts:
            for k, v in parts.items():
                try:
                    logs[f"aurora/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return super().log(logs, *args, **kwargs)


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --- make AURORA supervision fields survive the rollout round-trip -------------
# ``OnPolicySample.to_template_dict`` forwards only StandardKeys (messages/
# images/videos/audios/tools/objects) to ``template.encode`` — our template's
# ``_encode`` reads ``sam_frame_paths``/``mask_paths`` from the row's extra
# kwargs, which would otherwise be dropped. Patch ``to_template_dict`` to
# forward the AURORA supervision columns from ``extra`` so the training
# forward receives sam_pixel_values/gt_masks.
_AURORA_ROW_KEYS = ("sam_frame_paths", "mask_paths", "class_name", "sample_id")


def _patch_sample_to_template_dict() -> None:
    from swift.rl_core.data import OnPolicySample

    if getattr(OnPolicySample.to_template_dict, "_aurora_patched", False):
        return
    original = OnPolicySample.to_template_dict

    def aurora_to_template_dict(self):
        d = original(self)
        for key in _AURORA_ROW_KEYS:
            if key not in d and key in self.extra:
                d[key] = self.extra[key]
        return d

    aurora_to_template_dict._aurora_patched = True
    OnPolicySample.to_template_dict = aurora_to_template_dict


_patch_sample_to_template_dict()
if _MASK_LOSS_ENABLED:
    _orig_entry = TrainerFactory.TRAINER_MAPPING.get("gkd")
    if _orig_entry and "AuroraGKDTrainer" not in _orig_entry:
        TrainerFactory.TRAINER_MAPPING["gkd"] = (
            "swift_plugin.aurora_opsd.AuroraGKDTrainer"
            if _orig_entry.startswith("swift.")
            else _orig_entry  # already patched by another plugin; don't stack
        )
