from __future__ import annotations

import os
from contextlib import nullcontext
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.segment_anything import build_sam_vit_h
from model.segment_anything.utils.transforms import ResizeLongestSide


SEG_TOKEN = "[SEG]"
SAM_IMAGE_SIZE = 224
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class MaskDecoderSaveProxy(nn.Module):
    """Give SAM's keyword-oriented decoder a PEFT-saveable first argument."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, image_embeddings=None, *args, **kwargs):
        return self.decoder(image_embeddings=image_embeddings, *args, **kwargs)


def preprocess_sam_frames(
    frame_paths: Sequence[str], num_frames: int
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
    paths = list(frame_paths)
    if not paths:
        raise ValueError("sam_frame_paths must contain at least one frame")
    while len(paths) < num_frames:
        paths.append(paths[-1])
    paths = paths[:num_frames]

    transform = ResizeLongestSide(SAM_IMAGE_SIZE)
    tensors: List[torch.Tensor] = []
    resize_sizes: List[Tuple[int, int]] = []
    original_sizes: List[Tuple[int, int]] = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_sizes.append((image.shape[0], image.shape[1]))
        resized = transform.apply_image(image)
        resize_sizes.append((resized.shape[0], resized.shape[1]))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = (tensor - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
        height, width = tensor.shape[-2:]
        tensors.append(F.pad(tensor, (0, SAM_IMAGE_SIZE - width, 0, SAM_IMAGE_SIZE - height)))
    return torch.stack(tensors), resize_sizes, original_sizes


def load_gt_masks(mask_paths: Sequence[str], original_size: Tuple[int, int], num_frames: int) -> torch.Tensor:
    paths = list(mask_paths)
    while len(paths) < num_frames and paths:
        paths.append(paths[-1])
    masks: List[torch.Tensor] = []
    for path in paths[:num_frames]:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.shape != original_size:
            mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(torch.from_numpy((mask > 127).astype(np.float32)))
    while len(masks) < num_frames:
        masks.append(torch.zeros(original_size, dtype=torch.float32))
    return torch.stack(masks)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probabilities * targets).sum(-1)
    denominator = probabilities.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1e-6) / (denominator + 1e-6)).mean()


def attach_aurora_segmentation(model: nn.Module, processor: Any, sam_checkpoint: str) -> nn.Module:
    if not sam_checkpoint or not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            "Set AURORA_SAM_CHECKPOINT to the SAM ViT-H checkpoint; "
            f"received {sam_checkpoint!r}"
        )

    original_forward = model.forward
    model.sam = build_sam_vit_h(sam_checkpoint)
    model.sam.requires_grad_(False)
    model.sam.mask_decoder = MaskDecoderSaveProxy(model.sam.mask_decoder)
    model.sam.mask_decoder.requires_grad_(True)
    model.sam.image_encoder.eval()
    model.sam.prompt_encoder.eval()
    model.sam.mask_decoder.train()

    hidden_size = model.thinker.config.text_config.hidden_size
    model.text_hidden_fcs = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, 256),
        nn.Dropout(0.0),
    )
    model.aurora_tokenizer = processor.tokenizer
    model.aurora_ce_weight = _env_float("AURORA_CE_WEIGHT", 1.0)
    model.aurora_bce_weight = _env_float("AURORA_BCE_WEIGHT", 2.0)
    model.aurora_dice_weight = _env_float("AURORA_DICE_WEIGHT", 0.5)

    def aurora_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        sam_pixel_values=None,
        gt_masks=None,
        sam_resize_sizes=None,
        sam_original_sizes=None,
        **kwargs,
    ):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        outputs = original_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        # Generation has no labels and must bypass the supervised mask branch.
        # Free-generation evaluation reruns the generated sequence with labels
        # afterwards to obtain the [SEG] hidden state and mask.
        if sam_pixel_values is None or labels is None:
            return outputs
        

        seg_token_id = self.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        if seg_token_id is None or seg_token_id == self.aurora_tokenizer.unk_token_id:
            raise ValueError("[SEG] is missing; pass --new_special_tokens '[SEG]'")
        seg_mask = labels.eq(seg_token_id)
        seg_counts = seg_mask.sum(dim=-1)
        if not torch.all(seg_counts == 1):
            raise ValueError(f"Each SFT answer must contain exactly one [SEG], got {seg_counts.tolist()}")

        hidden = outputs.hidden_states[-1]
        projector_dtype = next(self.text_hidden_fcs.parameters()).dtype
        text_prompts = self.text_hidden_fcs(hidden[seg_mask].to(dtype=projector_dtype)).float()
        image_encoder_dtype = next(self.sam.image_encoder.parameters()).dtype
        decoder_dtype = next(self.sam.mask_decoder.parameters()).dtype
        prompt_encoder_dtype = next(self.sam.prompt_encoder.parameters()).dtype
        images = sam_pixel_values.to(device=hidden.device, dtype=image_encoder_dtype)
        batch_size, frame_count = images.shape[:2]
        encoder_context = torch.autocast("cuda", enabled=False) if images.is_cuda else nullcontext()
        with encoder_context, torch.no_grad():
            image_embeddings = self.sam.image_encoder(images.flatten(0, 1))
        image_embeddings = image_embeddings.view(batch_size, frame_count, *image_embeddings.shape[1:]).to(decoder_dtype)

        bce = hidden.new_zeros((), dtype=torch.float32)
        dice = hidden.new_zeros((), dtype=torch.float32)
        predicted_masks = []
        for batch_index in range(batch_size):
            prompt = text_prompts[batch_index].to(prompt_encoder_dtype).view(1, 1, -1)
            sparse, dense = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=prompt
            )
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings[batch_index],
                image_pe=self.sam.prompt_encoder.get_dense_pe().to(decoder_dtype),
                sparse_prompt_embeddings=sparse.to(decoder_dtype),
                dense_prompt_embeddings=dense.to(decoder_dtype),
                multimask_output=False,
            )
            pred_mask = self.sam.postprocess_masks(
                low_res_masks,
                input_size=tuple(sam_resize_sizes[batch_index][0]),
                original_size=tuple(sam_original_sizes[batch_index][0]),
            )[:, 0]
            target = gt_masks[batch_index].to(device=pred_mask.device, dtype=pred_mask.dtype)
            if target.shape != pred_mask.shape:
                raise ValueError(f"GT/predicted mask shape mismatch: {target.shape} != {pred_mask.shape}")
            bce = bce + F.binary_cross_entropy_with_logits(pred_mask, target)
            dice = dice + dice_loss(pred_mask, target)
            predicted_masks.append(pred_mask)

        bce = bce / batch_size
        dice = dice / batch_size
        ce = outputs.loss.float() if outputs.loss is not None else hidden.new_zeros((), dtype=torch.float32)
        mask_loss = self.aurora_bce_weight * bce + self.aurora_dice_weight * dice
        outputs.loss = self.aurora_ce_weight * ce + mask_loss
        outputs.hidden_states = None
        self.aurora_last_losses = {
            "ce_loss": ce.detach(),
            "mask_bce_loss": bce.detach(),
            "mask_dice_loss": dice.detach(),
            "mask_loss": mask_loss.detach(),
            "loss": outputs.loss.detach(),
        }
        self.aurora_last_pred_masks = [mask.detach() for mask in predicted_masks]
        if os.environ.get("AURORA_DEBUG", "0") == "1":
            debug_step = int(getattr(self, "_aurora_debug_step", 0)) + 1
            self._aurora_debug_step = debug_step
            print(
                f"[AURORA] joint loss forward={debug_step}: "
                f"ce={float(ce.detach()):.8f} bce={float(bce.detach()):.8f} "
                f"dice={float(dice.detach()):.8f} total={float(outputs.loss.detach()):.8f}",
                flush=True,
            )
        return outputs

    model.forward = MethodType(aurora_forward, model)
    return model


def attach_aurora_videollama2_segmentation(model: nn.Module, processor: Any,
                                           sam_checkpoint: str) -> nn.Module:
    """Attach the AURORA SAM head to VideoLLaMA2's ``images`` interface.

    VideoLLaMA2 has no ``thinker`` submodel and accepts multimodal inputs via
    negative token ids plus an ``images`` list.  Keeping this wrapper separate
    avoids changing the tested Qwen2.5-Omni path while retaining the same
    frozen-SAM/joint-loss contract.
    """
    if not sam_checkpoint or not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            "Set AURORA_SAM_CHECKPOINT to the SAM ViT-H checkpoint; "
            f"received {sam_checkpoint!r}"
        )

    original_forward = model.forward
    model.sam = build_sam_vit_h(sam_checkpoint)
    model.sam.requires_grad_(False)
    model.sam.mask_decoder = MaskDecoderSaveProxy(model.sam.mask_decoder)
    model.sam.mask_decoder.requires_grad_(True)
    model.sam.image_encoder.eval()
    model.sam.prompt_encoder.eval()
    model.sam.mask_decoder.train()

    hidden_size = int(getattr(model.config, "hidden_size"))
    model.text_hidden_fcs = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, 256),
        nn.Dropout(0.0),
    )
    model.aurora_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model._aurora_seg_token_idx = model.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    model.aurora_ce_weight = _env_float("AURORA_CE_WEIGHT", 1.0)
    model.aurora_bce_weight = _env_float("AURORA_BCE_WEIGHT", 2.0)
    model.aurora_dice_weight = _env_float("AURORA_DICE_WEIGHT", 0.5)

    def aurora_forward(self, labels=None, sam_pixel_values=None, gt_masks=None,
                       sam_resize_sizes=None, sam_original_sizes=None, **kwargs):
        # ``images`` remains in kwargs and is consumed by the native
        # VideoLLaMA2 forward.  Generation has no labels and bypasses SAM.
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        has_supervision = sam_pixel_values is not None and labels is not None
        seg_token_id = None
        if has_supervision:
            seg_token_id = self.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
            if seg_token_id is None or seg_token_id == self.aurora_tokenizer.unk_token_id:
                raise ValueError("[SEG] is missing; pass --new_special_tokens '[SEG]'")
            # ms-swift adds new tokens after the model loader attaches this
            # wrapper. Refresh the id on every supervised call so the
            # underlying Qwen forward can build its aligned mask.
            self._aurora_seg_token_idx = seg_token_id
        if os.environ.get("AURORA_DEBUG", "0") == "1":
            print(
                "[AURORA] VideoLLaMA2 forward inputs: "
                f"input_ids={getattr(kwargs.get('input_ids'), 'shape', None)} "
                f"inputs_embeds={getattr(kwargs.get('inputs_embeds'), 'shape', None)} "
                f"labels={getattr(labels, 'shape', None)} "
                f"images={'images' in kwargs}",
                flush=True,
            )
        outputs = original_forward(
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if not has_supervision:
            return outputs

        # VideoLLaMA2 inserts media embeddings into the sequence, so the raw
        # input labels are shorter than the returned hidden states.  Its Qwen
        # forward records the post-expansion mask for us; retain a text-only
        # fallback for callers that provide pre-expanded inputs_embeds.
        seg_mask = getattr(self, "_aurora_expanded_seg_token_mask", None)
        if seg_mask is None:
            seg_mask = labels.eq(seg_token_id)
        if seg_mask.shape != outputs.hidden_states[-1].shape[:2]:
            raise ValueError(
                "AURORA [SEG] mask does not align with multimodal hidden states: "
                f"mask={tuple(seg_mask.shape)} hidden={tuple(outputs.hidden_states[-1].shape[:2])}"
            )
        seg_counts = seg_mask.sum(dim=-1)
        if not torch.all(seg_counts == 1):
            raise ValueError(f"Each SFT answer must contain exactly one [SEG], got {seg_counts.tolist()}")

        hidden = outputs.hidden_states[-1]
        projector_dtype = next(self.text_hidden_fcs.parameters()).dtype
        text_prompts = self.text_hidden_fcs(hidden[seg_mask].to(dtype=projector_dtype)).float()
        image_encoder_dtype = next(self.sam.image_encoder.parameters()).dtype
        decoder_dtype = next(self.sam.mask_decoder.parameters()).dtype
        prompt_encoder_dtype = next(self.sam.prompt_encoder.parameters()).dtype
        images = sam_pixel_values.to(device=hidden.device, dtype=image_encoder_dtype)
        batch_size, frame_count = images.shape[:2]
        encoder_context = torch.autocast("cuda", enabled=False) if images.is_cuda else nullcontext()
        with encoder_context, torch.no_grad():
            image_embeddings = self.sam.image_encoder(images.flatten(0, 1))
        image_embeddings = image_embeddings.view(batch_size, frame_count, *image_embeddings.shape[1:]).to(decoder_dtype)

        bce = hidden.new_zeros((), dtype=torch.float32)
        dice = hidden.new_zeros((), dtype=torch.float32)
        predicted_masks = []
        for batch_index in range(batch_size):
            prompt = text_prompts[batch_index].to(prompt_encoder_dtype).view(1, 1, -1)
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=prompt)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings[batch_index],
                image_pe=self.sam.prompt_encoder.get_dense_pe().to(decoder_dtype),
                sparse_prompt_embeddings=sparse.to(decoder_dtype),
                dense_prompt_embeddings=dense.to(decoder_dtype),
                multimask_output=False,
            )
            pred_mask = self.sam.postprocess_masks(
                low_res_masks,
                input_size=tuple(sam_resize_sizes[batch_index][0]),
                original_size=tuple(sam_original_sizes[batch_index][0]),
            )[:, 0]
            target = gt_masks[batch_index].to(device=pred_mask.device, dtype=pred_mask.dtype)
            if target.shape != pred_mask.shape:
                raise ValueError(f"GT/predicted mask shape mismatch: {target.shape} != {pred_mask.shape}")
            bce = bce + F.binary_cross_entropy_with_logits(pred_mask, target)
            dice = dice + dice_loss(pred_mask, target)
            predicted_masks.append(pred_mask)

        bce = bce / batch_size
        dice = dice / batch_size
        ce = outputs.loss.float() if outputs.loss is not None else hidden.new_zeros((), dtype=torch.float32)
        mask_loss = self.aurora_bce_weight * bce + self.aurora_dice_weight * dice
        outputs.loss = self.aurora_ce_weight * ce + mask_loss
        outputs.hidden_states = None
        self.aurora_last_losses = {
            "ce_loss": ce.detach(), "mask_bce_loss": bce.detach(),
            "mask_dice_loss": dice.detach(), "mask_loss": mask_loss.detach(),
            "loss": outputs.loss.detach(),
        }
        self.aurora_last_pred_masks = [mask.detach() for mask in predicted_masks]
        return outputs

    model.forward = MethodType(aurora_forward, model)
    return model


def assert_trainable_boundary(model: nn.Module) -> Dict[str, int]:
    groups = {
        "sam_image_encoder": 0,
        "sam_mask_decoder": 0,
        "text_hidden_fcs": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if "sam.image_encoder" in name:
            groups["sam_image_encoder"] += count
        elif "sam.mask_decoder" in name:
            groups["sam_mask_decoder"] += count
        elif "text_hidden_fcs" in name:
            groups["text_hidden_fcs"] += count
        else:
            groups["other"] += count
    if groups["sam_image_encoder"]:
        raise RuntimeError("SAM image encoder must remain frozen")
    if not groups["sam_mask_decoder"] or not groups["text_hidden_fcs"]:
        raise RuntimeError(f"SAM decoder/projector are not trainable: {groups}")
    return groups
