from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from swift_plugin.segmentation import load_gt_masks, preprocess_sam_frames
from swift_plugin.segmentation import attach_aurora_segmentation

# ---------------------------------------------------------------------------
# ms-swift version compatibility layer.
#
# ms-swift <= 3.x exposes everything under ``swift.llm`` and registers models
# via a ``get_function`` loader callback. ms-swift >= 4.x flattened the
# package (``swift.model`` / ``swift.template`` / ``swift.dataset``) and
# switched registration to ``ModelLoader`` subclasses. Support both layouts
# from a single plugin file so the same code runs in the base env (swift
# 3.10.1, used by the existing SFT/eval scripts) and the swift46 env (swift
# 4.5.2, used for OPSD/GKD training).
# ---------------------------------------------------------------------------
try:  # ms-swift >= 4.x
    from swift.model.model_meta import Model, ModelGroup, ModelMeta
    from swift.model.model_arch import ModelArch
    from swift.model.register import register_model
    from swift.model.models.qwen import Qwen2_5OmniLoader
    from swift.template.register import register_template
    from swift.template.template_meta import TemplateMeta
    from swift.template.templates.qwen import Qwen2_5OmniTemplate
    from swift.template.template_inputs import StdTemplateInputs

    _SWIFT_MAJOR = 4
except ImportError:  # ms-swift <= 3.x
    from swift.llm import Model, ModelGroup, ModelMeta, TemplateMeta, register_model, register_template
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_omni
    from swift.llm.model.model_arch import ModelArch
    from swift.llm.template.template.qwen import Qwen2_5OmniTemplate
    from swift.llm.template.template_inputs import StdTemplateInputs

    _SWIFT_MAJOR = 3


# Transformers 4.57 calls unwrap_model(..., keep_torch_compile=...), while
# the environment's Accelerate may still expose the older signature.
try:
    import accelerate
    import inspect

    if "keep_torch_compile" not in inspect.signature(accelerate.Accelerator.unwrap_model).parameters:
        _aurora_unwrap_model = accelerate.Accelerator.unwrap_model

        def _aurora_compat_unwrap_model(self, model, keep_torch_compile=False):
            return _aurora_unwrap_model(self, model)

        accelerate.Accelerator.unwrap_model = _aurora_compat_unwrap_model
except Exception:
    pass


MODEL_TYPE = "aurora_qwen2_5_omni"
TEMPLATE_TYPE = "aurora_qwen2_5_omni"


def _attach_aurora(model, processor):
    """Attach the AURORA SAM segmentation head to a loaded Omni model."""
    if model is None:
        return
    model.aurora_processor = processor
    checkpoint = os.environ.get("AURORA_SAM_CHECKPOINT", "")
    attach_aurora_segmentation(model, processor, checkpoint)


if _SWIFT_MAJOR >= 4:

    class AuroraQwen2_5OmniLoader(Qwen2_5OmniLoader):
        """Qwen2_5OmniLoader + AURORA SAM head attachment.

        The 4.x loader class takes over everything the old
        ``get_model_tokenizer_qwen2_5_omni`` callback did (config tweaks,
        thinker/talker wiring, processor patching); we only wrap ``load`` to
        attach the segmentation head to whatever it produces.
        """

        def load(self):
            model, processor = super().load()
            _attach_aurora(model, processor)
            return model, processor

    _AURORA_LOADER = AuroraQwen2_5OmniLoader
else:

    def get_model_tokenizer_aurora_qwen2_5_omni(model_dir, *args, **kwargs):
        model, processor = get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs)
        _attach_aurora(model, processor)
        return model, processor

    _AURORA_LOADER = get_model_tokenizer_aurora_qwen2_5_omni


if _SWIFT_MAJOR >= 4:
    _MODEL_META_KWARGS = dict(
        loader=_AURORA_LOADER,
        template=TEMPLATE_TYPE,
    )
else:
    _MODEL_META_KWARGS = dict(
        get_function=_AURORA_LOADER,
        template=TEMPLATE_TYPE,
    )


register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([
            Model("Qwen/Qwen2.5-Omni-3B",
                  "Qwen/Qwen2.5-Omni-3B",
                  model_path="/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-3B"),
            Model("Qwen/Qwen2.5-Omni-7B",
                  "Qwen/Qwen2.5-Omni-7B",
                  model_path="/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-7B"),
        ])],
        model_arch=ModelArch.qwen2_5_omni,
        is_multimodal=True,
        architectures=["Qwen2_5OmniModel", "Qwen2_5OmniForConditionalGeneration"],
        requires=["transformers>=4.50", "soundfile", "qwen_omni_utils", "decord"],
        tags=["vision", "video", "audio"],
        additional_saved_files=["spk_dict.pt"],
        **_MODEL_META_KWARGS,
    )
)


class AuroraQwen2_5OmniTemplate(Qwen2_5OmniTemplate):
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        extras = inputs.extra_kwargs
        frame_paths = extras.get("sam_frame_paths")
        mask_paths = extras.get("mask_paths")
        # SAM supervision belongs only to the training forward. In
        # ms-swift's PtEngine generation mode (used by predict_with_generate)
        # validation rows retain these metadata fields, but encoding them
        # would perform needless image/mask preprocessing and pass extra
        # tensors through the LLM generation path.
        if not self.is_training:
            # Generation passes the encoded row's extra kwargs directly to
            # ``model.generate``. Strip AURORA-only metadata so the native
            # PtEngine path receives only model-supported arguments.
            for key in ("sam_frame_paths", "mask_paths", "class_name", "sample_id"):
                extras.pop(key, None)
            return encoded
        if frame_paths is None and mask_paths is None:
            return encoded
        if not frame_paths or mask_paths is None:
            raise ValueError("sam_frame_paths and mask_paths must be provided together")
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        sam_pixels, resize_sizes, original_sizes = preprocess_sam_frames(frame_paths, num_frames)
        gt_masks = load_gt_masks(mask_paths, original_sizes[0], num_frames)
        encoded.update(
            sam_pixel_values=sam_pixels,
            gt_masks=gt_masks,
            sam_resize_sizes=resize_sizes,
            sam_original_sizes=original_sizes,
        )
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # The native Omni hook converts media tokens to inputs_embeds and
        # intentionally returns only model-facing keys. Preserve AURORA's
        # per-sample supervision tensors across that hook as well.
        result = super()._post_encode(model, inputs)
        for key in ("sam_pixel_values", "gt_masks", "sam_resize_sizes", "sam_original_sizes"):
            if key in inputs:
                result[key] = inputs[key]
        return result

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._data_collator_mm_data(batch)
        samples = [item for item in batch if item.get("sam_pixel_values") is not None]
        if not samples:
            return result
        if len(samples) != len(batch):
            raise ValueError("Do not mix segmentation and non-segmentation examples in one batch")
        shapes = {tuple(item["gt_masks"].shape[-2:]) for item in samples}
        if len(shapes) != 1:
            raise ValueError("Variable mask sizes require per_device_train_batch_size=1")
        result["sam_pixel_values"] = torch.stack([item["sam_pixel_values"] for item in samples])
        result["gt_masks"] = [item["gt_masks"] for item in samples]
        result["sam_resize_sizes"] = [item["sam_resize_sizes"] for item in samples]
        result["sam_original_sizes"] = [item["sam_original_sizes"] for item in samples]
        return result


register_template(
    TemplateMeta(
        TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        # The original AURORA Qwen2.5-Omni Stage-1 message builder has no
        # system turn. Keep system support available for explicit callers,
        # but do not inject ms-swift's default system text implicitly.
        default_system=None,
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=AuroraQwen2_5OmniTemplate,
    )
)
