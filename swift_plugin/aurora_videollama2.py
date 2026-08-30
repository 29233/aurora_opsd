"""ms-swift adapter for the Qwen2-based VideoLLaMA2.1-7B-AV checkpoint.

The upstream model is a Transformers-compatible causal LM, but its media
contract is custom: ``<video>``/``<audio>`` become negative token ids and the
corresponding tensors are passed in ``images``.  This module keeps that
contract local to a new model/template registration.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.llm import (Model, ModelGroup, ModelMeta, MultiModelKeys, TemplateMeta,
                        register_model, register_model_arch, register_template)
from swift.llm.template.base import Template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context

from swift_plugin.segmentation import (load_gt_masks, preprocess_sam_frames,
                                       attach_aurora_videollama2_segmentation)


MODEL_TYPE = "aurora_videollama2_qwen2"
TEMPLATE_TYPE = "aurora_videollama2"
ARCH_TYPE = "aurora_videollama2_qwen2_arch"
VIDEO_TOKEN_INDEX = -201
AUDIO_TOKEN_INDEX = -202


@dataclass
class VideoLLaMA2Processor:
    """Minimal processor facade expected by ms-swift's Template base class."""

    tokenizer: Any
    model_info: Any
    model_meta: Any
    image_processor: Any = None


def _repo_model_path() -> str:
    return os.environ.get(
        "VIDEOLLAMA2_MODEL_PATH",
        "/mnt/tbo/lvyf/AURORA/AURORA-main/models/VideoLLaMA2.1-7B-AV",
    )


def _source_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VideoLLaMA2av"))


def _restore_beats_weight_norm(model: Any, audio_path: str) -> None:
    """Restore legacy ``weight_g/weight_v`` keys under torch 2.7.

    VideoLLaMA2 was released with the pre-parametrization weight_norm state
    dict.  Newer PyTorch exposes the same layer as ``original0/original1``;
    Transformers therefore reports the two keys as missing unless we copy
    them from the bundled BEATs tower explicitly.
    """
    if not os.path.isfile(audio_path):
        return
    layer = getattr(getattr(getattr(model, "model", None), "audio_tower", None), "encoder", None)
    pos_conv = getattr(layer, "pos_conv", None)
    layer = pos_conv[0] if pos_conv is not None else None
    if layer is None or not hasattr(layer, "weight_g") or not hasattr(layer, "weight_v"):
        return
    try:
        g = v = None
        # Prefer the main checkpoint's modern parametrization names.
        model_dir = getattr(model, "model_dir", None) or getattr(model, "_aurora_model_dir", None)
        index_path = os.path.join(model_dir, "model.safetensors.index.json") if model_dir else ""
        if os.path.isfile(index_path):
            import json
            from safetensors import safe_open
            weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]
            kg = "model.audio_tower.encoder.pos_conv.0.parametrizations.weight.original0"
            kv = "model.audio_tower.encoder.pos_conv.0.parametrizations.weight.original1"
            shard = weight_map.get(kg) or weight_map.get(kv)
            if shard:
                with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as handle:
                    if kg in handle.keys():
                        g = handle.get_tensor(kg)
                    if kv in handle.keys():
                        v = handle.get_tensor(kv)
        if g is None or v is None:
            state = torch.load(audio_path, map_location="cpu")
            state = state.get("model", state)
            g = state.get("model.audio_tower.encoder.pos_conv.0.weight_g", state.get("encoder.pos_conv.0.weight_g"))
            v = state.get("model.audio_tower.encoder.pos_conv.0.weight_v", state.get("encoder.pos_conv.0.weight_v"))
        if g is not None and v is not None:
            with torch.no_grad():
                layer.weight_g.copy_(g.to(layer.weight_g))
                layer.weight_v.copy_(v.to(layer.weight_v))
    except Exception as exc:
        print(f"[AURORA] warning: could not restore BEATs weight_norm: {exc}", flush=True)


def get_model_tokenizer_aurora_videollama2(model_dir, model_info, model_kwargs,
                                           load_model=True, **kwargs):
    """Load the upstream VideoLLaMA2 class through the normal ms-swift hook."""
    source_root = _source_root()
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from videollama2.model.videollama2_qwen2 import (Videollama2Qwen2Config,
                                                      Videollama2Qwen2ForCausalLM)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if not isinstance(config, Videollama2Qwen2Config):
        config = Videollama2Qwen2Config.from_dict(config.to_dict())
    # Resolve both machine-local paths from the released config.
    siglip = os.environ.get(
        "VIDEOLLAMA2_SIGLIP_PATH",
        "/mnt/tbo/lvyf/AURORA/AURORA-main/models/siglip-so400m-patch14-384",
    )
    audio = os.environ.get("VIDEOLLAMA2_AUDIO_TOWER_PATH", os.path.join(model_dir, "audio_tower.bin"))
    config.mm_vision_tower = siglip
    config.mm_audio_tower = audio
    # ms-swift records the requested attention implementation on the config
    # (see AttnImpl.update_attn_impl); re-building the config above would drop
    # it, so forward it into from_pretrained explicitly.
    attn_impl = kwargs.get("attn_impl")
    if attn_impl:
        model_kwargs = dict(model_kwargs or {})
        model_kwargs["attn_implementation"] = attn_impl
    model_info.config = config
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not load_model:
        return None, VideoLLaMA2Processor(tokenizer, model_info, kwargs.get("model_meta"))

    # ``model_kwargs`` is prepared by ms-swift (device map, dtype, attention
    # implementation, and zero3-compatible loading options).
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("low_cpu_mem_usage", True)
    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, **model_kwargs)
    model._aurora_model_dir = model_dir
    _restore_beats_weight_norm(model, audio)
    # Match the agreed AURORA boundary: VideoLLaMA2 media encoders are
    # inference-only; the projectors, LoRA/embedding/SAM modules are the SFT
    # trainable surface (the original train_ds.py leaves mm_projector and
    # mm_projector_a trainable and applies LoRA inside the audio tower too).
    model.model.vision_tower.requires_grad_(False)
    model.model.audio_tower.requires_grad_(False)
    processor = VideoLLaMA2Processor(tokenizer, model_info, kwargs.get("model_meta"))
    processor.image_processor = model.get_vision_tower().image_processor
    model.aurora_processor = processor
    sam_checkpoint = os.environ.get("AURORA_SAM_CHECKPOINT", "")
    if sam_checkpoint:
        attach_aurora_videollama2_segmentation(model, processor, sam_checkpoint)
    return model, processor


def _load_video(paths: Any, processor: Any, num_frames: int) -> torch.Tensor:
    from videollama2.mm_utils import process_video
    if isinstance(paths, (tuple, list)):
        paths = list(paths)
        if not paths:
            raise ValueError("VideoLLaMA2 video input is empty")
        while len(paths) < num_frames:
            paths.append(paths[-1])
        paths = paths[:num_frames]
    # The original AURORA dataset helper passes aspect_ratio=None (no
    # expand2square padding); keep the same contract here.
    return process_video(paths, processor, aspect_ratio=None, num_frames=num_frames).to(torch.bfloat16)


def _load_audio(path: str) -> torch.Tensor:
    # Mirror the original ``process_audio_from_video`` helper: convert the raw
    # waveform (10-second REFAVS clips) to the kaldi fbank expected by BEATs.
    # ``process_audio_file`` pads every clip to a fixed 30 seconds, which
    # dilutes the audio embedding with 20 seconds of silence (67% of frames)
    # and the downstream padding mask stays all-False.
    import numpy as np
    import soundfile as sf
    import torchaudio.compliance.kaldi as ta_kaldi

    sr = 16000
    wav, file_sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)).float(),
            orig_freq=file_sr, new_freq=sr,
        ).numpy()
    waveform = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0) * 2 ** 15
    fbank = ta_kaldi.fbank(
        waveform, num_mel_bins=128, sample_frequency=sr,
        frame_length=25, frame_shift=10,
    ).to(torch.bfloat16)
    return fbank.unsqueeze(0)


class AuroraVideoLLaMA2Template(Template):
    """ChatML formatting plus VideoLLaMA2 media and SAM supervision fields."""

    use_model = False
    load_images = False
    # VideoLLaMA2's embedding-only generate path returns completion ids.  The
    # default prompt slicing would incorrectly drop the first prompt-length
    # completion tokens because the expanded media prompt has no input ids.
    skip_prompt = False
    placeholder_tokens = []

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        num_frames = int(os.environ.get("VIDEOLLAMA2_NUM_FRAMES", "10"))
        media = inputs.extra_kwargs.setdefault("_videollama_images", [])
        if media_type == "video":
            video = _load_video(inputs.videos[index], self.processor.image_processor, num_frames)
            media.append((video, "video"))
            return [[VIDEO_TOKEN_INDEX]]
        if media_type == "audio":
            audio = _load_audio(inputs.audios[index])
            media.append((audio, "audio"))
            return [[AUDIO_TOKEN_INDEX]]
        raise ValueError(f"Unsupported VideoLLaMA2 media type: {media_type}")

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        extras = inputs.extra_kwargs
        media = extras.pop("_videollama_images", None)
        if media:
            extras["images"] = media
        if not self.is_training:
            for key in ("sam_frame_paths", "mask_paths", "class_name", "sample_id"):
                extras.pop(key, None)
            return encoded
        frame_paths = extras.get("sam_frame_paths")
        mask_paths = extras.get("mask_paths")
        if frame_paths is None and mask_paths is None:
            return encoded
        if not frame_paths or mask_paths is None:
            raise ValueError("sam_frame_paths and mask_paths must be provided together")
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        sam_pixels, resize_sizes, original_sizes = preprocess_sam_frames(frame_paths, num_frames)
        gt_masks = load_gt_masks(mask_paths, original_sizes[0], num_frames)
        extras.update({
            "sam_pixel_values": sam_pixels,
            "gt_masks": gt_masks,
            "sam_resize_sizes": resize_sizes,
            "sam_original_sizes": original_sizes,
        })
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        extras = [item.get("_extra_kwargs", {}) for item in batch]
        result: Dict[str, Any] = {}
        media = [item.get("images") for item in extras]
        if any(value is not None for value in media):
            if not all(value is not None for value in media):
                raise ValueError("Do not mix media and text-only examples in a VideoLLaMA2 batch")
            # The upstream implementation indexes ``images`` once per batch
            # row, but does not carry a separate sample dimension inside each
            # row.  Keep the documented SFT batch-size-one contract explicit;
            # this also prevents accidentally mixing media from two samples.
            if len(media) != 1:
                raise ValueError("VideoLLaMA2 SFT currently requires batch_size=1")
            result["images"] = media[0]
        sam_pixels = [item.get("sam_pixel_values") for item in extras]
        if any(value is not None for value in sam_pixels):
            if not all(value is not None for value in sam_pixels):
                raise ValueError("Do not mix SAM and non-SAM examples in one batch")
            result["sam_pixel_values"] = torch.stack(sam_pixels)
            result["gt_masks"] = [item["gt_masks"] for item in extras]
            result["sam_resize_sizes"] = [item["sam_resize_sizes"] for item in extras]
            result["sam_original_sizes"] = [item["sam_original_sizes"] for item in extras]
        return result

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # The upstream forward must see input_ids and images together.  Some
        # ms-swift versions precompute ``inputs_embeds`` before this hook;
        # discard that representation so VideoLLaMA2 can expand media tokens
        # and labels in one place (and expose the aligned [SEG] mask).
        inputs.pop("inputs_embeds", None)
        return inputs


register_model_arch(
    MultiModelKeys(
        arch_name=ARCH_TYPE,
        module_list="model.layers",
        mlp="model.layers.{}.mlp",
        down_proj="model.layers.{}.mlp.down_proj",
        attention="model.layers.{}.self_attn",
        q_proj="model.layers.{}.self_attn.q_proj",
        k_proj="model.layers.{}.self_attn.k_proj",
        v_proj="model.layers.{}.self_attn.v_proj",
        o_proj="model.layers.{}.self_attn.o_proj",
        embedding="model.embed_tokens",
        lm_head="lm_head",
        language_model="model",
        aligner=["model.mm_projector", "model.mm_projector_a"],
        vision_tower=["model.vision_tower", "model.audio_tower"],
    )
)

register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([Model("VideoLLaMA2.1-7B-AV", "VideoLLaMA2.1-7B-AV")])],
        TEMPLATE_TYPE,
        get_model_tokenizer_aurora_videollama2,
        model_arch=ARCH_TYPE,
        architectures=["Videollama2Qwen2ForCausalLM"],
        is_multimodal=True,
        requires=["timm", "einops", "soundfile", "librosa", "torchaudio"],
        tags=["vision", "video", "audio"],
    )
)

register_template(
    TemplateMeta(
        TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system=None,
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=AuroraVideoLLaMA2Template,
    )
)
