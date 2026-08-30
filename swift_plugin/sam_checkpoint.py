"""Save the trainable SAM decoder as a portable checkpoint sidecar."""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file
from transformers import TrainerCallback

from swift.plugin import extra_callbacks


def _base_model(model):
    current = model
    for _ in range(4):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        break
    return current


class AuroraSAMCheckpointCallback(TrainerCallback):
    """Write the trainable SAM mask decoder next to each ms-swift checkpoint."""

    def __init__(self):
        self._model = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Transformers' on_save event omits the model from callback kwargs;
        # retain the wrapped training model from the earlier lifecycle event.
        self._model = model
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        model = model or self._model
        if model is None or not getattr(state, "is_world_process_zero", True):
            return control
        base = _base_model(model)
        sam = getattr(base, "sam", None)
        decoder = getattr(sam, "mask_decoder", None)
        if decoder is None:
            return control

        # PEFT wraps modules_to_save while the default copy is the trainable one.
        trained = getattr(decoder, "modules_to_save", None)
        if trained is not None and "default" in trained:
            decoder = trained["default"]
        state_dict = {key: value.detach().cpu().contiguous() for key, value in decoder.state_dict().items()}
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_file(state_dict, os.path.join(checkpoint_dir, "sam_mask_decoder.safetensors"), metadata={"format": "pt"})
        return control


extra_callbacks.append(AuroraSAMCheckpointCallback())
