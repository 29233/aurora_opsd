#!/usr/bin/env python3
"""Free-generation evaluation for an AURORA VideoLLaMA2 checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "/mnt/tbo/lvyf/AURORA/AURORA-main/models/VideoLLaMA2.1-7B-AV"))
    parser.add_argument("--checkpoint", required=True, help="ms-swift checkpoint containing adapter_model.safetensors")
    parser.add_argument("--data-dir", default=str(root / "outputs/data/refavs"), help="Directory containing refavs_test_{s,u,n}.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation directory; defaults to outputs/eval/videollama2/<checkpoint-name>",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--splits", nargs="+", default=["s", "u", "n"])
    args = parser.parse_args()
    default_data_dir = root / "outputs/data/refavs"
    legacy_data_dir = root / "outputs"
    if args.data_dir == str(default_data_dir) and not (default_data_dir / "refavs_test_s.jsonl").is_file():
        if (legacy_data_dir / "refavs_test_s.jsonl").is_file():
            print(f"Warning: new data directory is missing; using legacy directory {legacy_data_dir}", flush=True)
            args.data_dir = str(legacy_data_dir)
    if args.output_dir is None:
        checkpoint_name = Path(args.checkpoint).name.replace("/", "_")
        args.output_dir = str(root / "outputs/eval/videollama2" / checkpoint_name)

    swift_root = os.environ.get("MS_SWIFT_ROOT", "/mnt/tbo/lvyf/cate-pred-embedding")
    for path in (root, root / "VideoLLaMA2av", Path(swift_root)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault(
        "AURORA_SAM_CHECKPOINT",
        "/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth",
    )
    import swift_plugin.aurora_videollama2  # noqa: F401
    from swift.llm import get_model_tokenizer, get_template
    from swift.tuners import Swift

    device = torch.device(args.device)
    model, processor = get_model_tokenizer(
        args.model,
        model_type="aurora_videollama2_qwen2",
        new_special_tokens=["[SEG]"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float16,
        device_map={"": str(device)},
        download_model=False,
    )
    model = Swift.from_pretrained(model, args.checkpoint)
    model.to(device).eval()
    template = get_template("aurora_videollama2", processor, remove_unused_columns=False)
    template.model = model
    template.set_mode("pt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        path = Path(args.data_dir) / f"refavs_test_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if args.max_samples > 0:
            rows = rows[:args.max_samples]
        output_path = output_dir / f"test_{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                request = copy.deepcopy(row)
                request["messages"] = request["messages"][:1]
                try:
                    encoded = template.encode(request)
                    batch = template.data_collator([encoded])
                    model_inputs = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                        if key in {"input_ids", "attention_mask", "images"}
                    }
                    with torch.inference_mode():
                        input_ids = model_inputs.pop("input_ids")
                        prompt_len = input_ids.shape[1]
                        generated = model.generate(
                            inputs=input_ids,
                            **model_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id,
                        )
                    if generated.ndim == 2 and generated.shape[1] > prompt_len:
                        generated = generated[:, prompt_len:]
                    response = processor.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
                    record = {
                        "split": split,
                        "sample_id": row.get("sample_id"),
                        "response": response,
                        "reference": row["messages"][-1].get("content", "") if len(row["messages"]) > 1 else "",
                    }
                except Exception as exc:
                    record = {"split": split, "sample_id": row.get("sample_id"), "error": repr(exc)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 10 == 0 or index == len(rows):
                    print(f"[{split}] {index}/{len(rows)}", flush=True)
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
