#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def sorted_media(directory: Path, suffix: str):
    return sorted(
        (str(path) for path in directory.glob(f"*{suffix}")),
        key=lambda path: int(Path(path).stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reasoning-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-frames", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    table = pd.read_csv(args.metadata)
    table = table[table["split"] == args.split].reset_index(drop=True)
    with open(args.reasoning_json, encoding="utf-8") as handle:
        reasoning = json.load(handle)
    if args.max_samples > 0:
        table = table.iloc[: args.max_samples].reset_index(drop=True)
    if args.split == "train" and len(reasoning) < len(table):
        raise ValueError(f"Reasoning entries ({len(reasoning)}) are fewer than samples ({len(table)})")
    if args.split == "train":
        split_reasoning = reasoning[: len(table)]
    else:
        # Keep validation/test teacher forcing deterministic; train reasoning
        # is indexed only for the train metadata subset.
        split_reasoning = [{"reasoning": "It is [SEG]."} for _ in range(len(table))]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in table.iterrows():
            video_id = str(row["vid"])
            fid = int(row["fid"])
            media_dir = dataset_dir / "media" / video_id
            frames = sorted_media(media_dir / "frames", ".jpg")[: args.num_frames]
            masks = sorted_media(dataset_dir / "gt_mask" / video_id / f"fid_{fid}", ".png")[: args.num_frames]
            if not frames or not masks:
                continue
            while len(frames) < args.num_frames:
                frames.append(frames[-1])
            while len(masks) < args.num_frames:
                masks.append(masks[-1])
            answer = str(split_reasoning[index].get("reasoning", "")).strip()
            if "[SEG]" not in answer:
                answer = f"{answer} It is [SEG].".strip()
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<video><audio>The reference is: "
                            f"{row['exp']} Please segment the corresponding object in the video."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
                "videos": [frames],
                "audios": [str(media_dir / "audio.wav")],
                "sam_frame_paths": frames,
                "mask_paths": masks,
                # Preserve the original REF-AVS referring expression instead
                # of requiring downstream evaluators to parse the prompt.
                "referring": str(row["exp"]),
                "class_name": str(row.get("label", "")),
                "sample_id": str(row["uid"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"output": str(output.resolve()), "samples": written}))


if __name__ == "__main__":
    main()
