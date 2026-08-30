#!/usr/bin/env python3
"""Build the OPSD dataset (teacher_prompt column) from the Stage-1 SFT JSONL.

Each row keeps its original ``messages`` (student view: plain referring
expression) and gains a ``teacher_prompt`` column — the full user content with
the privileged information (GT class name) injected. ms-swift 4.x's OPSD path
(``OnPolicySample.build_teacher_view``) replaces the last user message with
``teacher_prompt`` verbatim, so this column MUST carry the complete user turn
including the ``<video><audio>`` media placeholders — not just the privileged
sentence.

Privileged-information design (validated by scripts/probe_opsd_privilege.py on
test_s, 200 cases):
  - class privilege: mIoU 0.358 -> 0.651, negative control (wrong_class) passes
    (0.274, model rationalizes rather than parrots)
  - cot privilege: dead ([SEG] emission rate 3% — the model "consumes" [SEG]
    while reading the reference solution), so it is NOT emitted here

The teacher prompt keeps the original referring expression verbatim and appends
the privilege block, mirroring the probe's ``build_teacher_prompt`` wording so
the training-time privileged distribution matches the measured one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def class_from_sample_id(sample_id: str) -> str:
    """REFAVS sample_id format: ``<video_id>_<class>_<idx>``.

    The video id itself contains underscores, so parse from the right: drop the
    trailing index, then the trailing segment is the class name. Same parsing
    as the probe / eval scripts (test scripts use rsplit('_'))."""
    return sample_id.rsplit('_', 1)[0].rsplit('_', 1)[-1]


def build_teacher_prompt(user_content: str, gt_class: str) -> str:
    """Inject the class-name privilege into a user turn.

    The probe measured this exact wording (class condition): the privilege
    line goes AFTER the referring expression, and the instruction to solve it
    with the model's own reasoning keeps the distribution a "rationalization"
    rather than a copy of the hint.
    """
    privilege = (
        f"\n(The target object is: {gt_class}. "
        "After understanding this, solve with your own approach.)"
    )
    # user_content ends with the referring expression; append, do not rewrite.
    return user_content.rstrip() + privilege


def convert_file(src: str, dst: str, max_samples: int = -1) -> dict:
    n_in = n_out = n_bad = 0
    classes = set()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_in += 1
            if max_samples > 0 and n_out >= max_samples:
                break
            messages = row.get("messages") or []
            user = next((m for m in messages if m.get("role") == "user"), None)
            sample_id = row.get("sample_id") or ""
            if user is None or not sample_id:
                n_bad += 1
                continue
            gt_class = (row.get("class_name") or "").strip() or class_from_sample_id(sample_id)
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9-]*$", gt_class):
                n_bad += 1
                continue
            row["teacher_prompt"] = build_teacher_prompt(user["content"], gt_class)
            classes.add(gt_class)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    return {"input": n_in, "output": n_out, "skipped": n_bad, "classes": len(classes)}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=os.path.join(root, "outputs", "refavs_full.jsonl"))
    parser.add_argument("--dst", default=os.path.join(root, "outputs", "refavs_full_opsd.jsonl"))
    parser.add_argument("--val-src", default=os.path.join(root, "outputs", "refavs_val.jsonl"))
    parser.add_argument("--val-dst", default=os.path.join(root, "outputs", "refavs_val_opsd.jsonl"))
    parser.add_argument("--max-samples", type=int, default=-1, help="cap output rows (smoke tests)")
    args = parser.parse_args()

    for src, dst in [(args.src, args.dst), (args.val_src, args.val_dst)]:
        if not os.path.exists(src):
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        stats = convert_file(src, dst, args.max_samples)
        print(f"[ok] {src} -> {dst}")
        print(f"     rows: {stats['output']}/{stats['input']} (skipped {stats['skipped']}), classes: {stats['classes']}")


if __name__ == "__main__":
    main()
