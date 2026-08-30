#!/usr/bin/env python3
"""Layer-1 feasibility probe for OPSD on AURORA (Qwen2.5-Omni-3B).

Validates, WITHOUT any training, whether privileged information produces a
useful teacher signal on the student's own rollouts:

  Probe A (reasoning privilege): for each sampled train case, the student
    model rolls out freely; the SAME model then prefills the rollout under
    three conditions —
      base     : no privilege            (== student, KL ~ 0, sanity check)
      +class   : GT target class name in the prompt
      +cot     : full GT CoT label in the prompt
    and we measure per-token forward KL(teacher || student) and, at the token
    level, which tokens carry the divergence.

  Probe B (answer-leak negative control): the same as "+class" but the class
    name is deliberately WRONG (another class sampled from the dataset). If
    the model blindly parrots the privileged hint, its KL pattern will look
    identical to the correct-hint case; a model that truly rationalizes
    should distribute divergence differently.

Outputs one JSONL per rank plus a merged summary. Metrics reported:
  kl_mean / kl_p90            — per-token forward KL statistics
  kl_on_class_tokens          — share of total KL mass on the GT class tokens
  seg_embedding_cos / l2      — [SEG] prompt-embedding alignment (teacher vs
                                 student), probe for feature distillation
Usage:
  torchrun --nproc_per_node=N scripts/probe_opsd_privilege.py \
      --checkpoint outputs/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262 \
      --num-samples 200 --output-dir outputs/eval/opsd_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_import_paths() -> None:
    root = _repo_root()
    swift_root = os.environ.get("MS_SWIFT_ROOT", "/mnt/tbo/lvyf/cate-pred-embedding")
    for path in (root, Path(swift_root)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))




def load_rows(data_path: Path, num_samples: int, rank: int, world_size: int) -> List[Dict[str, Any]]:
    rows = []
    for line in data_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        user = r["messages"][0]["content"]
        gt = r["messages"][-1]["content"]
        sid = r.get("sample_id") or ""
        cls = sid.rsplit("_", 1)[0].rsplit("_", 1)[-1] if sid else ""
        rows.append({
            "sample_id": sid, "user": user, "gt": gt, "class": cls,
            # kept for SAM scoring against GT masks
            "videos": r.get("videos"), "audios": r.get("audios"),
            "sam_frame_paths": r.get("sam_frame_paths"),
            "mask_paths": r.get("mask_paths"),
        })
    if num_samples > 0:
        rows = rows[:num_samples]
    return rows[rank::world_size]


def build_teacher_prompt(user: str, privilege: str, privilege_kind: str) -> str:
    """Insert privileged text into the user turn (before the instruction)."""
    marker = "The reference is:"
    if marker not in user:
        return user
    head, tail = user.split(marker, 1)
    if privilege_kind == "class":
        insert = f"{marker}{tail}\n(The target object is: {privilege})\nAfter understanding this, answer with your own reasoning."
    else:  # cot
        insert = (f"{marker}{tail}\n(Here is a reference solution: {privilege})\n"
                  "After understanding it, solve with your own approach.")
    return head + insert


def forward_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Per-position forward KL(teacher || student), fp32, both [B, T, V]."""
    t = torch.log_softmax(teacher_logits.float(), dim=-1)
    s = torch.log_softmax(student_logits.float(), dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)          # [B, T]
    return kl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get(
        "MODEL_PATH", "/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-3B"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=str(_repo_root() / "outputs/refavs_full.jsonl"))
    parser.add_argument("--output-dir", default=str(_repo_root() / "outputs/eval/opsd_probe"))
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    parser.add_argument("--torch-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--conditions", nargs="+",
                        default=["base", "class", "cot", "wrong_class"])
    parser.add_argument("--gen-teacher", action=argparse.BooleanOptionalAction, default=True,
                        help="Also free-generate under each privileged condition and score its mask")
    parser.add_argument("--run-sam", action=argparse.BooleanOptionalAction, default=True,
                        help="Score [SEG] masks (student + privileged generations) against GT")
    parser.add_argument("--dist-backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    _add_import_paths()
    os.environ.setdefault(
        "AURORA_SAM_CHECKPOINT",
        "/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    import swift_plugin.aurora_qwen2_5_omni  # noqa: F401  (registers model/template)
    from swift.llm import get_model_tokenizer, get_template
    from swift.tuners import Swift

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.torch_dtype]
    model, processor = get_model_tokenizer(
        args.model, model_type="aurora_qwen2_5_omni",
        new_special_tokens=["[SEG]"], torch_dtype=dtype,
        device_map={"": str(device)}, download_model=False)
    # Older PEFT builds require the untouched talker embedding key that SFT
    # never saved; borrow the compat shim from test_qwen2_5_omni_refavs.py.
    from safetensors.torch import load_file, save_file
    adapter_dir = Path(args.checkpoint)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    talker_key = "base_model.model.talker.model.embed_tokens.weight"
    _tmp = None
    if talker_key not in state:
        import shutil, tempfile
        state[talker_key] = model.talker.model.embed_tokens.weight.detach().cpu().contiguous()
        _tmp = tempfile.TemporaryDirectory(prefix="aurora_probe_adapter_")
        shutil.copy2(adapter_dir / "adapter_config.json", Path(_tmp.name) / "adapter_config.json")
        save_file(state, Path(_tmp.name) / "adapter_model.safetensors", metadata={"format": "pt"})
        model = Swift.from_pretrained(model, _tmp.name)
    else:
        model = Swift.from_pretrained(model, str(adapter_dir))
    model.to(device).eval()
    template = get_template("aurora_qwen2_5_omni", processor, remove_unused_columns=False)
    template.model = model
    template.set_mode("pt")
    tokenizer = processor.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")

    rows = load_rows(Path(args.data), args.num_samples, rank, world_size)
    all_classes = sorted({r["class"] for r in rows if r["class"]})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (f"probe.rank{rank}.jsonl" if world_size > 1 else "probe.jsonl")

    def encode_generate(user: str):
        req = {"messages": [{"role": "user", "content": user}]}
        enc = template.encode(req)
        batch = template.data_collator([enc])
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                if k in {"input_ids", "attention_mask"} | (
                    {"pixel_values_videos", "video_grid_thw", "input_audio", "audio_feature_lengths"}
                    & set(batch.keys()))}

    def prefill_logits(user: str, assistant_text: str):
        """Prefill (user, assistant) and return model output (logits/labels)."""
        req = {"messages": [{"role": "user", "content": user},
                            {"role": "assistant", "content": assistant_text}]}
        template.set_mode("train")
        try:
            enc = template.encode(req)
        finally:
            template.set_mode("pt")
        batch = template.data_collator([enc])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        return out

    import random
    rng = random.Random(1234 + rank)

    # ---- optional SAM scoring: reuse the tested eval script's helper ----
    sys.path.insert(0, str(_repo_root() / "scripts"))
    from test_qwen2_5_omni_refavs import _sam_scores as _qwen_sam_scores

    def sam_scores(user_prompt, rollout_text, ds_row):
        req = {
            "messages": [{"role": "user", "content": user_prompt}],
            "videos": ds_row["videos"], "audios": ds_row["audios"],
            "sam_frame_paths": ds_row["sam_frame_paths"],
            "mask_paths": ds_row["mask_paths"],
            "sample_id": ds_row["sample_id"],
        }
        return _qwen_sam_scores(model, template, req, rollout_text, device=device)

    started = time.time()
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, 1):
            record = {"sample_id": row["sample_id"], "class": row["class"]}
            try:
                # 1) student rollout
                inputs = encode_generate(row["user"])
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=1.0, top_p=1.0,
                        pad_token_id=tokenizer.eos_token_id)
                new_tokens = gen[0, inputs["input_ids"].shape[1]:]
                rollout = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)
                record["rollout_tail"] = rollout[-160:]
                record["rollout_has_seg"] = "[SEG]" in rollout

                # student prefill on its own rollout (labels path) — reference logits
                student_out = prefill_logits(row["user"], rollout)
                s_logits = student_out.logits[0]          # [T, V]

                # student mask score (baseline segmentation ability)
                if args.run_sam and "[SEG]" in rollout:
                    try:
                        sc = sam_scores(row["user"], rollout, row)
                        record["student_iou"] = sc["iou"]
                        record["student_dice"] = sc["dice"]
                    except Exception as exc:
                        record["student_sam_error"] = repr(exc)[:200]

                # 2) teacher conditions: KL + privileged generation + mask
                for cond in args.conditions:
                    if cond == "base":
                        t_user = row["user"]
                    elif cond == "class":
                        t_user = build_teacher_prompt(row["user"], row["class"], "class")
                    elif cond == "wrong_class":
                        wrong = rng.choice([c for c in all_classes if c != row["class"]] or [row["class"]])
                        t_user = build_teacher_prompt(row["user"], wrong, "class")
                    elif cond == "cot":
                        t_user = build_teacher_prompt(row["user"], row["gt"], "cot")
                    else:
                        continue
                    t_out = prefill_logits(t_user, rollout)
                    t_logits = t_out.logits[0]
                    # align lengths (take min over sequence dim)
                    T = min(t_logits.shape[0], s_logits.shape[0])
                    kl = forward_kl(t_logits[:T][None], s_logits[:T][None])[0]  # [T]
                    record[f"kl_{cond}_mean"] = float(kl.mean())
                    record[f"kl_{cond}_p90"] = float(torch.quantile(kl, 0.9)) if kl.numel() else 0.0
                    # KL mass on the GT class tokens inside the rollout
                    cls_ids = set()
                    for v in [row["class"], row["class"].replace("-", " ")]:
                        toks = tokenizer(v, add_special_tokens=False)["input_ids"]
                        cls_ids.update(toks)
                    if cls_ids:
                        tok_ids = new_tokens[:T].tolist()
                        mass = sum(kl[i].item() for i, t in enumerate(tok_ids) if t in cls_ids)
                        record[f"kl_{cond}_on_class"] = mass / max(float(kl.sum()), 1e-6)
                    else:
                        record[f"kl_{cond}_on_class"] = None

                    # privileged-condition free generation + mask: the upper
                    # bound this teacher condition could distill
                    if args.gen_teacher and cond != "base":
                        t_inputs = encode_generate(t_user)
                        with torch.inference_mode():
                            t_gen = model.generate(
                                **t_inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
                        t_new = t_gen[0, t_inputs["input_ids"].shape[1]:]
                        t_rollout = tokenizer.decode(t_new.tolist(), skip_special_tokens=False)
                        record[f"teachgen_{cond}_has_seg"] = "[SEG]" in t_rollout
                        record[f"teachgen_{cond}_tail"] = t_rollout[-160:]
                        record[f"teachgen_{cond}_mentions_class"] = (
                            row["class"].lower() in t_rollout[-200:].lower())
                        if args.run_sam and "[SEG]" in t_rollout:
                            try:
                                sc = sam_scores(t_user, t_rollout, row)
                                record[f"teachgen_{cond}_iou"] = sc["iou"]
                                record[f"teachgen_{cond}_dice"] = sc["dice"]
                            except Exception as exc:
                                record[f"teachgen_{cond}_sam_error"] = repr(exc)[:200]
            except Exception as exc:
                record["error"] = repr(exc)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % args.log_every == 0 or idx == len(rows):
                print(f"[rank{rank}] {idx}/{len(rows)} ({time.time()-started:.0f}s)", flush=True)

    # merge on rank 0
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend=args.dist_backend, init_method="env://")
        dist.barrier()
        if rank == 0:
            merged = []
            for r in range(world_size):
                merged += [json.loads(l) for l in (output_dir / f"probe.rank{r}.jsonl").open(encoding="utf-8")]
            with (output_dir / "probe.jsonl").open("w", encoding="utf-8") as h:
                for rec in merged:
                    h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dist.barrier()

    if rank == 0:
        summarize(output_dir / "probe.jsonl", args.conditions)


def summarize(path: Path, conditions: List[str]) -> None:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if "error" not in r]
    print(f"\n===== OPSD privilege probe summary ({len(ok)}/{len(rows)} ok) =====")
    seg_rate = sum(r.get("rollout_has_seg", False) for r in ok) / max(len(ok), 1)
    print(f"student rollout [SEG] rate: {seg_rate:.3f}")
    st_iou = [r["student_iou"] for r in ok if r.get("student_iou") is not None]
    if st_iou:
        print(f"student greedy-free (T=1.0) mask mIoU: {sum(st_iou)/len(st_iou):.4f} (n={len(st_iou)})")
    for cond in conditions:
        means = [r[f"kl_{cond}_mean"] for r in ok if r.get(f"kl_{cond}_mean") is not None]
        p90s = [r[f"kl_{cond}_p90"] for r in ok if r.get(f"kl_{cond}_p90") is not None]
        on_cls = [r[f"kl_{cond}_on_class"] for r in ok if r.get(f"kl_{cond}_on_class") is not None]
        line = ""
        if means:
            line += f"KL mean={sum(means)/len(means):.4f} p90={sum(p90s)/len(p90s):.4f} "
        if on_cls:
            line += f"KL-on-class%={100*sum(on_cls)/len(on_cls):.1f} "
        ti = [r[f"teachgen_{cond}_iou"] for r in ok if r.get(f"teachgen_{cond}_iou") is not None]
        tc = [r.get(f"teachgen_{cond}_mentions_class") for r in ok]
        tg = [r.get(f"teachgen_{cond}_has_seg") for r in ok]
        if ti:
            line += (f"| teachgen: seg%={100*sum(1 for x in tg if x)/max(len(tg),1):.1f} "
                     f"class-mention%={100*sum(1 for x in tc if x)/max(len(tc),1):.1f} "
                     f"mIoU={sum(ti)/len(ti):.4f} (n={len(ti)})")
        if line:
            print(f"{cond:>10}: {line}")
    print("\nReading guide:")
    print(" - 'class'/'cot' KL >> 'base' KL  => privilege changes the distribution")
    print(" - teachgen mIoU(class/cot) > student mIoU => privileged generation is a")
    print("   better rollout; the distillation target is genuinely higher quality")
    print(" - wrong_class teachgen mIoU ≈ class teachgen mIoU => the model parrots")
    print("   the hint (bad); wrong_class << class => it rationalizes (good)")


if __name__ == "__main__":
    main()
