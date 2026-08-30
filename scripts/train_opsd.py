#!/usr/bin/env python3
"""On-Policy Self-Distillation (OPSD) training for AURORA — class-name privilege.

Implementation of the OPSD recipe (arXiv:2601.18734) adapted to Ref-AVS:
the SAME checkpoint acts as teacher and student. The teacher conditions on
privileged information (the GT target class name) while the student sees only
the referring expression; training minimises a clipped per-token forward
KL(teacher || student) over the student's own rollouts, plus the unchanged
AURORA mask supervision (BCE + Dice) on the rollout's [SEG] token.

Per optimizer step:
  1. student rolls out CoT+[SEG] for a train prompt (sampled, no grad)
  2. two prefills on that rollout (no grad for teacher; grad for student):
       teacher: prompt + "(The target object is: {gt_class})"
       student: plain prompt
  3. losses:
       L_kd   = mean over assistant positions of min(KL_fwd, tau)   (pointwise
                vocabulary-entry clipping, per OPSD §3.2 pointwise divergence
                clipping — stylistic tokens otherwise dominate)
       L_mask = 2.0*BCE + 0.5*Dice on the rollout's [SEG] hidden state vs GT
       total  = lambda_kd * L_kd + lambda_mask * L_mask
Rollouts without [SEG] fall back to teacher-forced SFT on the GT label for the
mask branch only (keeps every batch informative; no selection bias on the
text side because L_kd still trains on those rollouts).

This is a standalone Trainer (not an ms-swift plugin) so the dual-context
forward stays explicit. Student = base model + LoRA (updated); teacher = the
frozen initial checkpoint (base + frozen adapter).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]


def _add_import_paths() -> None:
    swift_root = os.environ.get("MS_SWIFT_ROOT", "/mnt/tbo/lvyf/cate-pred-embedding")
    for p in (REPO, Path(swift_root)):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "MODEL_PATH", "/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-3B"))
    p.add_argument("--checkpoint", required=True,
                   help="Stage-1 ms-swift checkpoint (initialises BOTH teacher and student)")
    p.add_argument("--data", default=str(REPO / "outputs/refavs_full.jsonl"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-kd", type=float, default=1.0)
    p.add_argument("--lambda-mask", type=float, default=1.0)
    p.add_argument("--kd-clip", type=float, default=5.0,
                   help="Pointwise vocabulary-entry KL clip tau (per OPSD)")
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--rollout-refresh", type=int, default=8,
                   help="Re-sample rollouts every K optimizer steps (1 = fully on-policy)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


PRIV_TEMPLATE = ("\n(The target object is: {cls})\n"
                 "After understanding this, answer with your own reasoning.")


def make_teacher_prompt(user: str, cls: str) -> str:
    return user + PRIV_TEMPLATE.format(cls=cls)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r.get("sample_id") or ""
        cls = sid.rsplit("_", 1)[0].rsplit("_", 1)[-1] if sid else ""
        rows.append({
            "sample_id": sid, "class": cls,
            "messages": r["messages"],
            "videos": r.get("videos"), "audios": r.get("audios"),
            "sam_frame_paths": r.get("sam_frame_paths"),
            "mask_paths": r.get("mask_paths"),
        })
    return rows


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    _add_import_paths()
    os.environ.setdefault(
        "AURORA_SAM_CHECKPOINT",
        "/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth")

    import swift_plugin.aurora_qwen2_5_omni  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from swift.llm import get_model_tokenizer, get_template
    from swift.tuners import Swift

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load the base model once, attach the Stage-1 adapter, freeze ----
    model, processor = get_model_tokenizer(
        args.model, model_type="aurora_qwen2_5_omni",
        new_special_tokens=["[SEG]"], torch_dtype=torch.float16,
        device_map={"": str(device)}, download_model=False)
    adapter_dir = Path(args.checkpoint)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    talker_key = "base_model.model.talker.model.embed_tokens.weight"
    _tmp = None
    if talker_key not in state:
        import shutil, tempfile
        state[talker_key] = model.talker.model.embed_tokens.weight.detach().cpu().contiguous()
        _tmp = tempfile.TemporaryDirectory(prefix="opsd_adapter_")
        shutil.copy2(adapter_dir / "adapter_config.json", Path(_tmp.name) / "adapter_config.json")
        save_file(state, Path(_tmp.name) / "adapter_model.safetensors", metadata={"format": "pt"})
        model = Swift.from_pretrained(model, _tmp.name)
    else:
        model = Swift.from_pretrained(model, str(adapter_dir))
    model.to(device).eval()

    template = get_template("aurora_qwen2_5_omni", processor, remove_unused_columns=False)
    template.model = model
    tokenizer = processor.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")

    # ---- teacher = frozen copy of this checkpoint --------------------
    # Reuse the same PEFT weights: since only the LoRA/modules_to_save params
    # will be updated for the student, snapshot them now and restore for the
    # teacher forward each step. Cheaper than a second full model copy.
    trainable_names = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_names.add(n)
    teacher_state = {n: p.detach().clone() for n, p in model.named_parameters()
                     if n in trainable_names}

    def switch_to_teacher():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in teacher_state:
                    p.copy_(teacher_state[n])

    def switch_to_student():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in student_state:
                    p.copy_(student_state[n])

    # optimiser over trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    rows = load_dataset(Path(args.data))
    print(f"dataset: {len(rows)} rows | trainable params: "
          f"{sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    # ---- helpers -----------------------------------------------------
    sys.path.insert(0, str(REPO / "scripts"))
    from test_qwen2_5_omni_refavs import _sam_scores

    def _to_dev(x):
        return x.to(device) if torch.is_tensor(x) else x

    def encode_row(user, assistant=None):
        msgs = [{"role": "user", "content": user}]
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
        mode = "pt" if assistant is None else "train"
        template.set_mode(mode)
        try:
            enc = template.encode({"messages": msgs})
        finally:
            template.set_mode("pt")
        return _to_dev(template.data_collator([enc]))

    def rollout_for(row):
        batch = encode_row(row["messages"][0]["content"])
        keep = {k: v for k, v in batch.items()
                if k in {"input_ids", "attention_mask"} or "video" in k or "audio" in k or "pixel" in k or "grid" in k}
        with torch.inference_mode():
            gen = model.generate(**keep, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.rollout_temperature > 0,
                                 temperature=max(args.rollout_temperature, 1e-4),
                                 top_p=1.0, pad_token_id=tokenizer.eos_token_id)
        new = gen[0, keep["input_ids"].shape[1]:]
        return tokenizer.decode(new.tolist(), skip_special_tokens=False)

    def kd_loss(t_out, s_out):
        """Clipped pointwise forward-KL over assistant-labelled positions."""
        t_logits, s_logits = t_out.logits.float(), s_out.logits.float()
        labels = s_out.labels                                   # [1, T], -100 padding
        mask = labels.ne(-100)
        if mask.sum() == 0:
            return None
        t_lp = torch.log_softmax(t_logits, dim=-1)
        s_lp = torch.log_softmax(s_logits, dim=-1)
        # pointwise vocabulary-entry contributions, clipped (OPSD §3.2)
        p_t = t_lp.exp()
        contrib = p_t * (t_lp - s_lp)
        contrib = torch.clip(contrib, min=None, max=args.kd_clip)
        kl = contrib.sum(-1)                                    # [1, T]
        kl = kl * mask
        return kl.sum() / mask.sum()

    log_path = out_dir / "opsd_log.jsonl"
    rollout_cache = {"batch": [], "step": -10**9}

    def sample_batch(bsz=1):
        idxs = torch.randint(0, len(rows), (bsz,)).tolist()
        return [rows[i] for i in idxs]

    started = time.time()
    student_state = None
    for step in range(1, args.max_steps + 1):
        # 1) rollouts (teacher weights are irrelevant here; use current student)
        if step - rollout_cache["step"] >= args.rollout_refresh or not rollout_cache["batch"]:
            batch_rows = sample_batch()
            rollouts = []
            for row in batch_rows:
                try:
                    rollouts.append(rollout_for(row))
                except Exception as exc:
                    rollouts.append(None)
            rollout_cache = {"batch": batch_rows, "rollouts": rollouts, "step": step}
        batch_rows = rollout_cache["batch"]
        rollouts = rollout_cache["rollouts"]

        student_state = {n: p.detach().clone() for n, p in model.named_parameters()
                         if n in trainable_names}

        step_kd, step_mask = 0.0, 0.0
        n_kd = 0
        optim.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            row = batch_rows[micro % len(batch_rows)]
            rollout = rollouts[micro % len(rollouts)]
            if rollout is None:
                continue
            try:
                user = row["messages"][0]["content"]
                # ---- teacher prefill (frozen weights, no grad) ----
                switch_to_teacher()
                with torch.no_grad():
                    t_out = model(**encode_row(make_teacher_prompt(user, row["class"]),
                                               rollout))
                # ---- student prefill (grad) + losses ----
                switch_to_student()
                s_out = model(**encode_row(user, rollout))
                l_kd = kd_loss(t_out, s_out)
                if l_kd is not None:
                    (args.lambda_kd * l_kd / args.grad_accum).backward()
                    step_kd += float(l_kd); n_kd += 1

                # ---- mask supervision on the rollout's [SEG] ----
                if "[SEG]" in rollout:
                    seg_req = {
                        "messages": [{"role": "user", "content": user},
                                     {"role": "assistant", "content": rollout}],
                        "videos": row["videos"], "audios": row["audios"],
                        "sam_frame_paths": row["sam_frame_paths"],
                        "mask_paths": row["mask_paths"],
                        "sample_id": row["sample_id"],
                    }
                    base = model
                    while hasattr(base, "get_base_model"):
                        nxt = base.get_base_model()
                        if nxt is base:
                            break
                        base = nxt
                    # _sam_scores runs model(**) with no_grad internally; the
                    # mask branch shares the student weights, so recompute the
                    # supervised forward with grad on the same batch instead.
                    template.set_mode("train")
                    try:
                        enc = template.encode(copy.deepcopy(seg_req))
                        enc.pop("_extra_kwargs", None)
                        m_batch = _to_dev(template.data_collator([enc]))
                    finally:
                        template.set_mode("pt")
                    template.register_post_encode_hook([model])
                    out = model(**m_batch)
                    l_mask = out.loss  # AURORA joint loss (CE + BCE + Dice)
                    (args.lambda_mask * l_mask / args.grad_accum).backward()
                    step_mask += float(l_mask)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        optim.step()
        if step % args.log_every == 0 or step == args.max_steps:
            rec = {"step": step, "kd": round(step_kd / max(n_kd, 1), 4),
                   "mask": round(step_mask / max(args.grad_accum, 1), 4),
                   "elapsed": int(time.time() - started)}
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as h:
                h.write(json.dumps(rec) + "\n")
        if step % args.save_steps == 0 or step == args.max_steps:
            sd = {n: p.detach().cpu() for n, p in model.named_parameters()
                  if n in trainable_names}
            torch.save(sd, out_dir / f"opsd_step{step}.pt")
            print(f"saved {out_dir / f'opsd_step{step}.pt'}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
