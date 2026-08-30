#!/usr/bin/env python3
"""Decoupled reasoning/segmentation metrics for AURORA checkpoints.

Splits performance into:
  TIA  (Target Identification Accuracy)  - keyword matching of the rollout's
        final target statement against the GT class parsed from sample_id.
  mIoU / mDice                            - mask quality (existing SAM scores)
  Cond. mIoU                              - mIoU restricted to TIA-correct
        samples: mask quality when the target was chosen correctly.
  Wrong. mIoU                             - mIoU on TIA-miss samples.

Keyword matching (TIA-v0) inspects only the conclusion zone of the rollout
(last 160 chars) and uses a small synonym table; it is an approximation.
Scope: test_s and test_u only (agreed for the OPSD feasibility study).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYNONYMS = {
    'hair-dryer': ['hair-dryer', 'hair dryer', 'hairdryer', 'dryer', 'blow dryer'],
    'vacuum-cleaner': ['vacuum-cleaner', 'vacuum cleaner', 'vacuum', 'hoover'],
    'emergency-car': ['emergency-car', 'emergency car', 'police car', 'ambulance',
                      'emergency vehicle', 'car'],
    'keyboard': ['keyboard', 'electronic keyboard', 'electric keyboard'],
    'ukulele': ['ukulele', 'ukelele'],
    'handpan': ['handpan', 'hand pan', 'steel drum'],
    'motorcycle': ['motorcycle', 'motorbike', 'bike'],
    'mower': ['mower', 'lawnmower', 'lawn mower'],
    'pipa': ['pipa', 'chinese lute'],
    'guzheng': ['guzheng', 'zither'],
    'woman': ['woman', 'female', 'lady', 'girl'],
    'man': ['man', 'male', 'guy', 'boy'],
}

TAIL_CHARS = 160


def gt_class(sample_id: str) -> str:
    return sample_id.rsplit('_', 1)[0].rsplit('_', 1)[-1]


def variants(cls: str) -> list:
    return SYNONYMS.get(cls, [cls.replace('-', ' '), cls])


def final_target_hit(rollout, cls: str) -> bool:
    if not rollout:
        return False
    tail = rollout[-TAIL_CHARS:].lower()
    return any(v.lower() in tail for v in variants(cls))


def evaluate_run(run_dir: Path, splits=('test_s', 'test_u')) -> dict:
    report = {}
    for sp in splits:
        rows = [json.loads(l) for l in (run_dir / f'{sp}.jsonl').open(encoding='utf-8')]
        n = len(rows)
        hits, cond, wrong, ious = 0, [], [], []
        for r in rows:
            iou = r.get('iou')
            hit = final_target_hit(r.get('rollout'), gt_class(r['sample_id']))
            hits += hit
            if iou is not None:
                ious.append(iou)
                (cond if hit else wrong).append(iou)
        seg = len(ious)
        dices = [r.get('dice') for r in rows if r.get('iou') is not None]
        report[sp] = {
            'n': n,
            'TIA': round(hits / n, 4),
            'seg_rate': round(seg / n, 4),
            'mIoU': round(sum(ious) / max(seg, 1), 4),
            'mDice': round(sum(d for d in dices if d is not None) / max(seg, 1), 4),
            'cond_mIoU': round(sum(cond) / max(len(cond), 1), 4),
            'cond_n': len(cond),
            'wrong_mIoU': round(sum(wrong) / max(len(wrong), 1), 4),
            'wrong_n': len(wrong),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+', help='eval run dirs containing test_{s,u}.jsonl')
    args = parser.parse_args()
    for run in args.run_dirs:
        run_dir = Path(run)
        print(f'== {run_dir}')
        report = evaluate_run(run_dir)
        for sp, m in report.items():
            print(f"  {sp}: TIA={m['TIA']:.3f} seg_rate={m['seg_rate']:.3f} "
                  f"mIoU={m['mIoU']:.3f} | cond-mIoU={m['cond_mIoU']:.3f} (n={m['cond_n']}) "
                  f"| wrong-mIoU={m['wrong_mIoU']:.3f} (n={m['wrong_n']})")
        out = run_dir / 'reasoning_seg_report.json'
        out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
