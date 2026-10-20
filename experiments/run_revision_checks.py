"""Supplemental mechanism tests with the original splits and training budget.

No archived result is overwritten. Selection uses validation only. Checkpoints
retain the best validation state and support separate DDE consistency checks.
"""
from pathlib import Path
import argparse
import json
import time
import pandas as pd
import torch
import run_multievent_baselines as base

OUT = base.ROOT / 'output/rigorous_revision'
ORIGINAL = base.RevisedRumorPINN
VARIANTS = ('full', 'no_fatigue', 'no_inhibition', 'no_physics')
CURRENT = 'full'
CAPTURED = None


class RevisionModel(ORIGINAL):
    def __init__(self, name, sequence_length, fixed_tau=None):
        global CAPTURED
        super().__init__(name, sequence_length, fixed_tau)
        self.mechanism_variant = CURRENT
        CAPTURED = self


def main():
    global CURRENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--events', nargs='+', choices=base.EVENTS, default=list(base.EVENTS))
    args = parser.parse_args()
    out = OUT / ('smoke' if args.smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    base.RevisedRumorPINN = RevisionModel
    events = args.events[:1] if args.smoke else args.events
    seeds = [11] if args.smoke else [11, 22, 33]
    total = 0
    for event in events:
        folds = base.load_event_folds(event)
        for fold, frame, split, _, _ in (folds[:1] if args.smoke else folds):
            for variant in VARIANTS:
                CURRENT = variant
                for seed in seeds:
                    stem = f'{event}_fold{fold}_{variant}_seed{seed}'
                    target = out / (stem + '.json')
                    if target.exists():
                        continue
                    cfg = base.Config(physics_weight=0.0 if variant == 'no_physics' else .2)
                    if args.smoke:
                        cfg = base.Config(epochs=2, min_epochs=2, check_every=1,
                                          collocation_points=12,
                                          physics_weight=cfg.physics_weight)
                    print(f'{time.strftime("%H:%M:%S")} START {stem}', flush=True)
                    rows, pred, info = base.train_one(event, fold, frame, split,
                        'TransPINN', seed, cfg, torch.device('cpu'))
                    pred.to_csv(out / (stem + '.csv'), index=False)
                    torch.save(CAPTURED.state_dict(), out / (stem + '.pt'))
                    payload = {'variant': variant, 'config': base.asdict(cfg),
                               'run': info, 'rows': rows}
                    target.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                    total += 1
                    print(f'DONE {stem} validation={info["validation_score"]:.6g}', flush=True)
    print(f'COMPLETE {total} new fits', flush=True)


if __name__ == '__main__':
    main()
