"""Audit archived independent mechanism fits and publish manuscript results.

Run after run_revision_checks.py. Never selects a fit using test errors.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_multievent_baselines as base
import analyze_multievent_results as analysis

ROOT = base.ROOT
SOURCE = ROOT / 'output/rigorous_revision/full'
OUT = ROOT / 'output/mechanism_ablation'
VARIANTS = {'full': 'TransPINN', 'no_fatigue': 'No fatigue',
            'no_inhibition': 'No antagonistic function'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows, infos, cache, hashes = [], [], {}, {}
    max_prediction_error = 0.
    for event in base.EVENTS:
        for fold, original, split, _, _ in base.load_event_folds(event):
            for variant, seed in itertools.product(VARIANTS, (11, 22, 33)):
                stem = f'{event}_fold{fold}_{variant}_seed{seed}'
                record = json.loads((SOURCE / (stem + '.json')).read_text())
                assert record['config'] == base.asdict(base.Config())
                assert record['variant'] == variant
                frame = pd.read_csv(SOURCE / (stem + '.csv'))
                actual = original[['rumor_activity', 'debunk_activity']].to_numpy(float)
                np.testing.assert_allclose(frame[['actual_rumor', 'actual_debunk']], actual)
                expected = np.where(np.arange(len(frame)) < split[0], 'train',
                    np.where(np.arange(len(frame)) < split[1], 'validation', 'test'))
                np.testing.assert_array_equal(frame['split'], expected)
                model = base.RevisedRumorPINN('TransPINN', 16)
                model.mechanism_variant = variant
                model.load_state_dict(torch.load(SOURCE / (stem + '.pt'), weights_only=True))
                times = (torch.arange(1, len(frame)+1, dtype=torch.float32)/len(frame)).view(-1, 1)
                with torch.no_grad():
                    prediction = model.observation(times).numpy() * np.maximum(actual[:split[0]].sum(0), 1.)
                saved = frame[['predicted_rumor', 'predicted_debunk']].to_numpy()
                np.testing.assert_allclose(prediction, saved, rtol=2e-5, atol=1e-6)
                max_prediction_error = max(max_prediction_error, float(np.max(abs(prediction-saved))))
                for row in record['rows']:
                    c = ('rumor', 'debunk').index(row['channel'])
                    metrics = base.regression_metrics(actual[split[1]:, c], saved[split[1]:, c], actual[:split[0], c])
                    np.testing.assert_allclose([row['nrmse'], row['nmae']], [metrics['nrmse'], metrics['nmae']], rtol=1e-10)
                    rows.append(dict(row, model=VARIANTS[variant]))
                infos.append(dict(record['run'], model=VARIANTS[variant]))
                cache[event, fold, VARIANTS[variant], seed] = frame
                for ext in ('.json', '.csv', '.pt'):
                    p = SOURCE / (stem+ext)
                    hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        print('Verified observations, metrics and all 27 checkpoints:', event, flush=True)
    runs, info = pd.DataFrame(rows), pd.DataFrame(infos)
    assert len(info) == 108 and len(runs) == 216
    selected = info.sort_values(['validation_score', 'seed']).drop_duplicates(['event', 'fold', 'model'])
    chosen = runs.merge(selected[['event', 'fold', 'model', 'seed']], on=['event', 'fold', 'model', 'seed'])
    selection = {(r.event, r.fold, r.model): r.seed for r in selected.itertuples()}
    index = analysis.ExperimentIndex(events=base.EVENTS, models=tuple(VARIANTS.values()), seeds=(11,22,33), folds_by_event={e:(1,2,3) for e in base.EVENTS})
    effects = analysis.paired_validation_selected_bootstrap(cache, selection, index, 5000, 20260917)
    summaries = []
    for model in VARIANTS.values():
        all_seeds = runs[runs.model==model].groupby('seed').nrmse.mean()
        sel = chosen[chosen.model==model]
        summaries.append(dict(model=model, all_mean=all_seeds.mean(), all_sd=all_seeds.std(),
            selected_nrmse=sel.nrmse.mean(), selected_nmae=sel.nmae.mean()))
    summary = pd.DataFrame(summaries)
    for name, table in [('summary',summary), ('paired_bootstrap',effects), ('selection',selected),
                        ('runs',runs), ('event_summary',chosen.groupby(['model','event'])[['nrmse','nmae']].mean().reset_index())]:
        table.to_csv(OUT / (name+'.csv'), index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(fits=108, checkpoint_prediction_max_absolute_error=max_prediction_error,
        bootstrap_replicates=5000, inputs_sha256=hashes), indent=2))
    print(summary.to_string(index=False))
    print(effects[['baseline','baseline_minus_transpinn_nrmse','ci95_low','ci95_high','relative_reduction_percent']].to_string(index=False))


if __name__ == '__main__':
    main()
