"""Report execution coverage without selecting or ranking partial results."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1] / 'output/robustness'


def status():
    empirical = []
    for path in (ROOT / 'empirical/runs').glob('*.json'):
        empirical.append(json.loads(path.read_text(encoding='utf-8')))
    stage1 = sum(p['budget'] in (260, 1040) and p['retention'] == 1
                 and p['noise'] == 0 and p['run']['seed'] in (11, 22, 33)
                 for p in empirical)
    stage2 = sum(p['budget'] == 1040 for p in empirical)
    synthetic = list((ROOT / 'synthetic_neural/runs').glob('*.json'))
    dde = list((ROOT / 'synthetic/dde_baseline').glob('*/seed*.json'))
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
              'stage1': {'completed': stage1, 'expected': 360},
              'stage2': {'completed': stage2, 'expected': 5400},
              'empirical_unique': {'completed': len(empirical), 'expected': 5580},
              'synthetic_neural': {'completed': len(synthetic), 'expected': 390},
              'synthetic_dde': {'completed': len(dde), 'expected': 78},
              'neural_total': {'completed': len(empirical) + len(synthetic), 'expected': 5970},
              'note': 'Stage1 and Stage2 share 180 clean, 1040-update fits.'}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'execution_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    status()
