"""Select the archived baseline version for a study."""
from pathlib import Path
import argparse
import shutil
p=argparse.ArgumentParser()
p.add_argument('version', choices=['frozen','delay'])
a=p.parse_args()
root=Path(__file__).resolve().parent
shutil.copyfile(root/'versions'/('baseline_'+a.version+'.py'),root/'run_multievent_baselines.py')
print('Selected baseline version: '+a.version)
