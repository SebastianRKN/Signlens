"""Packaging integration only: synthetic landmarks are NOT recognition evidence."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from signlens.training_job import TrainingJob, file_hash, active_model
from signlens.reporting import write_report

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('exe', type=Path)
args = ap.parse_args()
store = Path(tempfile.mkdtemp(prefix='SignLens training check '))
models, data = store/'models', store/'data'
models.mkdir(); (data/'static').mkdir(parents=True)
shutil.copy2(ROOT/'models/sign_classifier.joblib', models/'sign_classifier.joblib')
rng=np.random.default_rng(77)
def fixture(n):
    return dict(world=rng.normal(0,.04,(n,21,3)),y=np.array(['A','B']*(n//2)),
                group=np.array([f'packaging-{i}' for i in range(n)]),
                session=np.array([f'session-{(i//2)%6}' for i in range(n)]),
                signer=np.array(['dataset']*n))
np.savez_compressed(models/'landmarks.npz', **fixture(48))
np.savez_compressed(data/'static/fixture.npz', **fixture(24))
before=file_hash(models/'sign_classifier.joblib')
sys.executable=str(args.exe.resolve()); sys.frozen=True
os.environ['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
job=TrainingJob(models,data,timeout=300)
started=time.monotonic();job.start()
events=[]
while job.running:
    while not job.events.empty():
        event=job.events.get_nowait();events.append(event)
        print(event,flush=True)
    time.sleep(.2)
while not job.events.empty():events.append(job.events.get_nowait())
success=any(e[0]=='complete' for e in events)
result={'status':'PASS' if success else 'FAIL','scope':__doc__,
        'controller':'source TrainingJob with frozen executable dispatch',
        'worker':'real compiled SignLens.exe --train, no external Python in PATH',
        'duration_seconds':time.monotonic()-started,'events':events,
        'baseline_model_unchanged':before==file_hash(models/'sign_classifier.joblib'),
        'published_model':active_model(models).relative_to(models).as_posix(),
        'evaluation_reports':len(list((store/'reports').rglob('evaluation.json')))}
report=write_report('frozen_training', result,ROOT)
shutil.copy2(job.log_path,report.with_suffix('.log'))
print(report,flush=True)
if not success or not result['baseline_model_unchanged'] or result['evaluation_reports'] != 1:raise SystemExit(1)
