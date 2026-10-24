"""Exercise the real Windows binary from another cwd with external Python off PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from signlens.reporting import report_path, write_report
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('exe',type=Path)
args=ap.parse_args()
exe=args.exe.resolve()
binary=exe.read_bytes();offset=struct.unpack_from('<I',binary,0x3c)[0]
assert binary[:2]==b'MZ' and binary[offset:offset+4]==b'PE\0\0'
assert struct.unpack_from('<H',binary,offset+4)[0]==0x8664
temp=Path(tempfile.mkdtemp(prefix='SignLens verification '))
store=temp/'datos con espacios'
env=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(key,None)
env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
env['SIGNLENS_HOME']=str(store)
results=[]
def run(label,argv,environment=None,expected_code=0,expected_message=None):
    log=report_path(label,'.log',ROOT)
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run([exe,*map(str,argv)],cwd=os.environ['SystemRoot'],env=environment or env,
                         stdout=f,stderr=subprocess.STDOUT,timeout=180)
    results.append({'test':label,'exit_code':p.returncode,'seconds':time.monotonic()-start,'log':log.relative_to(ROOT).as_posix()})
    print(label,p.returncode,flush=True)
    if p.returncode != expected_code:raise RuntimeError(f'{label} failed; inspect {log}')
    if expected_message and expected_message not in log.read_text(encoding='utf-8'):
        raise RuntimeError(f'{label}: expected explanation missing')
run('release_self_check',['--self-check'])
run('release_native_tracker',['--tracker-check'])
pointer=store/'models/active_model.json'
for name in ('sign_classifier_all.joblib','sign_classifier_previous.joblib'):
    pointer.write_text(json.dumps({'path':name,'sha256':hashlib.sha256((store/'models'/name).read_bytes()).hexdigest()}))
    run('release_'+name.split('.')[0],['--self-check'])
pointer.unlink()
run('release_empty_evaluation',['--evaluate','--models',store/'models','--data',store/'data'],expected_code=1,expected_message='No hay tomas reservadas')
# Synthetic fixture validates execution/report writing only, never recognition quality.
import numpy as np
reserved=store/'data/validation';reserved.mkdir(parents=True)
np.savez_compressed(reserved/'PACKAGING_ONLY.npz',world=np.random.default_rng(991).normal(0,.04,(16,21,3)),
                    y=np.array(['A']*16),session=np.array(['packaging-unseen']*16),signer=np.array(['packaging-only']*16),
                    group=np.array([f'packaging-{i}' for i in range(16)]))
run('release_evaluation',['--evaluate','--models',store/'models','--data',store/'data','--explore'])
run('release_extraction_cli',['--extract','--help'])
run('release_training_cli',['--train','--help'])
blocked=temp/'not a writable directory';blocked.write_text('This file makes the configured directory unavailable.')
fallback=env.copy();fallback['SIGNLENS_HOME']=str(blocked);fallback['LOCALAPPDATA']=str(temp/'local')
run('release_storage_fallback',['--self-check'],fallback)
assert list((temp/'local/SignLens/reports').rglob('installation_*.json'))
run('release_no_camera',['--camera','99','--duration','12'])
health=sorted((store/'reports').rglob('live_health_*.json'))[-1]
state=json.loads(health.read_text());assert state['ended_cleanly'] and state['processed_frames']==0 and state['camera_errors']
result={'status':'PASS','exe_sha256':hashlib.sha256(binary).hexdigest(),'pe_machine':'AMD64 (0x8664)',
        'working_directory':'Windows system directory','executable_in_path_with_spaces':' ' in str(exe),
        'PATH':'Windows/System32 only; PYTHONPATH, PYTHONHOME and VIRTUAL_ENV removed',
        'runtime':'Bundled Python; external Python was not uninstalled',
        'fallback_scope':'Unavailable storage path represented by a file; permission denial additionally unit-tested',
        'evaluation_scope':'Synthetic landmarks exercise pipeline; metrics are NOT recognition evidence',
        'no_camera_health':state,'tests':results,
        'limitations':['No physical test on a second PC','No independent live sign accuracy evaluation',
                       'Extraction CLI help tested; original source images are not in the input ZIP']}
path=write_report('distribution_checks',result,ROOT);print(path,flush=True)
