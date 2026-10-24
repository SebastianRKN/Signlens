from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH)
datas = []
for folder in ('models', 'data'):
    for path in (root / folder).rglob('*'):
        if path.is_file() and path.suffix in ('.task', '.joblib', '.npz', '.zip'):
            datas.append((str(path), str(Path('seed') / folder / path.relative_to(root / folder).parent)))
datas += [(str(root / 'DATASET_LICENSE.txt'), 'seed')]
datas += collect_data_files('mediapipe')
for path in (root / 'THIRD_PARTY_LICENSES').rglob('*'):
    if path.is_file(): datas.append((str(path), str(Path('licenses') / path.relative_to(root / 'THIRD_PARTY_LICENSES').parent)))
for package in ('numpy','mediapipe','opencv-python','scikit-learn','joblib'):
    datas += copy_metadata(package)
hidden = collect_submodules('sklearn', filter=lambda name: '.tests' not in name and not name.endswith('conftest'))
hidden += ['signlens.train','signlens.dataset','signlens.validation','signlens.diagnostics','matplotlib.backends.backend_agg']
a = Analysis([str(root / 'run_local.py')], pathex=[str(root / 'src')], binaries=collect_dynamic_libs('mediapipe'), datas=datas, hiddenimports=hidden, excludes=['tkinter','IPython','pytest'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='SignLens', debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='SignLens')
