"""Dated, unique report locations shared by source and frozen processes."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import uuid


def now():
    return datetime.now(timezone(timedelta(hours=-5)))


def root_dir():
    return Path(os.environ.get('SIGNLENS_HOME', Path(__file__).resolve().parents[2]))


def report_folder(root=None):
    path = Path(root or root_dir()) / 'reports' / now().date().isoformat()
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(name, suffix='.json', root=None):
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid report name')
    return report_folder(root) / f'{name}_{now():%H%M%S_%f}_{uuid.uuid4().hex[:8]}{suffix}'


def report_run(name, root=None):
    path = report_path(name, '', root)
    path.mkdir()
    return path


def write_report(name, payload, root=None):
    path = report_path(name, root=root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
