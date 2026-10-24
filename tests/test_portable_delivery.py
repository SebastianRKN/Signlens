import json
import os
from pathlib import Path
import pytest
from signlens import paths, reporting
from signlens.config import Config
from signlens.capture import _cache_path
from signlens.models import ensure_model
from signlens.training_job import publish_model
from test_demo_safety import fixture_model


def test_reports_are_dated_and_unique(tmp_path):
    a = reporting.write_report('check', {'ok': True}, tmp_path)
    b = reporting.write_report('check', {'ok': True}, tmp_path)
    assert a != b and a.parent == b.parent
    assert a.parent.parent == tmp_path / 'reports'
    assert a.parent.name == reporting.now().date().isoformat()


def test_storage_seed_keeps_user_changes_and_omits_camera(tmp_path, monkeypatch):
    seed, dest = tmp_path / 'seed', tmp_path / 'portable app'
    (seed / 'models').mkdir(parents=True)
    (seed / 'models/model.joblib').write_bytes(b'seed')
    (seed / 'models/camera_0.json').write_text('{}')
    monkeypatch.setattr(paths, 'resource_root', lambda: seed)
    monkeypatch.setenv('SIGNLENS_HOME', str(dest))
    assert paths.prepare_storage() == dest
    (dest / 'models/model.joblib').write_bytes(b'user model')
    paths.prepare_storage()
    assert (dest / 'models/model.joblib').read_bytes() == b'user model'
    assert not (dest / 'models/camera_0.json').exists()


def test_read_only_root_falls_back(tmp_path, monkeypatch):
    seed = tmp_path / 'seed'; seed.mkdir()
    refused = tmp_path / 'read only'
    actual = paths.writable
    monkeypatch.setattr(paths, 'resource_root', lambda: seed)
    monkeypatch.setattr(paths, 'writable', lambda p: False if p == refused else actual(p))
    monkeypatch.setenv('SIGNLENS_HOME', str(refused))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    assert paths.prepare_storage() == tmp_path / 'local/SignLens'


def test_camera_cache_is_not_a_model(tmp_path):
    assert _cache_path(Config(models_dir=tmp_path/'models')) == tmp_path/'config/camera_0.json'


def test_missing_model_never_downloads(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlretrieve', lambda *a: pytest.fail('Unexpected network'))
    with pytest.raises(FileNotFoundError, match='modelo local'):
        ensure_model('hand', tmp_path)


def test_publication_uses_relocated_report(tmp_path):
    models = tmp_path / 'models'
    candidate = fixture_model(models, 'runs/new')
    report = reporting.report_path('evaluation', root=tmp_path)
    (candidate/'evaluation.json').replace(report)
    (candidate/'report_manifest.json').write_text(json.dumps({'evaluation': os.path.relpath(report, candidate)}))
    assert publish_model(models, candidate).is_file()


def test_publication_rejects_external_report(tmp_path):
    models = tmp_path / 'models'
    candidate = fixture_model(models, 'runs/new')
    (candidate/'report_manifest.json').write_text(json.dumps({'evaluation': 'evaluation.json'}))
    with pytest.raises(ValueError, match='reports'):
        publish_model(models, candidate)
