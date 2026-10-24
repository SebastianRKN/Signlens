"""Cancellable training in an isolated directory; publish a complete version atomically."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from signlens.classifier import SignClassifier
from signlens.reporting import report_path, report_run


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_model(models: Path) -> Path:
    pointer = models / "active_model.json"
    if not pointer.exists():
        return models / "sign_classifier.joblib"
    data = json.loads(pointer.read_text(encoding="utf-8"))
    path = (models / data["path"]).resolve()
    if not path.is_relative_to(models.resolve()) or not path.is_file():
        raise ValueError("La referencia del modelo activo es invalida.")
    if file_hash(path) != data["sha256"]:
        raise ValueError("El modelo activo cambio o esta incompleto.")
    return path


def publish_model(models: Path, candidate: Path) -> Path:
    """The old version remains usable until the one-file pointer swap succeeds."""
    candidate = candidate.resolve()
    if not candidate.is_relative_to(models.resolve()):
        raise ValueError("El candidato debe estar dentro de la carpeta de modelos.")
    model = candidate / "sign_classifier.joblib"
    manifest_path = candidate / "report_manifest.json"
    report_file = candidate / "evaluation.json"
    if manifest_path.exists():
        report_file = (candidate / json.loads(manifest_path.read_text(encoding="utf-8"))["evaluation"]).resolve()
        if not report_file.is_relative_to(models.parent.resolve() / "reports"):
            raise ValueError("El informe debe estar dentro de reports.")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    classifier = SignClassifier.load(str(model))
    if not classifier.classes or report["trained_at"] != classifier.meta["trained_at"]:
        raise ValueError("El modelo y su informe no corresponden a la misma version.")
    with np.load(candidate / "training_data" / "base.npz", allow_pickle=True) as data:
        example = data["world"][0]
    classifier.predict(example)  # Execute real inference before activation.
    pointer = models / "active_model.json"
    payload = {"path": model.relative_to(models.resolve()).as_posix(), "sha256": file_hash(model),
               "trained_at": report["trained_at"]}
    if pointer.exists():
        payload["previous"] = json.loads(pointer.read_text(encoding="utf-8"))["path"]
    else:
        payload["previous"] = "sign_classifier.joblib"
    pending = models / f".active-{uuid.uuid4().hex}.json"
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, pointer)
    return model


class TrainingJob:
    def __init__(self, models: Path, data: Path, timeout: float = 3600):
        self.models, self.data, self.timeout = Path(models), Path(data), timeout
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self.log_path = report_path("training", ".log", self.models.parent)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return False
        self.log_path = report_path("training", ".log", self.models.parent)
        self._cancel.clear()
        self.thread = threading.Thread(target=self._run, name="signlens-training", daemon=True)
        self.thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.cancel()
        if self.thread:
            self.thread.join(timeout=8)

    def _run(self) -> None:
        lock = self.models / ".training.lock"
        lock_owned = False
        try:
            self.models.mkdir(parents=True, exist_ok=True)
            with lock.open("x", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            lock_owned = True
            run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self.run_dir = self.models / "runs" / run_id
            snapshot = self.run_dir / "training_data"
            snapshot.mkdir(parents=True)
            self.events.put(("progress", "Preparando copia de los datos de entrenamiento..."))
            shutil.copy2(self.models / "landmarks.npz", snapshot / "base.npz")
            captures = sorted((self.data / "static").rglob("*.npz"))
            if not captures:
                raise ValueError("Primero guarda capturas para entrenar.")
            reports = report_run("training", self.models.parent)
            manifest = []
            for index, source in enumerate(captures):
                target = snapshot / "captures" / f"{index:06d}.npz"
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source, target)
                manifest.append({"source": source.relative_to(self.data).as_posix(), "sha256": file_hash(target)})
            (reports / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            prefix = [sys.executable, "--train"] if getattr(sys, "frozen", False) else [sys.executable, "-u", "-m", "signlens.train"]
            command = prefix + [str(snapshot / "base.npz"), "--reports-dir", str(reports),
                       "--extra", str(snapshot / "captures"), "--out", str(self.run_dir),
                       "--models", "ensemble", "--target-precision", "0.8"]
            env = os.environ.copy()
            env["SIGNLENS_HOME"] = str(self.models.parent.resolve())
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            env["PYTHONIOENCODING"] = "utf-8"
            env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
            with self.log_path.open("w", encoding="utf-8", buffering=1) as log:
                log.write(f"Training run {run_id}\n")
                if self._cancel.is_set():
                    raise InterruptedError("Entrenamiento cancelado.")
                self.process = subprocess.Popen(
                    command, cwd=self.models.parent, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

                def read_output():
                    assert self.process and self.process.stdout
                    for line in self.process.stdout:
                        log.write(line)
                        if line.strip():
                            self.events.put(("progress", line.strip()))

                reader = threading.Thread(target=read_output, daemon=True)
                reader.start()
                started = time.monotonic()
                while self.process.poll() is None:
                    if self._cancel.wait(0.15) or time.monotonic() - started > self.timeout:
                        self.process.kill()
                        self.process.wait(timeout=5)
                        reader.join(timeout=3)
                        if self._cancel.is_set():
                            raise InterruptedError("Entrenamiento cancelado; modelo anterior conservado.")
                        raise TimeoutError("Tiempo agotado; modelo anterior conservado.")
                reader.join(timeout=5)
                if self.process.returncode != 0:
                    raise RuntimeError(f"Entrenamiento fallo ({self.process.returncode}). Consulta {self.log_path}.")
            if self._cancel.is_set():
                raise InterruptedError("Entrenamiento cancelado; modelo anterior conservado.")
            model = publish_model(self.models, self.run_dir)
            self.events.put(("complete", str(model)))
        except Exception as exc:
            message = "Ya hay otro entrenamiento. Revisa .training.lock." if isinstance(exc, FileExistsError) else str(exc)
            self.events.put(("error", message))
            if lock_owned:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n{type(exc).__name__}: {message}\n")
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            if lock_owned:
                lock.unlink(missing_ok=True)
            self.events.put(("finished", ""))
