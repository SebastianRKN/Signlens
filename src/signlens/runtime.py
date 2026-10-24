"""UI-independent controller. All UI state changes happen on the camera thread."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time

from signlens.classifier import DEMO_LABELS, DEMO_THRESHOLD, SignClassifier, Stabilizer
from signlens.composer import AutomaticTextComposer
from signlens.gestures import PushToErase
from signlens.panel import ControlPanel
from signlens.recorder import StaticSampleRecorder, new_session, safe_id
from signlens.training_job import TrainingJob, active_model
from signlens.validation import UNKNOWN, dataset_inventory, evaluate_reserved
from signlens.types import HANDS


class Runtime:
    def __init__(self, models: Path, data: Path, signer="me", hand="Right", every=3, model_path=None):
        self.models, self.data = Path(models), Path(data)
        self.model_path = Path(model_path) if model_path else active_model(self.models)
        self.classifier = SignClassifier.load(str(self.model_path)) if self.model_path.exists() else None
        self.stabilizers = {side: Stabilizer() for side in HANDS}
        self.panel = ControlPanel(list(self.classifier.classes) if self.classifier else [], hand)
        self.composer = AutomaticTextComposer()
        self.erase_gesture = PushToErase()
        self.gesture_enabled = False
        self.panel.composer = self.composer
        self.panel.signer = safe_id(signer)
        self.panel.session = new_session(signer)
        self.every = every
        self.capture = None
        self.capture_at = 0.0
        self.job = TrainingJob(self.models, self.data)
        self.evaluation_events = queue.Queue()
        self.evaluation_thread = None
        self._inventory = {}
        self.refresh_counts()
        self.refresh_model_date()

    def refresh_model_date(self):
        self.panel.model_date = self.classifier.meta.get("trained_at", "sin fecha") if self.classifier else "sin modelo"

    def refresh_counts(self):
        self._inventory = {kind: dataset_inventory(self.data / kind) for kind in ("static", "validation")}
        self.update_counts()
        errors = sum(len(x["errors"]) for x in self._inventory.values())
        if errors:
            self.panel.status = f"{errors} archivos no se pudieron leer. Revisa los datos antes de entrenar."

    def update_counts(self):
        kind = "validation" if self.panel.reserved or self.panel.label == UNKNOWN else "static"
        inv = self._inventory[kind]
        self.panel.sample_count = inv["samples"]
        self.panel.label_count = inv["per_label"].get(self.panel.label, 0)

    @property
    def stabilizer(self):
        return self.stabilizers[self.panel.hand]

    def clear_prediction(self, side=None, message="Sin mano visible"):
        self.erase_gesture.reset(side)
        self.composer.reset_input(side=side)
        for key in HANDS if side is None else (side,):
            self.stabilizers[key].reset()
            state = self.panel.results[key]
            state.confirmed = state.provisional = ""
            state.recognition = message

    def toggle_capture(self):
        self.composer.set_active(False)
        p = self.panel
        if self.capture:
            count = self.capture.count
            path = self.capture.toggle()  # On save failure, leave the recorder available for retry.
            p.status = (f"Guardadas {count} muestras de {self.capture.label}. "
                        "Retira la mano antes de la siguiente toma.") if path else "Toma descartada: menos de 5 muestras validas."
            self.capture = None
            p.capturing, p.take_count = False, 0
            self.refresh_counts()
            return path
        if p.training or p.evaluating:
            p.status = "Espera a que termine el proceso activo."
            return None
        kind = "validation" if p.reserved or p.label == UNKNOWN else "static"
        self.capture = StaticSampleRecorder(self.data / kind, p.label, p.signer, self.every, p.session)
        self.capture.toggle()
        self.capture_at = time.monotonic() + 2.0
        p.capturing, p.take_count = True, 0
        p.status = f"Prepara {p.label}. Captura en 2 segundos; C para guardar."
        self.clear_prediction(side=p.hand)

    def handle(self, action):
        p = self.panel
        try:
            if action == "gesture" and p.view == "write" and not p.busy:
                self.gesture_enabled = not self.gesture_enabled
                self.erase_gesture.reset()
                self.composer.reset_input()
                p.status = "Borrado por gesto EXPERIMENTAL activado. G desactiva." if self.gesture_enabled else "Borrado estable: Retroceso. Ctrl+Z deshace."
                return
            if action == "help":
                p.help_open = not p.help_open
                return
            if action in {"view_write", "view_data"}:
                if not p.busy:
                    p.view = "write" if action == "view_write" else "data"
                    p.help_open = False
                    self.composer.set_active(p.view == "write")
                    p.status = "Subtitulos automaticos: ambas manos." if p.view == "write" else "Herramientas de datos. Tab para volver."
                return
            if action in {"space", "backspace", "period", "clear_text", "undo_text", "save_text"}:
                if p.busy or p.view != "write":
                    return
                if action == "save_text":
                    if self.composer.text.strip():
                        folder = self.data.parent / "texts"
                        folder.mkdir(exist_ok=True)
                        dest = folder / f"texto_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000}.txt"
                        dest.write_text(self.composer.text, encoding="utf-8")
                        p.status = f"Texto guardado en texts/{dest.name}"
                    else:
                        p.status = "Aun no hay texto para guardar."
                else:
                    method = {"space": "space", "backspace": "backspace", "period": "period",
                              "clear_text": "clear", "undo_text": "undo"}[action]
                    getattr(self.composer, method)()
                return
            if action == "capture":
                self.toggle_capture()
                return
            if action == "cancel":
                self.job.cancel()
                p.status = "Cancelando; el modelo anterior se conserva."
                return
            if p.busy:
                return
            if action in {"next", "previous"}:
                p.label_index = (p.label_index + (1 if action == "next" else -1)) % len(p.labels)
                self.update_counts()
            elif action == "hand":
                p.hand = "Left" if p.hand == "Right" else "Right"
                p.status = "Cambia solo la mano a grabar. Ambas manos siguen reconociendose."
            elif action == "mode":
                self.composer.set_active(False)
                p.demo = not p.demo
                self.stabilizers = {side: Stabilizer(hold_seconds=.45 if p.demo else 0.0) for side in HANDS}
                self.clear_prediction()
                p.status = ("Demo limitado: 10 letras, umbral minimo 0,80 y espera de 0,45 s."
                            if p.demo else f"Reconocimiento completo: umbral original del modelo ({self.classifier.threshold:.2f})."
                            if self.classifier else "Reconocimiento completo. Falta cargar un modelo.")
            elif action == "reserved":
                p.reserved = not p.reserved
                p.status = ("Tomas reservadas: usa una sesion distinta a la de entrenamiento."
                            if p.reserved else "Estas tomas se usaran para entrenar.")
                self.update_counts()
            elif action == "signer":
                p.editing_signer, p.signer_input = True, p.signer
            elif action == "save_signer":
                p.signer = safe_id(p.signer_input.strip())
                p.session = new_session(p.signer)
                p.editing_signer = False
                p.status = f"Persona: {p.signer}. Nueva sesion iniciada."
            elif action == "session":
                p.session = new_session(p.signer)
                p.status = "Nueva sesion. Usala tras una pausa real, con otra luz o posicion."
            elif action == "train":
                self.composer.set_active(False)
                if self.job.start():
                    p.training = True
                    p.status = "Entrenando con datos guardados. Puedes cancelar con X."
            elif action == "evaluate":
                self.composer.set_active(False)
                p.evaluating = True
                p.status = "Evaluando tomas reservadas..."
                model_path, demo = self.model_path, p.demo

                def worker():
                    try:
                        report = evaluate_reserved(self.models, self.data, model_path, demo)
                        precision, coverage = report["accuracy_on_accepted_takes"], report["coverage"]
                        summary = (f"Acierto: {precision:.0%}. Cobertura: {coverage:.0%}. "
                                   f"Demo: {report['demo_gate']}. Informe en reports.") if precision is not None else "Sin respuestas aceptadas. Informe en reports."
                        self.evaluation_events.put(summary)
                    except Exception as exc:
                        self.evaluation_events.put(str(exc))

                self.evaluation_thread = threading.Thread(target=worker, daemon=True)
                self.evaluation_thread.start()
        except Exception as exc:
            p.status = f"No se pudo completar: {exc}"

    def poll(self):
        while not self.job.events.empty():
            kind, detail = self.job.events.get_nowait()
            if kind in {"progress", "error"}:
                self.panel.status = detail
            elif kind == "complete":
                try:
                    candidate = SignClassifier.load(detail)
                    self.classifier, self.model_path = candidate, Path(detail)
                    self.clear_prediction()
                    self.refresh_model_date()
                    self.panel.status = "Modelo entrenado y cargado. Evalua tomas de otra sesion."
                except Exception as exc:
                    self.panel.status = f"No se pudo recargar el modelo: {exc}"
            elif kind == "finished":
                self.panel.training = False
        while not self.evaluation_events.empty():
            self.panel.status = self.evaluation_events.get_nowait()
            self.panel.evaluating = False

    def process_hand(self, hand, now=None):
        """Compatibility entry point for one-hand capture clients."""
        self.process_hands({self.panel.hand: hand}, now)

    def process_hands(self, hands, now=None, aspect=4/3):
        """One frame, two independent predictions/vote histories. Stale tracks never count."""
        now = time.monotonic() if now is None else now
        p = self.panel
        fresh = {side: hand if hand is not None and not hand.stale else None
                 for side in HANDS for hand in (hands.get(side),)}
        if self.capture:
            if now >= self.capture_at:
                self.capture.add(fresh[p.hand])
                p.take_count = self.capture.count
                p.status = f"Capturando {self.capture.label}: {p.take_count} muestras. C para guardar."
                if self.capture.count >= 100:
                    self.toggle_capture()
            else:
                p.status = f"Prepara {self.capture.label}: {max(1, int(self.capture_at - now) + 1)}..."
        if self.classifier is None:
            self.clear_prediction(message="Sin modelo cargado")
            return
        for side in HANDS:
            hand, state = fresh[side], p.results[side]
            raw = self.classifier.predict(hand.world, threshold=max(DEMO_THRESHOLD, self.classifier.threshold) if p.demo else None,
                                          allowed=DEMO_LABELS if p.demo else None) if hand else None
            out = self.stabilizers[side].update(raw, now)
            state.provisional = raw.label if raw else ""
            state.confirmed = out.label if out else ""
            state.recognition = ("Sin mano visible" if hand is None else "Letra confirmada" if out else
                                 "Mantente estable..." if raw else "Sin reconocimiento suficiente")
            if self.capture and side == p.hand and (p.reserved or p.label == UNKNOWN):
                state.provisional = state.confirmed = ""
                state.recognition = "Toma reservada: resultado oculto"
        if p.view == "write" and not p.busy:
            erase, blocked = self.erase_gesture.update(fresh, now, aspect) if self.gesture_enabled else (False, set())
            if erase:
                self.composer.backspace()
                self.composer.reset_input()
                p.status = "Gesto de borrar. Ctrl+Z para deshacer."
            self.composer.update_hands({side: p.results[side].confirmed if side not in blocked else None for side in HANDS}, now,
                                       present={side: fresh[side] is not None for side in HANDS})
        else:
            self.erase_gesture.reset()

    def close(self):
        if self.capture:
            try:
                self.toggle_capture()
            except Exception as exc:
                # Persist the buffered data to a recovery location if the regular save failed.
                recovery = self.data.parent / "reports" / "capture_recovery"
                self.capture.root = recovery
                self.capture.toggle()
                print(f"[signlens] Captura recuperada en {recovery}: {exc}", flush=True)
        self.job.close()
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
