"""Camera, recognition and collection UI. Run `run_local.py` for the bundled app."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from signlens.reporting import report_path
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from signlens import features
from signlens.capture import Camera, format_probe, probe_all
from signlens.config import Config
from signlens.recorder import SequenceRecorder, StaticSampleRecorder, new_session
from signlens.runtime import Runtime
from signlens.tracker import Tracker
from signlens.viz import draw


def _fresh_hand(result, side: str):
    hand = result.hands.get(side)
    return None if hand is None or hand.stale else hand


def _benchmark(cam, tracker, cfg, seconds):
    start = time.perf_counter()
    latency, processed = [], 0
    while time.perf_counter() - start < seconds:
        frame = cam.read()
        if frame is None:
            continue
        t0 = time.perf_counter()
        features.extract(tracker.process(frame, time.monotonic_ns() // 1_000_000), cfg.min_pose_visibility)
        latency.append(1000 * (time.perf_counter() - t0))
        processed += 1
    print(json.dumps({"seconds": time.perf_counter() - start, "camera_fps": cam.fps,
                      "pipeline_fps": processed / (time.perf_counter() - start), "frames": processed,
                      "latency_p95_ms": float(np.percentile(latency[10:] or latency or [0], 95)),
                      "backend": cam.backend}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description="SignLens: reconocimiento experimental de letras estaticas.")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--pose-model", choices=("lite", "full"), default="lite")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--label", help="Grabar secuencias con ESPACIO")
    ap.add_argument("--collect", help="Grabar una letra con ESPACIO")
    ap.add_argument("--signer", default="me")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--model", help="Modelo especifico; por defecto se resuelve la version activa")
    ap.add_argument("--benchmark", type=float, metavar="SECONDS")
    ap.add_argument("--hand", choices=("Left", "Right"), default="Right", help="Mano a grabar; ambas se reconocen siempre")
    ap.add_argument("--probe-camera", action="store_true")
    ap.add_argument("--reprobe", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--duration", type=float, help="Cerrar tras N segundos; para ensayos controlados")
    ap.add_argument("--reserve", action="store_true", help="Guardar capturas para evaluacion, sin entrenar con ellas")
    args = ap.parse_args()
    cfg = Config(camera_index=args.camera if args.camera is not None else 0, smoothing=not args.no_smoothing, pose_model=args.pose_model,
                 mirror=not args.no_mirror, camera_probe=not args.no_probe)
    if args.probe_camera:
        print(format_probe(probe_all(cfg), cfg.fps))
        return
    if args.benchmark:
        cam = Camera(cfg, reprobe=args.reprobe)
        try:
            tracker = Tracker(cfg)
            try:
                _benchmark(cam, tracker, cfg, args.benchmark)
            finally:
                tracker.close()
        finally:
            cam.release()
        return

    runtime = Runtime(cfg.models_dir, cfg.data_dir, args.signer, args.hand, args.every, args.model)
    p = runtime.panel
    p.reserved = args.reserve
    if os.environ.get('SIGNLENS_STORAGE_FALLBACK') == '1':
        p.status = f"Datos y reportes: {cfg.data_dir.parent}"
    if args.collect or args.label or args.reserve:
        runtime.handle("view_data")
    print(f"[signlens] Model: {runtime.model_path}; mode=COMPLETO; "
          f"threshold={runtime.classifier.threshold if runtime.classifier else 'none'}; "
          f"reserved_destination={p.reserved}; recognition_hands=Left,Right", flush=True)
    if args.collect:
        if args.collect not in p.labels:
            ap.error("--collect debe ser una letra del modelo o NINGUNA.")
        p.label_index = p.labels.index(args.collect)
    runtime.update_counts()
    rec = SequenceRecorder(cfg.data_dir / "sequences", args.label) if args.label else None
    tracker = Tracker(cfg)
    cam = None
    camera_events = queue.Queue()
    opening = False
    closing = threading.Event()
    camera_thread = None
    camera_errors, reopen_count = [], 0

    def reconnect():
        nonlocal cam, opening, camera_thread, reopen_count, last_frame_at
        if opening:
            return
        if runtime.capture:
            runtime.toggle_capture()
        runtime.clear_prediction()
        if cam:
            cam.release()
            cam = None
        opening = True
        reopen_count += 1
        last_frame_at = time.monotonic()
        p.camera_status = "Conectando camara..."

        def open_worker():
            try:
                candidate = None
                failures = []
                for index in ([args.camera] if args.camera is not None else range(5)):
                    if closing.is_set():
                        return
                    try:
                        candidate = Camera(replace(cfg, camera_index=index), reprobe=args.reprobe and reopen_count == 1)
                        break
                    except RuntimeError as error:
                        failures.append(str(error))
                if candidate is None:
                    selection = str(args.camera) if args.camera is not None else "0-4"
                    raise RuntimeError(f"No hay camara disponible (indices {selection}). Revisa permisos o cierra otras apps. " + "; ".join(failures))
                if closing.is_set():
                    candidate.release()
                else:
                    camera_events.put(candidate)
            except Exception as exc:
                camera_events.put(str(exc))
        camera_thread = threading.Thread(target=open_worker, daemon=True)
        camera_thread.start()

    cv2.namedWindow("signlens", cv2.WINDOW_AUTOSIZE)
    start = last_frame_at = last_health = time.monotonic()
    processed, latency, arrivals = 0, deque(maxlen=1000), deque(maxlen=120)
    health_path = report_path("live_health", root=cfg.data_dir.parent)
    last_image = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)

    def save_health(ended=False):
        health_path.parent.mkdir(exist_ok=True)
        state = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "uptime_seconds": time.monotonic() - start,
                 "processed_frames": processed, "camera_fps": cam.fps if cam else 0,
                 "pipeline_fps": (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], .001) if len(arrivals) > 1 else 0,
                 "latency_p95_ms": float(np.percentile(latency, 95)) if latency else None,
                 "camera_errors": camera_errors, "camera_open_attempts": reopen_count,
                 "model": str(runtime.model_path), "model_date": p.model_date,
                 "ended_cleanly": ended, "training": p.training,
                 "note": "Health telemetry measures uptime and processing; it is not sign accuracy."}
        tmp = health_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(health_path)

    reconnect()
    try:
        while True:
            now = time.monotonic()
            runtime.poll()
            if not camera_events.empty():
                value = camera_events.get_nowait()
                opening = False
                if isinstance(value, str):
                    camera_errors.append(value)
                    p.camera_status = "Camara no disponible. R para reintentar."
                    p.status = value
                else:
                    cam, last_frame_at = value, now
                    print(f"[signlens] Camera: {cam.backend} {cam.size}, {cam.info.measured_fps:.1f} FPS", flush=True)
            frame = None
            if cam:
                try:
                    frame = cam.read(timeout=.02)
                    if frame is None and now - last_frame_at > 3:
                        raise RuntimeError("La camara lleva 3 segundos sin entregar imagen.")
                except RuntimeError as exc:
                    camera_errors.append(str(exc))
                    cam.release()
                    cam = None
                    runtime.clear_prediction()
                    last_image = np.zeros_like(last_image)
                    if runtime.capture:
                        runtime.toggle_capture()
                    p.camera_status = "Camara desconectada. Pulsa R para reconectar."
                    runtime.clear_prediction(message="Sin imagen de camara")
            if frame is not None:
                last_frame_at = now
                t0 = time.perf_counter()
                result = tracker.process(frame, time.monotonic_ns() // 1_000_000)
                hands = {side: _fresh_hand(result, side) for side in ("Left", "Right")}
                runtime.process_hands(hands, now, result.aspect)
                if p.capturing and hands[p.hand] is None:
                    other = "Left" if p.hand == "Right" else "Right"
                    if _fresh_hand(result, other) is not None:
                        p.status = "Para grabar la otra mano: deten con C, cambia con H e inicia con C."
                if rec:
                    rec.add(features.extract(result, cfg.min_pose_visibility))
                processed += 1
                arrivals.append(now)
                fps = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], .001) if len(arrivals) > 1 else 0
                last_image = draw(frame.copy(), result, fps, p.capturing or bool(rec and rec.active),
                                  cfg.min_pose_visibility, "", "")
                # Consistent camera viewport on 360p/720p devices.
                if last_image.shape[:2] != (480, 640):
                    h, w = last_image.shape[:2]
                    scale = min(640 / w, 480 / h)
                    thumb = cv2.resize(last_image, (int(w * scale), int(h * scale)))
                    last_image = np.zeros((480, 640, 3), np.uint8)
                    y, x = (480 - thumb.shape[0]) // 2, (640 - thumb.shape[1]) // 2
                    last_image[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
                latency.append(1000 * (time.perf_counter() - t0))
                p.camera_status = f"Camara {cam.fps:.0f} FPS | Procesamiento {fps:.0f} FPS | {cam.backend}"
            elif now - last_frame_at > .35:
                runtime.clear_prediction(message="Esperando imagen de camara")
            cv2.imshow("signlens", p.draw(last_image))
            key = cv2.waitKey(1) & 0xFF
            if rec and key == ord(" "):
                saved = rec.toggle()
                if saved:
                    p.status = f"Secuencia guardada: {saved.name}"
            else:
                p.on_key(key)
            quit_requested = False
            for action in p.pop_actions():
                if action == "quit":
                    quit_requested = True
                elif action == "reconnect":
                    reconnect()
                else:
                    runtime.handle(action)
            if quit_requested or cv2.getWindowProperty("signlens", cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.duration and now - start >= args.duration:
                break
            if now - last_health >= 10:
                save_health()
                last_health = now
    finally:
        closing.set()
        runtime.close()
        if rec and rec.active:
            rec.toggle()
        save_health(ended=True)
        if cam:
            cam.release()
        if camera_thread:
            camera_thread.join(timeout=2)
        while not camera_events.empty():
            pending = camera_events.get_nowait()
            if not isinstance(pending, str):
                pending.release()
        tracker.close()
        cv2.destroyAllWindows()
