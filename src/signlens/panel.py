"""Camera subtitles, independent hand readouts and keyboard-only data tools."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import unicodedata

import cv2

from signlens.types import HANDS


@dataclass
class HandReadout:
    confirmed: str = ""
    provisional: str = ""
    recognition: str = "Sin mano visible"


def display_text(value):
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()


def text(image, value, x, y, scale=.47, color=(228, 234, 241), thickness=1):
    cv2.putText(image, display_text(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrap_pixels(value, width, scale=.8, thickness=2):
    """Wrap long words too, using the actual rendered width."""
    lines, line = [], ""
    for char in display_text(value):
        candidate = line + char
        if char == "\n":
            lines.append(line)
            line = ""
        elif cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > width:
            split = line.rfind(" ")
            if split > 0:
                lines.append(line[:split])
                line = line[split + 1:] + char
            else:
                lines.append(line)
                line = char
        else:
            line = candidate
    if line or not lines:
        lines.append(line)
    return lines


class ControlPanel:
    def __init__(self, labels: list[str], hand: str = "Right"):
        self.labels = sorted(map(str, labels)) or list("ABCDEFGHIKLMNOPQRSTUWXY")
        self.labels += ["NINGUNA"] if "NINGUNA" not in self.labels else []
        self.label_index, self.hand = 0, hand
        self.capturing = self.training = self.evaluating = False
        self.sample_count = self.take_count = self.label_count = 0
        self.signer, self.session = "me", ""
        self.reserved, self.demo = False, False
        self.view, self.help_open, self.composer = "write", False, None
        self.status = "Subtitulos automaticos: ambas manos."
        self.model_date = "sin modelo"
        self.results = {side: HandReadout() for side in HANDS}
        self.camera_status = "Iniciando camara..."
        self.editing_signer, self.signer_input = False, ""
        self._actions = deque()
        self._buttons = []  # No clickable controls in either view.

    @property
    def confirmed(self):
        return self.results[self.hand].confirmed

    @confirmed.setter
    def confirmed(self, value):
        self.results[self.hand].confirmed = value

    @property
    def provisional(self):
        return self.results[self.hand].provisional

    @provisional.setter
    def provisional(self, value):
        self.results[self.hand].provisional = value

    @property
    def recognition(self):
        return self.results[self.hand].recognition

    @recognition.setter
    def recognition(self, value):
        self.results[self.hand].recognition = value

    @property
    def label(self):
        return self.labels[self.label_index]

    @property
    def busy(self):
        return self.capturing or self.training or self.evaluating

    def enabled(self, action):
        if action == "capture":
            return not self.training and not self.evaluating
        if action in {"cancel", "reconnect", "help"}:
            return True
        return not self.busy

    def request(self, action):
        if not self.editing_signer and self.enabled(action):
            self._actions.append(action)

    def on_mouse(self, event, x, y, flags, param):
        pass

    def pop_actions(self):
        out = list(self._actions)
        self._actions.clear()
        return out

    def on_key(self, key):
        if key < 0 or key == 255:
            return
        if self.editing_signer:
            if key == 27:
                self.editing_signer = False
            elif key in (10, 13):
                self._actions.append("save_signer")
            elif key in (8, 127):
                self.signer_input = self.signer_input[:-1]
            elif 32 <= key <= 126 and len(self.signer_input) < 48:
                self.signer_input += chr(key)
            return
        if key in (27, ord("q")):
            self._actions.append("quit")
            return
        if key == ord("?"):
            self.request("help")
            return
        if key == 9:
            self.request("view_data" if self.view == "write" else "view_write")
            return
        if self.view == "write":
            special = {8: "backspace", 127: "backspace", 26: "undo_text", 19: "save_text", 21: "clear_text"}
            if key in special:
                self.request(special[key])
                return
            mapping = {" ": "space", ".": "period", "r": "reconnect", "g": "gesture"}
        else:
            mapping = {"c": "capture", " ": "capture", "t": "train", "h": "hand", "l": "next",
                       "k": "previous", "v": "reserved", "p": "signer", "n": "session",
                       "d": "mode", "e": "evaluate", "r": "reconnect", "x": "cancel"}
        if chr(key).lower() in mapping:
            self.request(mapping[chr(key).lower()])

    def draw(self, image):
        canvas = image.copy()
        height, width = canvas.shape[:2]
        canvas[:42] = (20, 24, 29)
        text(canvas, "SignLens", 14, 27, .56, thickness=2)
        for i, side in enumerate(HANDS):
            state = self.results[side]
            name = "IZQ" if side == "Left" else "DER"
            text(canvas, f"{name}  {state.confirmed or '--'}", 160 + i * 140, 27, .55, thickness=2)
        text(canvas, "? ayuda", width - 79, 26, .38)

        body = self.composer.text if self.composer else ""
        if self.view == "write":
            lines = wrap_pixels(body.strip(), width - 48) if body.strip() else []
            band_height = 108 if len(lines) > 1 else 76
            canvas[-band_height:] = (canvas[-band_height:].astype("float32") * .18).astype("uint8")
            if lines:
                shown = lines[-2:]
                for i, line in enumerate(shown):
                    tw = cv2.getTextSize(display_text(line), cv2.FONT_HERSHEY_SIMPLEX, .8, 2)[0][0]
                    y = height - 40 - (len(shown) - 1 - i) * 33
                    text(canvas, line, (width - tw) // 2, y, .8, (255, 255, 255), 2)
                hint = "Retira ambas manos 1,4 s para separar palabras"
                if len(body) >= 1000:
                    hint = "Texto lleno: Ctrl+S guarda | Ctrl+U vacia | Ctrl+Z deshace"
            else:
                text(canvas, "Haz una sena para comenzar", 24, height - 40, .65, thickness=2)
                hint = "Ambas manos escriben automaticamente"
            text(canvas, hint, 24, height - 14, .38)
            if self.status != "Subtitulos automaticos: ambas manos.":
                self._notice(canvas, self.status, 65, width)
            if "FPS" not in self.camera_status:
                self._notice(canvas, self.camera_status, 104, width)
        else:
            destination = "EVALUAR" if self.reserved or self.label == "NINGUNA" else "ENTRENAR"
            side = "DERECHA" if self.hand == "Right" else "IZQUIERDA"
            rows = ["DATOS  |  Tab: volver a subtitulos", f"Letra: {self.label}   |   Mano a grabar: {side}",
                    f"Destino: {destination}   |   Muestras: {self.label_count}",
                    f"Persona: {self.signer}", f"Sesion: {self.session}",
                    "C: iniciar / guardar toma   |   K/L: cambiar letra", "H: mano a grabar   |   V: destino",
                    "P: persona   |   N: nueva sesion", "T: entrenar   |   X: cancelar   |   E: evaluar",
                    "D: modo limitado / completo   |   R: reconectar",
                    f"Modo: {'LIMITADO (10)' if self.demo else 'COMPLETO (23)'}  |  Modelo: {self.model_date}",
                    f"Toma actual: {self.take_count} muestras" if self.capturing else self.camera_status]
            self._sheet(canvas, rows, width)
            self._notice(canvas, self.status, height - 50, width)
        if self.help_open:
            self._sheet(canvas, ["SUBTITULOS AUTOMATICOS", "Las dos manos aportan letras sin activar nada.",
                        "Cada letra estable se agrega una vez por mano.", "Empate simultaneo: izquierda y luego derecha.",
                        "Misma letra en ambas manos: se agregan dos letras.",
                        "Repetir: retira esa mano 0,6 s y vuelve a mostrarla.",
                        "Separar palabras: retira ambas manos 1,4 s.",
                        "G: activar/desactivar borrado por gesto experimental", "Retroceso: borrar | Ctrl+Z: deshacer | Espacio: separar",
                        "Ctrl+S: guardar texto   |   Ctrl+U: vaciar", "Tab: herramientas de datos   |   R: reconectar",
                        "?: cerrar ayuda   |   Q / Esc: salir"], width)
        if self.editing_signer:
            self._sheet(canvas, ["Identificador de persona", (self.signer_input or "_")[-48:],
                                "Enter: guardar   |   Esc: cancelar"], width)
        return canvas

    @staticmethod
    def _notice(canvas, value, y, width):
        lines = wrap_pixels(value, width - 48, .4, 1)[:2]
        cv2.rectangle(canvas, (12, y - 17), (width - 12, y + 20 * (len(lines) - 1) + 5), (20, 24, 29), -1)
        for i, line in enumerate(lines):
            text(canvas, line, 24, y + i * 20, .4)

    @staticmethod
    def _sheet(canvas, rows, width):
        cv2.rectangle(canvas, (12, 54), (width - 12, 74 + len(rows) * 26), (20, 24, 29), -1)
        for i, row in enumerate(rows):
            value = display_text(row)
            measured = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, .46, 1)[0][0]
            scale = .46 * min(1, (width - 50) / max(measured, 1))
            text(canvas, value, 24, 80 + i * 26, scale)

