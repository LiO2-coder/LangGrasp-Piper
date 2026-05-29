# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

from vosk import KaldiRecognizer, Model

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODULE_DIR / "model" / "vosk-model-small-cn-0.22"


@dataclass(frozen=True)
class RecognitionConfig:
    model_path: str = str(DEFAULT_MODEL_PATH)
    sample_rate: int = 16000


class OfflineSpeechRecognizer:
    """Offline speech recognizer based on Vosk.

    This class only handles recognition and text output.
    It does not handle recording or keyboard events.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self.model_path = self._resolve_model_path(self.config.model_path)
        self.model = Model(str(self.model_path))
        self._print_info(f"Vosk model loaded: {self.model_path}")

    @staticmethod
    def _print_info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def _print_warn(message: str) -> None:
        print(f"[WARN] {message}")

    @staticmethod
    def _print_error(message: str) -> None:
        print(f"[ERR ] {message}")

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path)
        if path.is_absolute():
            return path
        return (MODULE_DIR / path).resolve()

    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe one wav file and return plain text.

        Args:
            audio_file: Path to a mono PCM wav file.

        Returns:
            Concatenated recognized text. Empty string if nothing is recognized.
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_file}")

        recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
        text_pieces: list[str] = []

        try:
            wave_file = wave.open(str(audio_path), "rb")
        except wave.Error as exc:
            raise ValueError(f"Audio file must be a readable wav file: {audio_file}") from exc

        with wave_file:
            if wave_file.getnchannels() != 1:
                raise ValueError("Offline recognizer requires mono wav audio.")
            if wave_file.getframerate() != self.config.sample_rate:
                raise ValueError(
                    f"Offline recognizer requires {self.config.sample_rate} Hz wav audio."
                )
            if wave_file.getnframes() <= 0:
                raise ValueError("Audio file contains no frames.")

            while True:
                data = wave_file.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        text_pieces.append(text)

            final_result = json.loads(recognizer.FinalResult())
            final_text_piece = final_result.get("text", "").strip()
            if final_text_piece:
                text_pieces.append(final_text_piece)

        final_text = " ".join(text_pieces).strip()
        if final_text:
            self._print_info(f"Recognized text: {final_text}")
        else:
            self._print_warn("No text recognized.")
        return final_text


def main() -> None:
    recognizer = OfflineSpeechRecognizer(RecognitionConfig())
    audio_file = str(MODULE_DIR / "temp" / "voice" / "temp_audio.wav")
    try:
        recognizer.transcribe_file(audio_file)
    except Exception as exc:  # noqa: BLE001
        recognizer._print_error(str(exc))


if __name__ == "__main__":
    main()
