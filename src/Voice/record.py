# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import pyaudio


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_FILE = MODULE_DIR / "temp" / "voice" / "temp_audio.wav"


@dataclass(frozen=True)
class RecordingConfig:
    audio_file: str = str(DEFAULT_AUDIO_FILE)
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    audio_format: int = pyaudio.paInt16


class AudioRecorder:
    """Programmatic audio recorder for wav capture."""

    def __init__(self, config: RecordingConfig) -> None:
        self.config = config
        self._audio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._record_thread: threading.Thread | None = None

        self.is_recording = False
        self.audio_frames: list[bytes] = []
        self.last_audio_file: str | None = None

    @staticmethod
    def _print_info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def _print_ok(message: str) -> None:
        print(f"[ OK ] {message}")

    @staticmethod
    def _print_warn(message: str) -> None:
        print(f"[WARN] {message}")

    @staticmethod
    def _print_error(message: str) -> None:
        print(f"[ERR ] {message}")

    def start(self) -> None:
        if self.is_recording:
            raise RuntimeError("Recorder is already recording.")

        self._ensure_audio()
        assert self._audio is not None

        self.audio_frames = []
        self.last_audio_file = None
        self.is_recording = True
        self._stream = self._audio.open(
            format=self.config.audio_format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
        )
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        self._print_info("Recording started.")

    def stop(self) -> str:
        if not self.is_recording:
            raise RuntimeError("Recorder is not recording.")

        self.is_recording = False
        self._join_record_thread()
        self._close_stream()

        if not self.audio_frames:
            raise RuntimeError("No audio frames were captured.")

        audio_path = self._save_audio_to_file()
        self.last_audio_file = str(audio_path)
        self._print_ok(f"Audio saved: {self.last_audio_file}")
        return self.last_audio_file

    def shutdown(self) -> None:
        self.is_recording = False
        self._join_record_thread()
        self._close_stream()
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None
        self._print_ok("Recorder resources released.")

    def _ensure_audio(self) -> None:
        if self._audio is None:
            self._audio = pyaudio.PyAudio()

    def _record_loop(self) -> None:
        assert self._stream is not None
        while self.is_recording:
            data = self._stream.read(
                self.config.chunk_size,
                exception_on_overflow=False,
            )
            self.audio_frames.append(data)

    def _join_record_thread(self) -> None:
        if self._record_thread is not None and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)
        self._record_thread = None

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _save_audio_to_file(self) -> Path:
        if self._audio is None:
            raise RuntimeError("Audio interface is not initialized.")

        audio_path = self._resolve_audio_path(self.config.audio_file)
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(audio_path), "wb") as wave_file:
            wave_file.setnchannels(self.config.channels)
            wave_file.setsampwidth(self._audio.get_sample_size(self.config.audio_format))
            wave_file.setframerate(self.config.sample_rate)
            wave_file.writeframes(b"".join(self.audio_frames))

        return audio_path

    @staticmethod
    def _resolve_audio_path(audio_file: str) -> Path:
        path = Path(audio_file)
        if path.is_absolute():
            return path
        return (MODULE_DIR / path).resolve()


def main() -> None:
    recorder = AudioRecorder(RecordingConfig())
    print("=" * 56)
    print("Recorder demo (press ENTER to start, ENTER again to stop)")
    print("=" * 56)

    try:
        input("Press ENTER to start recording...")
        recorder.start()
        input("Recording... Press ENTER to stop.")
        audio_file = recorder.stop()
        recorder._print_info(f"Saved audio file: {audio_file}")
    except KeyboardInterrupt:
        recorder._print_info("KeyboardInterrupt received. Exiting...")
    except Exception as exc:  # noqa: BLE001
        recorder._print_error(str(exc))
    finally:
        recorder.shutdown()


if __name__ == "__main__":
    main()
