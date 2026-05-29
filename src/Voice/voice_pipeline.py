# -*- coding: utf-8 -*-
from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path


from .offline_recognize import OfflineSpeechRecognizer, RecognitionConfig
from .online_recognize import TencentASRConfig, TencentASRRecognizer
from .record import AudioRecorder, RecordingConfig


MODULE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIO_FILE = MODULE_DIR / "temp" / "voice" / "temp_audio.wav"
DEFAULT_MODEL_PATH = MODULE_DIR / "model" / "vosk-model-small-cn-0.22"
DEFAULT_CREDENTIALS_PATH = MODULE_DIR / "config" / "TX-cloud_API.yaml"
DEFAULT_REQUEST_CONFIG_PATH = MODULE_DIR / "config" / "request.json"


@dataclass(frozen=True)
class VoicePipelineConfig:
    audio_file: str = str(DEFAULT_AUDIO_FILE)
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    offline_model_path: str = str(DEFAULT_MODEL_PATH)
    tencent_credentials_file: str = str(DEFAULT_CREDENTIALS_PATH)
    tencent_request_config_file: str = str(DEFAULT_REQUEST_CONFIG_PATH)
    tencent_region: str = "ap-guangzhou"
    tencent_engine_model_type: str = "16k_zh"
    tencent_res_text_format: int = 0
    network_probe_host: str = "asr.tencentcloudapi.com"
    network_probe_port: int = 443
    network_probe_timeout_s: float = 0.8
    online_network_timeout_s: float = 10.0
    online_poll_interval_s: float = 0.5
    online_poll_timeout_s: float = 30.0
    max_inline_audio_bytes: int = 5 * 1024 * 1024

    def build_recording_config(self) -> RecordingConfig:
        return RecordingConfig(
            audio_file=self.audio_file,
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_size=self.chunk_size,
        )

    def build_offline_config(self) -> RecognitionConfig:
        return RecognitionConfig(
            model_path=self.offline_model_path,
            sample_rate=self.sample_rate,
        )

    def build_online_config(self) -> TencentASRConfig:
        return TencentASRConfig(
            region=self.tencent_region,
            engine_model_type=self.tencent_engine_model_type,
            channel_num=self.channels,
            res_text_format=self.tencent_res_text_format,
            network_timeout_s=self.online_network_timeout_s,
            poll_interval_s=self.online_poll_interval_s,
            poll_timeout_s=self.online_poll_timeout_s,
            credentials_file=self.tencent_credentials_file,
            request_config_file=self.tencent_request_config_file,
            max_inline_audio_bytes=self.max_inline_audio_bytes,
        )


class VoicePipeline:
    """Unified recording and speech recognition entrypoint."""

    def __init__(self, config: VoicePipelineConfig | None = None) -> None:
        self.config = config or VoicePipelineConfig()
        self.last_audio_file: str | None = None
        self.last_text: str | None = None
        self.last_backend: str | None = None
        self.last_online_error: str | None = None

        self._recorder: AudioRecorder | None = None
        self._offline_recognizer: OfflineSpeechRecognizer | None = None
        self._online_recognizer: TencentASRRecognizer | None = None

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    @staticmethod
    def _print_info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def _print_warn(message: str) -> None:
        print(f"[WARN] {message}")

    @staticmethod
    def _print_error(message: str) -> None:
        print(f"[ERR ] {message}")

    def start(self) -> None:
        if self.is_recording:
            raise RuntimeError("Voice pipeline is already recording.")

        self.last_text = None
        self.last_backend = None
        self.last_online_error = None
        self._get_recorder().start()

    def stop(self) -> str:
        if not self.is_recording:
            raise RuntimeError("Voice pipeline is not recording.")

        audio_file = self._get_recorder().stop()
        return self.transcribe_file(audio_file)

    def transcribe_file(self, audio_file: str) -> str:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_file}")

        self.last_audio_file = str(audio_path)
        self.last_text = None
        self.last_backend = None
        self.last_online_error = None

        if not self._network_is_healthy():
            reason = self._network_failure_reason()
            return self._transcribe_offline(audio_path, reason)

        try:
            result = self._get_online_recognizer().transcribe_file(str(audio_path))
        except Exception as exc:  # noqa: BLE001
            return self._transcribe_offline(audio_path, str(exc))

        if not result.strip():
            return self._transcribe_offline(audio_path, "Tencent ASR returned an empty transcription.")

        self.last_backend = "online"
        self.last_text = result
        return result

    def shutdown(self) -> None:
        if self._recorder is not None:
            self._recorder.shutdown()
            self._recorder = None

    def _network_is_healthy(self) -> bool:
        try:
            with socket.create_connection(
                (self.config.network_probe_host, self.config.network_probe_port),
                timeout=self.config.network_probe_timeout_s,
            ):
                return True
        except OSError:
            return False

    def _network_failure_reason(self) -> str:
        return (
            "Network probe failed: "
            f"{self.config.network_probe_host}:{self.config.network_probe_port} "
            f"was unreachable within {self.config.network_probe_timeout_s:.1f}s."
        )

    def _transcribe_offline(self, audio_path: Path, reason: str) -> str:
        self.last_backend = "offline"
        self.last_online_error = reason
        self._print_warn(f"{reason} Falling back to offline recognition.")
        result = self._get_offline_recognizer().transcribe_file(str(audio_path))
        self.last_text = result
        return result

    def _get_recorder(self) -> AudioRecorder:
        if self._recorder is None:
            self._recorder = AudioRecorder(self.config.build_recording_config())
        return self._recorder

    def _get_offline_recognizer(self) -> OfflineSpeechRecognizer:
        if self._offline_recognizer is None:
            self._offline_recognizer = OfflineSpeechRecognizer(self.config.build_offline_config())
        return self._offline_recognizer

    def _get_online_recognizer(self) -> TencentASRRecognizer:
        if self._online_recognizer is None:
            self._online_recognizer = TencentASRRecognizer(self.config.build_online_config())
        return self._online_recognizer
