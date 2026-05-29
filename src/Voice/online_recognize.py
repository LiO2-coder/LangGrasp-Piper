# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from tencentcloud.common import credential
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.asr.v20190614 import asr_client, models
    TENCENT_ASR_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    credential = None  # type: ignore[assignment]
    ClientProfile = None  # type: ignore[assignment]
    HttpProfile = None  # type: ignore[assignment]
    TencentCloudSDKException = Exception  # type: ignore[assignment,misc]
    asr_client = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    TENCENT_ASR_IMPORT_ERROR = exc


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = MODULE_DIR / "config" / "TX-cloud_API.yaml"
DEFAULT_REQUEST_CONFIG_PATH = MODULE_DIR / "config" / "request.json"
DEFAULT_INLINE_AUDIO_LIMIT_BYTES = 5 * 1024 * 1024


class OnlineRecognitionError(RuntimeError):
    """Raised when Tencent online ASR cannot provide a usable result."""


@dataclass(frozen=True)
class TencentASRConfig:
    region: str = "ap-guangzhou"
    engine_model_type: str = "16k_zh"
    channel_num: int = 1
    res_text_format: int = 0
    network_timeout_s: float = 10.0
    poll_interval_s: float = 0.5
    poll_timeout_s: float = 30.0
    endpoint: str = "asr.tencentcloudapi.com"
    credentials_file: str = str(DEFAULT_CREDENTIALS_PATH)
    request_config_file: str = str(DEFAULT_REQUEST_CONFIG_PATH)
    max_inline_audio_bytes: int = DEFAULT_INLINE_AUDIO_LIMIT_BYTES


class TencentASRRecognizer:
    """Tencent Cloud ASR client for local audio files."""

    def __init__(self, config: TencentASRConfig) -> None:
        self.config = config
        self._client = None
        self._request_defaults: dict[str, Any] | None = None

    @staticmethod
    def _print_info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def _print_warn(message: str) -> None:
        print(f"[WARN] {message}")

    @staticmethod
    def _print_error(message: str) -> None:
        print(f"[ERR ] {message}")

    def transcribe_file(self, audio_file: str) -> str:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        client = self._get_client()
        payload = self._build_request_payload(audio_path)
        request = models.CreateRecTaskRequest()
        request.from_json_string(json.dumps(payload))

        try:
            response = client.CreateRecTask(request)
        except TencentCloudSDKException as exc:
            raise OnlineRecognitionError(f"Tencent CreateRecTask failed: {exc}") from exc

        task = getattr(response, "Data", None)
        task_id = getattr(task, "TaskId", None)
        if task_id is None:
            raise OnlineRecognitionError("Tencent CreateRecTask did not return a TaskId.")

        self._print_info(f"Tencent ASR task created: {task_id}")
        result = self._poll_task_result(client, task_id)
        if not result:
            raise OnlineRecognitionError("Tencent ASR returned an empty transcription.")

        self._print_info(f"Online recognized text: {result}")
        return result

    def _get_client(self):
        if self._client is not None:
            return self._client

        self._ensure_sdk_available()
        secret_id, secret_key, region = self._load_credentials()
        cred = credential.Credential(secret_id, secret_key)

        http_profile = HttpProfile(
            endpoint=self.config.endpoint,
            reqTimeout=max(1, int(math.ceil(self.config.network_timeout_s))),
        )
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        self._client = asr_client.AsrClient(cred, region, client_profile)
        return self._client

    def _ensure_sdk_available(self) -> None:
        if TENCENT_ASR_IMPORT_ERROR is not None:
            raise OnlineRecognitionError(
                "Tencent Cloud ASR SDK is unavailable. "
                "Install `tencentcloud-sdk-python-asr` in the target environment."
            ) from TENCENT_ASR_IMPORT_ERROR

    def _load_credentials(self) -> tuple[str, str, str]:
        secret_id = os.getenv("TENCENTCLOUD_SECRET_ID", "").strip()
        secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY", "").strip()
        region = os.getenv("TENCENTCLOUD_REGION", "").strip() or self.config.region

        if secret_id and secret_key:
            return secret_id, secret_key, region

        credential_path = self._resolve_path(self.config.credentials_file)
        config_data = self._load_yaml_dict(credential_path)
        secret_id = str(config_data.get("ID", "")).strip()
        secret_key = str(config_data.get("KEY", "")).strip()
        region = str(config_data.get("Region", "")).strip() or region

        if not secret_id or not secret_key:
            raise OnlineRecognitionError(
                "Tencent Cloud credentials are missing. "
                "Set TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY or provide TX-cloud_API.yaml."
            )

        return secret_id, secret_key, region

    def _load_request_defaults(self) -> dict[str, Any]:
        if self._request_defaults is not None:
            return self._request_defaults

        request_path = self._resolve_path(self.config.request_config_file)
        if not request_path.exists():
            self._request_defaults = {}
            return self._request_defaults

        data = self._load_json_dict(request_path)
        self._request_defaults = {key: value for key, value in data.items() if value is not None}
        return self._request_defaults

    def _build_request_payload(self, audio_path: Path) -> dict[str, Any]:
        payload = dict(self._load_request_defaults())
        payload["EngineModelType"] = self.config.engine_model_type
        payload["ChannelNum"] = self.config.channel_num
        payload["ResTextFormat"] = self.config.res_text_format

        source_type = payload.get("SourceType")
        url = str(payload.get("Url", "")).strip() if payload.get("Url") is not None else ""
        if source_type == 0 and url:
            payload["SourceType"] = 0
            payload["Url"] = url
            payload.pop("Data", None)
            payload.pop("DataLen", None)
            return payload

        audio_bytes = audio_path.read_bytes()
        if len(audio_bytes) > self.config.max_inline_audio_bytes:
            raise OnlineRecognitionError(
                "Audio file is too large for Tencent inline upload. "
                f"Size={len(audio_bytes)} bytes, limit={self.config.max_inline_audio_bytes} bytes."
            )

        payload["SourceType"] = 1
        payload["Data"] = base64.b64encode(audio_bytes).decode("utf-8")
        payload["DataLen"] = len(audio_bytes)
        payload.pop("Url", None)
        return payload

    def _poll_task_result(self, client, task_id: int) -> str:
        deadline = time.monotonic() + self.config.poll_timeout_s

        while True:
            request = models.DescribeTaskStatusRequest()
            request.TaskId = task_id

            try:
                response = client.DescribeTaskStatus(request)
            except TencentCloudSDKException as exc:
                raise OnlineRecognitionError(f"Tencent DescribeTaskStatus failed: {exc}") from exc

            task_status = getattr(response, "Data", None)
            if task_status is None:
                raise OnlineRecognitionError("Tencent DescribeTaskStatus returned no task data.")

            status = (task_status.StatusStr or "").strip().lower()
            status_code = getattr(task_status, "Status", None)

            if status == "success" or status_code == 2:
                return (task_status.Result or "").strip()

            if status == "failed" or status_code == 3:
                error_message = (task_status.ErrorMsg or "").strip() or "Tencent ASR task failed."
                raise OnlineRecognitionError(error_message)

            if time.monotonic() >= deadline:
                raise OnlineRecognitionError(
                    f"Tencent ASR polling timed out after {self.config.poll_timeout_s:.1f}s."
                )

            time.sleep(self.config.poll_interval_s)

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (MODULE_DIR / path).resolve()

    @staticmethod
    def _load_yaml_dict(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise OnlineRecognitionError(f"Invalid YAML config: {path}") from exc
        except OSError as exc:
            raise OnlineRecognitionError(f"Unable to read YAML config: {path}") from exc

        if not isinstance(data, dict):
            raise OnlineRecognitionError(f"YAML config must be a dictionary: {path}")
        return data

    @staticmethod
    def _load_json_dict(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise OnlineRecognitionError(f"Invalid JSON config: {path}") from exc
        except OSError as exc:
            raise OnlineRecognitionError(f"Unable to read JSON config: {path}") from exc

        if not isinstance(data, dict):
            raise OnlineRecognitionError(f"JSON config must be an object: {path}")
        return data


def main() -> None:
    import sys

    audio_file = sys.argv[1] if len(sys.argv) > 1 else str(MODULE_DIR / "temp" / "voice" / "temp_audio.wav")
    recognizer = TencentASRRecognizer(TencentASRConfig())

    try:
        recognizer.transcribe_file(audio_file)
    except Exception as exc:  # noqa: BLE001
        recognizer._print_error(str(exc))


if __name__ == "__main__":
    main()
