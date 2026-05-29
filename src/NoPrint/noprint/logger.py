# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union


SUPPORTED_FORMATS = frozenset(("log", "jsonl", "txt", "markdown"))
FORMAT_EXTENSIONS = {
    "log": "log",
    "jsonl": "jsonl",
    "txt": "txt",
    "markdown": "md",
}
DISPLAY_LEVELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


@dataclass(frozen=True)
class NoPrintConfig:
    log_dir: Union[str, Path]
    formats: Sequence[str] = ("log", "jsonl")
    name: Optional[str] = None
    level: Union[int, str] = logging.INFO
    to_stdout: bool = True
    capture_exceptions: bool = False


class NoPrint:
    def __init__(self, config: NoPrintConfig) -> None:
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.formats = self._normalize_formats(config.formats)
        self.level = self._normalize_level(config.level)
        self.name = self._resolve_name(config.name)
        self.started_at = datetime.now()
        self.stem = "{}_{}".format(self.started_at.strftime("%Y%m%d_%H%M%S"), self.name)
        self.paths = self._build_paths()

        self._lock = threading.RLock()
        self._closed = False
        self._hooks_installed = False
        self._previous_sys_excepthook = None
        self._previous_threading_excepthook = None
        self._sys_excepthook = None
        self._threading_excepthook = None
        self._logger = self._build_python_logger()

        if config.capture_exceptions:
            self.install_exception_hooks()

    def debug(self, message: str, **fields: Any) -> None:
        self._record(logging.DEBUG, "debug", message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._record(logging.INFO, "info", message, fields)

    def warn(self, message: str, **fields: Any) -> None:
        self.warning(message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._record(logging.WARNING, "warning", message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._record(logging.ERROR, "error", message, fields)

    def critical(self, message: str, **fields: Any) -> None:
        self._record(logging.CRITICAL, "critical", message, fields)

    def exception(self, message: str, exc: Optional[BaseException] = None, **fields: Any) -> None:
        if exc is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_value is None:
                self.error(message, **fields)
                return
        else:
            exc_type = type(exc)
            exc_value = exc
            exc_tb = exc.__traceback__

        self._record_exception(
            message=message,
            exc_type=exc_type,
            exc_value=exc_value,
            exc_tb=exc_tb,
            event=str(fields.pop("event", "exception")),
            fields=fields,
        )

    def event(self, event: str, message: Optional[str] = None, level: Union[int, str] = "info", **fields: Any) -> None:
        levelno = self._normalize_level(level)
        self._record(levelno, event, message if message is not None else event, fields)

    def install_exception_hooks(self) -> "NoPrint":
        if self._hooks_installed:
            return self

        self._previous_sys_excepthook = sys.excepthook
        self._previous_threading_excepthook = getattr(threading, "excepthook", None)

        def sys_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
            try:
                if not self._is_quiet_exception(exc_type):
                    self._record_exception(
                        message="Uncaught exception",
                        exc_type=exc_type,
                        exc_value=exc_value,
                        exc_tb=exc_tb,
                        event="uncaught_exception",
                        fields={},
                    )
            finally:
                if self._previous_sys_excepthook is not None:
                    self._previous_sys_excepthook(exc_type, exc_value, exc_tb)

        self._sys_excepthook = sys_excepthook
        sys.excepthook = sys_excepthook

        if hasattr(threading, "excepthook"):
            def threading_excepthook(args: Any) -> None:
                try:
                    if not self._is_quiet_exception(args.exc_type):
                        thread = getattr(args, "thread", None)
                        self._record_exception(
                            message="Uncaught thread exception",
                            exc_type=args.exc_type,
                            exc_value=args.exc_value,
                            exc_tb=args.exc_traceback,
                            event="uncaught_thread_exception",
                            fields={"thread": getattr(thread, "name", None)},
                        )
                finally:
                    if self._previous_threading_excepthook is not None:
                        self._previous_threading_excepthook(args)

            self._threading_excepthook = threading_excepthook
            threading.excepthook = threading_excepthook

        self._hooks_installed = True
        return self

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._restore_exception_hooks()
            for handler in list(self._logger.handlers):
                self._logger.removeHandler(handler)
                handler.close()
            self._closed = True

    def __enter__(self) -> "NoPrint":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _normalize_formats(formats: Union[str, Iterable[str]]) -> Tuple[str, ...]:
        if isinstance(formats, str):
            requested = (formats,)
        else:
            requested = tuple(formats)
        if not requested:
            raise ValueError("formats must contain at least one format.")

        normalized = []
        for item in requested:
            fmt = str(item).strip().lower()
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Unsupported log format {!r}. Supported formats: {}".format(
                        item,
                        ", ".join(sorted(SUPPORTED_FORMATS)),
                    )
                )
            if fmt not in normalized:
                normalized.append(fmt)
        return tuple(normalized)

    @staticmethod
    def _normalize_level(level: Union[int, str]) -> int:
        if isinstance(level, int):
            return level

        text = str(level).strip().upper()
        aliases = {"WARN": "WARNING", "ERR": "ERROR"}
        text = aliases.get(text, text)
        value = logging.getLevelName(text)
        if isinstance(value, int):
            return value
        raise ValueError("Unsupported log level {!r}.".format(level))

    @staticmethod
    def _resolve_name(name: Optional[str]) -> str:
        raw_name = name
        if not raw_name:
            argv0 = sys.argv[0] if sys.argv else ""
            raw_name = Path(argv0).stem if argv0 else "python"
        cleaned = re.sub(r"\s+", "_", str(raw_name).strip())
        cleaned = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", cleaned)
        return cleaned.strip("._-") or "python"

    @staticmethod
    def _display_level(levelno: int) -> str:
        return DISPLAY_LEVELS.get(levelno, logging.getLevelName(levelno))

    @staticmethod
    def _is_quiet_exception(exc_type: Optional[type]) -> bool:
        return exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit))

    def _build_paths(self) -> Dict[str, Path]:
        return {
            fmt: self.log_dir / "{}.{}".format(self.stem, FORMAT_EXTENSIONS[fmt])
            for fmt in self.formats
        }

    def _build_python_logger(self) -> logging.Logger:
        logger = logging.getLogger("noprint.{}.{}".format(self.stem, id(self)))
        logger.setLevel(self.level)
        logger.propagate = False

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if "log" in self.formats:
            file_handler = logging.FileHandler(self.paths["log"], encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if self.config.to_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(self.level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def _record(
        self,
        levelno: int,
        event: str,
        message: str,
        fields: Dict[str, Any],
        exception_payload: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Tuple[type, BaseException, Any]] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("NoPrint is already closed.")
        if levelno < self.level:
            return

        now = datetime.now()
        display_level = self._display_level(levelno)
        safe_fields = self._json_ready(fields)
        safe_exception = self._json_ready(exception_payload) if exception_payload else None

        with self._lock:
            human_message = self._format_human_message(message, safe_fields)
            if self._logger.handlers:
                self._logger.log(levelno, human_message, exc_info=exc_info)

            if "jsonl" in self.formats:
                self._write_jsonl(now, display_level, event, message, safe_fields, safe_exception)
            if "txt" in self.formats:
                self._write_txt(now, display_level, event, message, safe_fields, safe_exception)
            if "markdown" in self.formats:
                self._write_markdown(now, display_level, event, message, safe_fields, safe_exception)

    def _record_exception(
        self,
        message: str,
        exc_type: type,
        exc_value: BaseException,
        exc_tb: Any,
        event: str,
        fields: Dict[str, Any],
    ) -> None:
        exception_payload = self._format_exception_payload(exc_type, exc_value, exc_tb)
        self._record(
            logging.ERROR,
            event,
            message,
            fields,
            exception_payload=exception_payload,
            exc_info=(exc_type, exc_value, exc_tb),
        )

    def _write_jsonl(
        self,
        now: datetime,
        level: str,
        event: str,
        message: str,
        fields: Any,
        exception_payload: Optional[Any],
    ) -> None:
        payload = {
            "time": now.isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "message": message,
            "fields": fields,
            "exception": exception_payload,
        }
        with self.paths["jsonl"].open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=str))
            file.write("\n")

    def _write_txt(
        self,
        now: datetime,
        level: str,
        event: str,
        message: str,
        fields: Any,
        exception_payload: Optional[Any],
    ) -> None:
        line = "{} [{}] {}{}".format(
            now.strftime("%Y-%m-%d %H:%M:%S"),
            level,
            message,
            self._format_fields_suffix(fields),
        )
        with self.paths["txt"].open("a", encoding="utf-8") as file:
            file.write(line)
            file.write("\n")
            if event:
                file.write("event={}\n".format(event))
            if exception_payload:
                file.write(exception_payload.get("traceback", ""))
                if not str(exception_payload.get("traceback", "")).endswith("\n"):
                    file.write("\n")

    def _write_markdown(
        self,
        now: datetime,
        level: str,
        event: str,
        message: str,
        fields: Any,
        exception_payload: Optional[Any],
    ) -> None:
        with self.paths["markdown"].open("a", encoding="utf-8") as file:
            file.write("### {} [{}] {}\n\n".format(now.strftime("%Y-%m-%d %H:%M:%S"), level, event))
            file.write("{}\n\n".format(message))
            if fields:
                file.write("```json\n")
                file.write(json.dumps(fields, ensure_ascii=False, indent=2, default=str))
                file.write("\n```\n\n")
            if exception_payload:
                file.write("```text\n")
                file.write(exception_payload.get("traceback", ""))
                if not str(exception_payload.get("traceback", "")).endswith("\n"):
                    file.write("\n")
                file.write("```\n\n")

    @staticmethod
    def _format_exception_payload(exc_type: type, exc_value: BaseException, exc_tb: Any) -> Dict[str, Any]:
        return {
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        }

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, BaseException):
            return repr(value)
        if isinstance(value, dict):
            return {str(key): cls._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_ready(item) for item in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    @classmethod
    def _format_human_message(cls, message: str, fields: Any) -> str:
        return "{}{}".format(message, cls._format_fields_suffix(fields))

    @staticmethod
    def _format_fields_suffix(fields: Any) -> str:
        if not fields:
            return ""
        return " | fields={}".format(json.dumps(fields, ensure_ascii=False, default=str, sort_keys=True))

    def _restore_exception_hooks(self) -> None:
        if not self._hooks_installed:
            return
        if self._sys_excepthook is not None and sys.excepthook is self._sys_excepthook:
            sys.excepthook = self._previous_sys_excepthook
        if (
            self._threading_excepthook is not None
            and hasattr(threading, "excepthook")
            and threading.excepthook is self._threading_excepthook
        ):
            threading.excepthook = self._previous_threading_excepthook
        self._hooks_installed = False


def create_logger(
    log_dir: Union[str, Path],
    formats: Sequence[str] = ("log", "jsonl"),
    name: Optional[str] = None,
    **options: Any
) -> NoPrint:
    config = NoPrintConfig(log_dir=log_dir, formats=formats, name=name, **options)
    return NoPrint(config)
