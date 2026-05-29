# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from noprint import NoPrintConfig, create_logger


class NoPrintTests(unittest.TestCase):
    def test_multi_format_utf8_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(
                Path(temp_dir) / "logs",
                formats=("log", "jsonl", "txt", "markdown"),
                name="main",
                to_stdout=False,
            )
            logger.info("程序启动", item="红色杯子")
            logger.warn("网络不可用，切换离线模式", backend="offline")
            logger.error("任务失败", task="grasp")
            logger.close()

            self.assertTrue((Path(temp_dir) / "logs").exists())
            self.assertEqual(set(logger.paths), {"log", "jsonl", "txt", "markdown"})
            for path in logger.paths.values():
                self.assertTrue(path.exists(), path)
                self.assertIn("main", path.name)

            log_text = logger.paths["log"].read_text(encoding="utf-8")
            txt_text = logger.paths["txt"].read_text(encoding="utf-8")
            md_text = logger.paths["markdown"].read_text(encoding="utf-8")
            json_lines = logger.paths["jsonl"].read_text(encoding="utf-8").splitlines()
            payload = json.loads(json_lines[0])

            self.assertIn("程序启动", log_text)
            self.assertIn("[INFO] 程序启动", txt_text)
            self.assertIn("[WARN] 网络不可用", txt_text)
            self.assertIn("程序启动", md_text)
            self.assertEqual(payload["level"], "INFO")
            self.assertEqual(payload["event"], "info")
            self.assertEqual(payload["fields"]["item"], "红色杯子")

    def test_single_format_only_creates_selected_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(temp_dir, formats=("txt",), name="only_txt", to_stdout=False)
            logger.info("hello")
            logger.close()

            self.assertEqual(set(logger.paths), {"txt"})
            self.assertTrue(logger.paths["txt"].exists())
            self.assertEqual(len(list(Path(temp_dir).iterdir())), 1)
            self.assertEqual(logger.paths["txt"].suffix, ".txt")

    def test_debug_level_and_custom_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(temp_dir, formats=("jsonl",), name="events", level="DEBUG", to_stdout=False)
            logger.debug("debug detail", step=1)
            logger.event("voice_recognized", "语音识别完成", level="warn", text="红色杯子")
            logger.close()

            payloads = [
                json.loads(line)
                for line in logger.paths["jsonl"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(payloads[0]["level"], "DEBUG")
            self.assertEqual(payloads[0]["event"], "debug")
            self.assertEqual(payloads[1]["level"], "WARN")
            self.assertEqual(payloads[1]["event"], "voice_recognized")
            self.assertEqual(payloads[1]["fields"]["text"], "红色杯子")

    def test_exception_writes_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(temp_dir, formats=("jsonl", "txt"), name="errors", to_stdout=False)
            try:
                raise ValueError("bad thing")
            except ValueError:
                logger.exception("处理失败", path=Path("input.txt"))
            logger.close()

            payload = json.loads(logger.paths["jsonl"].read_text(encoding="utf-8").strip())
            txt_text = logger.paths["txt"].read_text(encoding="utf-8")

            self.assertEqual(payload["level"], "ERROR")
            self.assertEqual(payload["event"], "exception")
            self.assertEqual(payload["exception"]["type"], "ValueError")
            self.assertIn("bad thing", payload["exception"]["message"])
            self.assertIn("Traceback", txt_text)
            self.assertEqual(payload["fields"]["path"], "input.txt")

    def test_exception_hooks_install_and_close_restore(self) -> None:
        original_sys_hook = sys.excepthook
        original_threading_hook = getattr(threading, "excepthook", None)
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(
                temp_dir,
                formats=("jsonl",),
                name="hooks",
                to_stdout=False,
                capture_exceptions=True,
            )
            try:
                self.assertIsNot(sys.excepthook, original_sys_hook)
                if original_threading_hook is not None:
                    self.assertIsNot(threading.excepthook, original_threading_hook)
            finally:
                logger.close()

        self.assertIs(sys.excepthook, original_sys_hook)
        if original_threading_hook is not None:
            self.assertIs(threading.excepthook, original_threading_hook)

    def test_invalid_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                create_logger(temp_dir, formats=("csv",), to_stdout=False)

    def test_config_class_can_build_logger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger(
                log_dir=temp_dir,
                formats=NoPrintConfig(log_dir=temp_dir).formats,
                name="config",
                to_stdout=False,
            )
            logger.info("from config")
            logger.close()
            self.assertTrue(logger.paths["log"].exists())
            self.assertTrue(logger.paths["jsonl"].exists())


if __name__ == "__main__":
    unittest.main()
