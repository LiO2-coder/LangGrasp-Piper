# NoPrint

[中文文档](README.md)

NoPrint is a small Python logging helper for scripts that start with `print()` and later need clean log files. It writes human-readable logs and structured events at the same time, with optional automatic capture of uncaught Python exceptions.

The project came from a very ordinary debugging pain: using `print()` everywhere is quick at first, but it becomes hard to search, manage, and share with AI tools once a project grows. I occasionally used `rospy.log`, but while building a MuJoCo project, Codex gave me the spark to make a lighter logger package that is easy to drop into experiments, debug sessions, and AI-assisted workflows.

## Features

- Standard-library only.
- Python 3.8 compatible.
- Class-based logger management.
- User-selected output directory.
- Multiple output formats: `log`, `jsonl`, `txt`, `markdown`.
- Levels such as `debug`, `info`, `warn`, `error`, and `critical`.
- Optional `sys.excepthook` and `threading.excepthook` integration.

## Local Use

From this repository root, import the package directly:

```python
from noprint import create_logger

log = create_logger(
    "/home/robot/logs",
    formats=("log", "jsonl", "markdown"),
    capture_exceptions=True,
)

log.info("程序启动")
log.warn("网络不可用，切换离线模式", backend="offline")
log.error("任务失败", task="grasp")
log.close()
```

## API

```python
from noprint import NoPrint, NoPrintConfig, create_logger
```

`create_logger(...)` is the simplest entrypoint:

```python
log = create_logger(
    log_dir="logs",
    formats=("log", "jsonl"),
    name=None,
    level="INFO",
    to_stdout=True,
    capture_exceptions=False,
)
```

`NoPrintConfig` exposes the same options when you want to construct the class directly.

## Output Files

File names use the current run time and the running script name:

```text
YYYYMMDD_HHMMSS_main.log
YYYYMMDD_HHMMSS_main.jsonl
YYYYMMDD_HHMMSS_main.txt
YYYYMMDD_HHMMSS_main.md
```

Format behavior:

- `log`: standard logging output for terminal-style reading and `tail`.
- `jsonl`: one JSON event per line with time, level, event, message, fields, and exception.
- `txt`: compact text output.
- `markdown`: readable Markdown sections for notes and reports.

## Exceptions

Enable automatic uncaught exception logging:

```python
log = create_logger("logs", capture_exceptions=True)
```

Or install hooks later:

```python
log.install_exception_hooks()
```

NoPrint records the exception and then lets Python keep printing the original traceback.

## Roadmap

TODO: development plans, also known as promises the author is making to their future self.

- Add optional log rotation by file size or date.
- Add simple colored console output.
- Add a CLI helper for quickly viewing recent logs.
- Add richer examples for robotics, simulation, and AI-assisted debugging workflows.

## Tests

Run this command from the project root directory, `NoPrint/`:

```bash
python3 -m unittest tests.test_noprint
```

## License

This project is licensed under the MIT License. See the[LICENSE](LICENSE)file for details.

## Author

GitHub: [https://github.com/LiO2-coder](https://github.com/LiO2-coder)

## Version History

- v1.0 - Initial release.
  - Multi-format logging output.
  - Simple class-based API and examples.
  - Optional uncaught Python exception capture.

## Support

If you encounter any issue, contact via:

- GitHub Issues: [https://github.com/LiO2-coder/NoPrint/issues](https://github.com/LiO2-coder/NoPrint/issues)

---
