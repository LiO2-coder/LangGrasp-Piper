# NoPrint

[English Version](README_en.md)

NoPrint 是一个轻量级 Python 日志管理包，用来替代调试阶段到处散落的 `print()`。它可以同时写入人工易读的日志和结构化事件，并支持自动接入 Python 未捕获异常，方便调试、复盘，也方便把日志交给 AI 分析。

## 项目缘起

以前调试总是用 `print()`，一开始确实方便，但项目稍微复杂一点就会变得效率低、不好搜索、不好管理，也不方便把关键上下文整理给 AI。偶尔会用一下 `rospy.log`，但并不是所有项目都在 ROS 里。

在做 MuJoCo 项目的时候，Codex 给了我一些灵感，于是做了这个轻松使用的 logger 日志管理包。目标很简单：少写样板代码，快速接入 Python 脚本，让调试记录更清楚，也让人与 AI 协作时有更完整的上下文。

## 特性

- 只使用 Python 标准库。
- 兼容 Python 3.8+。
- 类管理形式，创建和关闭都很简单。
- 用户可设置日志保存地址。
- 支持多种保存格式：`log`、`jsonl`、`txt`、`markdown`。
- 支持多个日志等级：`debug`、`info`、`warn`、`error`、`critical`。
- 可自动接入 `sys.excepthook` 和 `threading.excepthook`，记录主线程和子线程未捕获异常。

## 快速使用

在项目根目录下直接导入：

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

最简单的入口是 `create_logger(...)`：

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

如果你想显式管理配置，也可以直接使用 `NoPrintConfig` 和 `NoPrint`。

## 输出文件

日志文件名会使用当前运行时间和运行脚本名：

```text
YYYYMMDD_HHMMSS_main.log
YYYYMMDD_HHMMSS_main.jsonl
YYYYMMDD_HHMMSS_main.txt
YYYYMMDD_HHMMSS_main.md
```

格式说明：

- `log`：标准日志行，适合终端查看和 `tail`。
- `jsonl`：每行一个 JSON 事件，包含时间、等级、事件名、消息、字段和异常信息。
- `txt`：简洁文本日志。
- `markdown`：Markdown 小节格式，适合直接阅读和归档。

## 异常捕获

创建 logger 时开启：

```python
log = create_logger("logs", capture_exceptions=True)
```

或者稍后手动安装：

```python
log.install_exception_hooks()
```

NoPrint 会先记录异常，然后保留 Python 原始 traceback 输出，不吞掉系统报错。

## TODO: 开发计划（作者给自己画饼）

- 支持按日期或文件大小自动切分日志。
- 支持控制台彩色输出。
- 增加命令行工具，快速查看最近日志。
- 增加更多机器人、AI 协作调试示例。
- 增加更多文档和实际项目接入样例。

## 测试

请在项目根目录 `NoPrint/` 下运行：

```bash
python3 -m unittest tests.test_noprint
```

## 许可证

本项目采用 MIT 许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 作者

GitHub: [https://github.com/LiO2-coder](https://github.com/LiO2-coder)

## 版本历史

- v1.0 - 初始版本发布
  - 实现多格式日志输出：`log`、`jsonl`、`txt`、`markdown`
  - 提供简单的类管理 API、说明文档和示例
  - 支持 Python 未捕获异常记录，方便调试和 AI 交互

## 支持

如果您在使用过程中遇到任何问题，可以通过以下方式联系：

- GitHub Issues: [项目 Issues 页面](https://github.com/LiO2-coder/NoPrint/issues)

---

⭐ 如果这个项目对您有帮助，请给个Star！
