# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import count
from queue import Empty, Queue
from typing import Any, Callable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.widgets import Button, RectangleSelector, TextBox

from src.grasp_sequence import MotionRuntime, apply_home_pose, execute_pick_place_sequence_with_runtime
from src.ik.piper_ik import PiperIKController
from src.Vision import CaptureResult, DetectionResult, GraspResult, SegmentationResult, VisionPipeline
from src.Voice.voice_pipeline import VoicePipeline


REFRESH_INTERVAL_MS = 120
BUTTON_COLOR = "#f2f2f2"
BUTTON_ACTIVE_COLOR = "#d8ecff"
BUTTON_DISABLED_COLOR = "#dddddd"


@dataclass(frozen=True)
class WorkflowButton:
    key: str
    label: str
    rect: tuple[float, float, float, float]


BUTTONS = (
    WorkflowButton("record", "Record", (0.04, 0.05, 0.11, 0.055)),
    WorkflowButton("voice_yes", "Confirm", (0.16, 0.05, 0.11, 0.055)),
    WorkflowButton("voice_no", "Redo Input", (0.28, 0.05, 0.11, 0.055)),
    WorkflowButton("seg_yes", "Accept Mask", (0.40, 0.05, 0.11, 0.055)),
    WorkflowButton("manual", "Box Select", (0.52, 0.05, 0.11, 0.055)),
    WorkflowButton("retry", "Retry Input", (0.64, 0.05, 0.11, 0.055)),
    WorkflowButton("grasp_yes", "Execute", (0.76, 0.05, 0.10, 0.055)),
    WorkflowButton("grasp_no", "Redetect", (0.87, 0.05, 0.09, 0.055)),
)

InputMode = str


class MatplotlibMotionRuntime(MotionRuntime):
    def __init__(
        self,
        app: "MatplotlibGraspWorkflow",
        stage_callback: Callable[[str, np.ndarray | None], None],
    ) -> None:
        super().__init__(
            is_running=app.is_running,
            sync=app.sync_from_motion,
            stage=stage_callback,
            lock=app.motion_lock,
        )


class MatplotlibGraspWorkflow:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik_controller: PiperIKController,
        voice_pipeline: VoicePipeline | None,
        vision_pipeline: VisionPipeline,
        logs: Any,
        input_mode: InputMode = "voice",
        viewer: Any | None = None,
    ) -> None:
        if input_mode not in {"voice", "text"}:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        self.model = model
        self.data = data
        self.ik_controller = ik_controller
        self.voice_pipeline = voice_pipeline
        self.vision_pipeline = vision_pipeline
        self.logs = logs
        self.input_mode = input_mode
        self.viewer = viewer
        self._ui_thread_id = threading.get_ident()

        self._running = True
        self._busy = False
        self._recording = False
        self._manual_selecting = False
        self._preview_capture: CaptureResult | None = None
        self._capture: CaptureResult | None = None
        self._detection: DetectionResult | None = None
        self._segmentation: SegmentationResult | None = None
        self._grasp: GraspResult | None = None
        self._recognized_text = ""
        self._attempt_ids = count(1)
        self._current_attempt_id: int | None = None
        self._last_refresh = 0.0
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ui_queue: Queue[Callable[[], None]] = Queue()
        self._last_error = ""

        self.fig, ((self.ax_rgb, self.ax_depth), (self.ax_result, self.ax_info)) = plt.subplots(
            2,
            2,
            figsize=(13, 9),
        )
        self.fig.canvas.manager.set_window_title("MuJoCo Grasp Workflow")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.14, wspace=0.08, hspace=0.16)

        self.rgb_artist = None
        self.depth_artist = None
        self.result_artist = None
        self.info_artist = self.ax_info.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            fontsize=11,
            transform=self.ax_info.transAxes,
        )
        self.ax_info.axis("off")
        self._buttons: dict[str, Button] = {}
        self._text_box: TextBox | None = None
        self._button_defs = {button.key: button for button in BUTTONS}
        self._create_buttons()
        self._selector: RectangleSelector | None = None

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._timer = self.fig.canvas.new_timer(interval=REFRESH_INTERVAL_MS)
        self._timer.add_callback(self._on_timer)
        self._timer.start()

        if self.input_mode == "voice":
            self._set_status("Ready. Click Record to start a new round.")
        else:
            self._set_status("Ready. Enter target text and click Use Text.")
        self._update_buttons()

    def run(self) -> None:
        self.logs.event(
            "workflow_start",
            "Matplotlib 主控界面已启动",
            text_log=self._log_path("log"),
            jsonl_log=self._log_path("jsonl"),
            input_mode=self.input_mode,
            viewer_enabled=self.viewer is not None,
        )
        plt.show()

    def _log_path(self, fmt: str) -> str | None:
        paths = getattr(self.logs, "paths", {})
        if not isinstance(paths, dict):
            return None
        path = paths.get(fmt)
        return None if path is None else str(path)

    def is_running(self) -> bool:
        return self._running and plt.fignum_exists(self.fig.number)

    def sync_from_motion(self) -> None:
        if time.time() - self._last_refresh > REFRESH_INTERVAL_MS / 1000.0:
            self._capture_and_post_camera()
        self._sync_viewer()
        if threading.get_ident() == self._ui_thread_id:
            self._drain_ui_queue()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def motion_lock(self):
        if self._viewer_is_running():
            return self.viewer.lock()
        return nullcontext()

    def _viewer_is_running(self) -> bool:
        return self.viewer is not None and bool(self.viewer.is_running())

    def _sync_viewer(self) -> None:
        if self._viewer_is_running():
            self.viewer.sync()

    def _create_buttons(self) -> None:
        for button_def in BUTTONS:
            ax = self.fig.add_axes(button_def.rect)
            button = Button(ax, self._button_label(button_def.key), color=BUTTON_COLOR, hovercolor=BUTTON_ACTIVE_COLOR)
            button.on_clicked(lambda _event, key=button_def.key: self._on_button(key))
            self._buttons[button_def.key] = button

        if self.input_mode == "text":
            text_ax = self.fig.add_axes((0.04, 0.005, 0.67, 0.035))
            self._text_box = TextBox(text_ax, "Target ", initial="")
            self._text_box.on_submit(self._on_text_submitted)

        quit_ax = self.fig.add_axes((0.87, 0.005, 0.09, 0.035))
        quit_button = Button(quit_ax, "Quit", color="#f8dddd", hovercolor="#ffd0d0")
        quit_button.on_clicked(lambda _event: self.stop())
        self._buttons["quit"] = quit_button

    def _button_label(self, key: str) -> str:
        if self.input_mode == "text":
            labels = {
                "record": "Use Text",
                "voice_yes": "Confirm Text",
                "voice_no": "Edit Text",
                "retry": "Re-enter Text",
            }
            if key in labels:
                return labels[key]
        return self._button_defs[key].label

    def _set_button_enabled(self, key: str, enabled: bool) -> None:
        button = self._buttons[key]
        button.eventson = enabled
        button.ax.set_facecolor(BUTTON_COLOR if enabled else BUTTON_DISABLED_COLOR)
        button.label.set_color("black" if enabled else "#777777")

    def _button_is_enabled(self, key: str) -> bool:
        button = self._buttons.get(key)
        return bool(button is not None and button.eventson)

    def _update_buttons(self) -> None:
        with self._lock:
            busy = self._busy
            recording = self._recording
            has_text = bool(self._recognized_text)
            has_segmentation = self._segmentation is not None
            has_grasp = self._grasp is not None

        if self.input_mode == "voice":
            self._buttons["record"].label.set_text("Stop" if recording else self._button_label("record"))
        else:
            self._buttons["record"].label.set_text(self._button_label("record"))
            self._buttons["voice_yes"].label.set_text(self._button_label("voice_yes"))
            self._buttons["voice_no"].label.set_text(self._button_label("voice_no"))
            self._buttons["retry"].label.set_text(self._button_label("retry"))
        enabled = {
            "record": (not busy or recording) if self.input_mode == "voice" else not busy,
            "voice_yes": has_text and not busy and not recording,
            "voice_no": has_text and not busy and not recording,
            "seg_yes": has_segmentation and not busy and not recording,
            "manual": has_text and not busy and not recording,
            "retry": has_text and not busy and not recording,
            "grasp_yes": has_grasp and not busy and not recording,
            "grasp_no": has_grasp and not busy and not recording,
            "quit": True,
        }
        for key, is_enabled in enabled.items():
            self._set_button_enabled(key, is_enabled)

    def _on_button(self, key: str) -> None:
        if not self._button_is_enabled(key):
            return
        if key == "record":
            if self.input_mode == "voice":
                self._toggle_recording()
            else:
                self._use_text_input()
        elif key == "voice_yes":
            self._start_auto_detection()
        elif key == "voice_no":
            self._reset_round("Input rejected. Please enter or record again.", clear_text=True)
        elif key == "seg_yes":
            self._start_anygrasp()
        elif key == "manual":
            self._start_manual_selection()
        elif key == "retry":
            self._reset_round("Retry requested. Please enter or record again.", clear_text=True)
        elif key == "grasp_yes":
            self._start_pick_place()
        elif key == "grasp_no":
            self._restart_detection_after_grasp_reject()

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key in {" ", "space"}:
            self._on_button("record")
        elif key in {"enter", "return"}:
            self._on_button("voice_yes")
        elif key == "r":
            self._on_button("retry")
        elif key == "q":
            self.stop()

    def _on_close(self, _event) -> None:
        self.stop()

    def stop(self) -> None:
        self._running = False
        try:
            if self.voice_pipeline is not None and self._recording:
                self.voice_pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            self.logs.event("recording_stop_on_exit_failed", f"退出时停止录音失败：{exc}", level="warning")
        try:
            if self._viewer_is_running():
                self.viewer.close()
        except Exception as exc:  # noqa: BLE001
            self.logs.event("viewer_close_failed", f"关闭 MuJoCo simulate 窗口失败：{exc}", level="warning")
        self.logs.event("workflow_stop", "用户退出 Matplotlib 主控界面")
        plt.close(self.fig)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        if error:
            self._last_error = message
            self.logs.event("ui_error", message, level="error")
        self.info_artist.set_text(self._info_text(message))
        self.fig.canvas.draw_idle()

    def _post_ui(self, callback: Callable[[], None]) -> None:
        self._ui_queue.put(callback)

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except Empty:
                break
            callback()

    def _info_text(self, message: str) -> str:
        lines = [
            f"Status: {message}",
            f"Target text: {self._recognized_text or 'None'}",
        ]
        if self._current_attempt_id is not None:
            lines.append(f"Attempt: {self._current_attempt_id}")
        if self._detection is not None:
            lines.append(f"Detection boxes: {len(self._detection.boxes_xyxy)}")
            lines.append(f"Detection image: {self._detection.annotated_image_path}")
        if self._segmentation is not None:
            lines.append(f"Mask source: {self._segmentation.source}")
            lines.append(f"Mask points: {len(self._segmentation.points)}")
            lines.append(f"Mask image: {self._segmentation.overlay_image_path}")
        if self._grasp is not None:
            lines.append(f"Grasp score: {self._grasp.best_score:.4f}")
            lines.append(f"Grasp file: {self._grasp.grasp_json_path}")
        if self._last_error:
            lines.append(f"Last error: {self._last_error}")
        return "\n".join(lines)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        with self._lock:
            self._busy = busy
        if message:
            self._set_status(message)
        self._update_buttons()

    def _start_worker(self, target: Callable[[], None], busy_message: str) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._set_status(busy_message)
        self._update_buttons()

        def run_worker() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                self.logs.exception(f"后台任务失败：{exc}", exc=exc, event="worker_failed")
                self._post_ui(lambda exc=exc: self._set_status(f"Task failed: {exc}", error=False))
            finally:
                with self._lock:
                    self._busy = False
                self._post_ui(self._update_buttons)

        self._worker = threading.Thread(target=run_worker, daemon=True, name="mujoco-matplotlib-workflow")
        self._worker.start()

    def _toggle_recording(self) -> None:
        if self.voice_pipeline is None:
            self._set_status("Text input mode is active; recording is unavailable.", error=True)
            return
        if self._busy and not self._recording:
            return
        if not self._recording:
            try:
                self._reset_round("Recording started.", clear_text=True)
                self.voice_pipeline.start()
                self._recording = True
                self.logs.event("voice_record_start", "开始录音")
                self._set_status("Recording. Click Stop when finished.")
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Failed to start recording: {exc}", error=True)
        else:
            self._recording = False
            self._start_worker(self._stop_recording_worker, "Stopping recording and transcribing...")
        self._update_buttons()

    def _stop_recording_worker(self) -> None:
        if self.voice_pipeline is None:
            raise RuntimeError("Voice pipeline is not available in text input mode.")
        text = self.voice_pipeline.stop().strip()
        backend = self.voice_pipeline.last_backend
        self._recognized_text = text
        self.logs.event(
            "voice_recognized",
            f"语音识别完成：{text or '空文本'}",
            text=text,
            backend=backend,
            audio_file=self.voice_pipeline.last_audio_file,
            online_error=self.voice_pipeline.last_online_error,
        )
        if not text:
            self._post_ui(lambda: self._set_status("Transcription is empty. Please record again."))
            return
        self._post_ui(lambda: self._set_status("Confirm the recognized text or redo input."))

    def _on_text_submitted(self, _text: str) -> None:
        if self.input_mode == "text" and not self._busy:
            self._use_text_input()

    def _use_text_input(self) -> None:
        if self.input_mode != "text":
            return
        raw_text = self._text_box.text if self._text_box is not None else ""
        text = raw_text.strip()
        if not text:
            self._set_status("Target text cannot be empty.", error=True)
            return
        self._reset_round("Text input loaded.", clear_text=True)
        self._recognized_text = text
        self.logs.event("text_input_received", f"文字输入完成：{text}", text=text)
        self._set_status("Confirm the target text or edit it.")
        self._update_buttons()

    def _reset_round(self, message: str, *, clear_text: bool = False) -> None:
        self._disable_selector()
        if self._detection is not None:
            self.vision_pipeline.mark_artifact_status(
                self._detection,
                "rejected",
                reason=message,
                attempt_id=self._current_attempt_id,
            )
        if self._segmentation is not None:
            self.vision_pipeline.mark_artifact_status(
                self._segmentation,
                "rejected",
                reason=message,
                attempt_id=self._current_attempt_id,
            )
        if self._grasp is not None:
            self.vision_pipeline.mark_artifact_status(
                self._grasp,
                "rejected",
                reason=message,
                attempt_id=self._current_attempt_id,
            )
        self._capture = None
        self._detection = None
        self._segmentation = None
        self._grasp = None
        self._current_attempt_id = None
        self._last_error = ""
        if clear_text:
            self._recognized_text = ""
        self.ax_result.clear()
        self.ax_result.set_title("Workflow Result")
        self.ax_result.axis("off")
        self.logs.event("round_reset", message, clear_text=clear_text)
        self._set_status(message)
        self._update_buttons()

    def _new_attempt_id(self) -> int:
        self._current_attempt_id = next(self._attempt_ids)
        return self._current_attempt_id

    def _start_auto_detection(self) -> None:
        if not self._recognized_text:
            self._set_status("No confirmed target text is available.", error=True)
            return
        self._start_worker(self._auto_detection_worker, "Running detection and segmentation...")

    def _auto_detection_worker(self) -> None:
        attempt_id = self._new_attempt_id()
        with self.motion_lock():
            apply_home_pose(self.model, self.data)
            capture = self.vision_pipeline.capture_rgbd(self.model, self.data)
        self._sync_viewer()
        detection = self.vision_pipeline.detect(self._recognized_text, capture)
        box_index = self.vision_pipeline.select_best_box_index(detection)
        segmentation = self.vision_pipeline.segment(capture, detection, box_index, source="auto")
        self._capture = capture
        self._detection = detection
        self._segmentation = segmentation
        self._grasp = None
        self.logs.event(
            "auto_segmentation_created",
            "自动检测和分割完成，请确认分割结果。",
            attempt_id=attempt_id,
            text=self._recognized_text,
            selected_box_index=box_index,
            detection_image=detection.annotated_image_path,
            segmentation_overlay=segmentation.overlay_image_path,
        )
        self._post_ui(
            lambda: (
                self._show_segmentation(segmentation),
                self._set_status("Auto segmentation complete. Accept, box select, or retry input."),
            )
        )

    def _show_segmentation(self, segmentation: SegmentationResult) -> None:
        preview = self.vision_pipeline.build_segmentation_preview(segmentation)
        self.ax_result.clear()
        self.ax_result.imshow(preview.overlay)
        box = segmentation.selected_box_xyxy
        self.ax_result.plot(
            [box[0], box[2], box[2], box[0], box[0]],
            [box[1], box[1], box[3], box[3], box[1]],
            color="yellow",
            linewidth=2,
        )
        self.ax_result.set_title("Mask Preview")
        self.ax_result.axis("off")
        self.fig.canvas.draw_idle()

    def _start_manual_selection(self) -> None:
        if not self._recognized_text:
            self._set_status("No target text is available for box selection.", error=True)
            return
        try:
            with self.motion_lock():
                self._capture = self.vision_pipeline.capture_rgbd(self.model, self.data)
            self._sync_viewer()
            self._manual_selecting = True
            self._grasp = None
            self._segmentation = None
            self._detection = None
            self.ax_result.clear()
            self.ax_result.imshow(self._capture.rgb_image)
            self.ax_result.set_title("Drag a box around the target")
            self.ax_result.axis("off")
            self._enable_selector()
            self.logs.event("manual_selection_start", "已截帧，等待用户框选。", text=self._recognized_text)
            self._set_status("Drag a box around the target in the result panel.")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to start box selection: {exc}", error=True)

    def _enable_selector(self) -> None:
        self._disable_selector()
        self._selector = RectangleSelector(
            self.ax_result,
            self._on_box_selected,
            useblit=True,
            button=[1],
            minspanx=8,
            minspany=8,
            spancoords="pixels",
            interactive=True,
        )

    def _disable_selector(self) -> None:
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector.disconnect_events()
            self._selector = None
        self._manual_selecting = False

    def _on_box_selected(self, eclick, erelease) -> None:
        if self._capture is None or eclick.xdata is None or erelease.xdata is None:
            self._set_status("Invalid box. Please drag again.")
            return
        box = np.array([eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata], dtype=np.float64)
        self._disable_selector()
        self._start_worker(lambda: self._manual_segment_worker(box), "Segmenting from the selected box...")

    def _manual_segment_worker(self, box: np.ndarray) -> None:
        if self._capture is None:
            raise RuntimeError("Manual capture is missing.")
        attempt_id = self._new_attempt_id()
        segmentation = self.vision_pipeline.segment_manual_box(self._capture, self._recognized_text, box)
        self._segmentation = segmentation
        self._detection = None
        self._grasp = None
        self.logs.event(
            "manual_segmentation_created",
            "手动框选分割完成，请确认分割结果。",
            attempt_id=attempt_id,
            text=self._recognized_text,
            box_xyxy=segmentation.selected_box_xyxy.tolist(),
            segmentation_overlay=segmentation.overlay_image_path,
        )
        self._post_ui(
            lambda: (
                self._show_segmentation(segmentation),
                self._set_status("Manual segmentation complete. Accept, reselect, or retry input."),
            )
        )

    def _start_anygrasp(self) -> None:
        if self._segmentation is None:
            self._set_status("No mask result is available to accept.", error=True)
            return
        self._start_worker(self._anygrasp_worker, "Running AnyGrasp...")

    def _anygrasp_worker(self) -> None:
        if self._segmentation is None:
            raise RuntimeError("Segmentation result is missing.")
        if self._detection is not None:
            self.vision_pipeline.mark_artifact_status(
                self._detection,
                "accepted",
                attempt_id=self._current_attempt_id,
                text=self._recognized_text,
            )
        self.vision_pipeline.mark_artifact_status(
            self._segmentation,
            "accepted",
            attempt_id=self._current_attempt_id,
            text=self._recognized_text,
        )
        grasp = self.vision_pipeline.run_anygrasp(self._segmentation)
        self._grasp = grasp
        self.logs.event(
            "grasp_created",
            "AnyGrasp 已输出候选抓取，请确认。",
            attempt_id=self._current_attempt_id,
            text=self._recognized_text,
            best_score=grasp.best_score,
            grasp_json=grasp.grasp_json_path,
            visualization=grasp.visualization_path,
        )
        self._post_ui(
            lambda: (
                self._show_grasp(grasp),
                self._set_status("AnyGrasp complete. Execute or redetect."),
            )
        )

    def _show_grasp(self, grasp: GraspResult) -> None:
        preview = self.vision_pipeline.build_grasp_preview(grasp)
        self.ax_result.clear()
        self.ax_result.imshow(preview.projection)
        self.ax_result.set_title("AnyGrasp Preview")
        self.ax_result.axis("off")
        self.info_artist.set_text(self._info_text("AnyGrasp complete.\n" + preview.summary_text))
        self.fig.canvas.draw_idle()

    def _restart_detection_after_grasp_reject(self) -> None:
        if self._grasp is not None:
            self.vision_pipeline.mark_artifact_status(
                self._grasp,
                "rejected",
                attempt_id=self._current_attempt_id,
                reason="用户否决 AnyGrasp 结果，使用同一语音文本重新检测。",
            )
        self._grasp = None
        self._segmentation = None
        self._detection = None
        self.logs.event(
            "grasp_rejected_redetect",
            "用户否决抓取结果，使用同一语音文本重新检测。",
            text=self._recognized_text,
        )
        self._start_auto_detection()

    def _start_pick_place(self) -> None:
        if self._grasp is None:
            self._set_status("No grasp result is available to execute.", error=True)
            return
        self._start_worker(self._pick_place_worker, "Executing pick-and-place...")

    def _pick_place_worker(self) -> None:
        if self._grasp is None:
            raise RuntimeError("Grasp result is missing.")
        self.vision_pipeline.mark_artifact_status(
            self._grasp,
            "accepted",
            attempt_id=self._current_attempt_id,
            text=self._recognized_text,
        )

        def stage_callback(stage: str, position: np.ndarray | None = None) -> None:
            pos_text = "" if position is None else f"，目标位置={np.round(position, 4).tolist()}"
            self.logs.event(
                "motion_stage",
                f"机械臂阶段：{stage}{pos_text}",
                stage=stage,
                target_position=None if position is None else np.asarray(position).tolist(),
            )
            self._post_ui(lambda stage=stage: self._set_status(f"Robot stage: {stage}"))

        runtime = MatplotlibMotionRuntime(self, stage_callback)
        ok = execute_pick_place_sequence_with_runtime(
            self.model,
            self.data,
            runtime,
            self.ik_controller,
            self._grasp,
        )
        if not ok:
            self.logs.event("motion_interrupted", "抓取放置动作被中断。", level="warning")
            self._post_ui(lambda: self._set_status("Pick-and-place was interrupted."))
            return
        self.logs.event(
            "motion_completed",
            "抓取放置完成，机械臂已回到 home，准备下一轮。",
            text=self._recognized_text,
            attempt_id=self._current_attempt_id,
        )
        self._recognized_text = ""
        self._capture = None
        self._detection = None
        self._segmentation = None
        self._grasp = None
        self._current_attempt_id = None
        self._post_ui(lambda: self._set_status("Pick-and-place complete. Ready for the next round."))

    def _on_timer(self) -> bool:
        if not self.is_running():
            return False
        self._drain_ui_queue()
        if not self._busy and not self._manual_selecting:
            self._refresh_camera()
        self._update_buttons()
        return True

    def _refresh_camera(self) -> None:
        try:
            with self.motion_lock():
                mujoco.mj_step(self.model, self.data)
                capture = self.vision_pipeline.capture_rgbd(self.model, self.data)
            self._sync_viewer()
            self._display_camera_capture(capture)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logs.event("camera_refresh_failed", f"相机刷新失败：{exc}", level="warning")

    def _capture_and_post_camera(self) -> None:
        try:
            with self.motion_lock():
                capture = self.vision_pipeline.capture_rgbd(self.model, self.data)
            self._post_ui(lambda capture=capture: self._display_camera_capture(capture))
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logs.event("camera_refresh_failed", f"运动同步相机刷新失败：{exc}", level="warning")

    def _display_camera_capture(self, capture: CaptureResult) -> None:
        self._preview_capture = capture
        self._last_refresh = time.time()
        depth_preview = self.vision_pipeline.build_depth_preview(capture.depth_meters)

        if self.rgb_artist is None:
            self.rgb_artist = self.ax_rgb.imshow(capture.rgb_image)
            self.ax_rgb.set_title("RGB Camera")
            self.ax_rgb.axis("off")
        else:
            self.rgb_artist.set_data(capture.rgb_image)

        if self.depth_artist is None:
            self.depth_artist = self.ax_depth.imshow(depth_preview)
            self.ax_depth.set_title("Depth Camera")
            self.ax_depth.axis("off")
        else:
            self.depth_artist.set_data(depth_preview)
        self.fig.canvas.draw_idle()
