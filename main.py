from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import mujoco.viewer

MODULE_DIR = Path(__file__).resolve().parent


from src.grasp_sequence import HOME_CTRL, apply_home_pose
from src.ik.piper_ik import PiperIKController
from src.Vision.matplotlib_workflow import MatplotlibGraspWorkflow
from src.NoPrint.noprint import create_logger
from src.Vision import VisionPipeline, VisionPipelineConfig
from src.Voice.voice_pipeline import VoicePipeline


DEFAULT_SCENE_XML = MODULE_DIR / "piper_d435i" / "scene.xml"


def main() -> None:
    scene_xml = DEFAULT_SCENE_XML
    logs = create_logger(
        MODULE_DIR / "temp" / "log",
        formats=("log", "jsonl"),
        name=Path(__file__).stem,
        capture_exceptions=True,
    )
    voice_pipeline: VoicePipeline | None = None
    vision_pipeline: VisionPipeline | None = None
    viewer = None

    try:
        logs.event("program_start", "程序启动，正在加载 MuJoCo 模型。", scene_xml=str(scene_xml))

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        apply_home_pose(model, data)
        logs.event("home_pose_applied", "机械臂已初始化到 home 位置并打开夹爪。", home_ctrl=HOME_CTRL.tolist())

        ik_controller = PiperIKController(model, data)
        voice_pipeline = VoicePipeline()
        viewer = mujoco.viewer.launch_passive(model, data)
        logs.event("viewer_start", "MuJoCo simulate 窗口已启动。")
        vision_pipeline = VisionPipeline(
            VisionPipelineConfig(scene_xml_path=str(scene_xml))
        )
        app = MatplotlibGraspWorkflow(
            model=model,
            data=data,
            ik_controller=ik_controller,
            voice_pipeline=voice_pipeline,
            vision_pipeline=vision_pipeline,
            logs=logs,
            viewer=viewer,
        )
        app.run()
    except Exception as exc:  # noqa: BLE001
        logs.exception(f"程序异常退出：{exc}", exc=exc, event="program_failed", error=str(exc))
        raise
    finally:
        try:
            try:
                if viewer is not None and viewer.is_running():
                    viewer.close()
            except Exception as exc:  # noqa: BLE001
                logs.event("viewer_close_failed", f"关闭 MuJoCo simulate 窗口失败：{exc}", level="warning")
            try:
                if vision_pipeline is not None:
                    vision_pipeline.shutdown()
            except Exception as exc:  # noqa: BLE001
                logs.event("vision_shutdown_failed", f"关闭视觉流程失败：{exc}", level="warning")
            try:
                if voice_pipeline is not None:
                    voice_pipeline.shutdown()
            except Exception as exc:  # noqa: BLE001
                logs.event("voice_shutdown_failed", f"关闭语音流程失败：{exc}", level="warning")
            logs.event("program_shutdown", "程序资源已释放。")
        finally:
            logs.close()


if __name__ == "__main__":
    main()
