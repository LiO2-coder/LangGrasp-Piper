from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import glfw
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial.transform import Rotation
from ultralytics import FastSAM, YOLOWorld


MODULE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_XML = MODULE_DIR / "piper_d435i" / "scene.xml"
DEFAULT_CAMERA_CONFIG = MODULE_DIR / "config" / "d435i_camera_params.yaml"
DEFAULT_YOLO_MODEL = MODULE_DIR / "model" / "yolov8x-worldv2.pt"
DEFAULT_FASTSAM_MODEL = MODULE_DIR / "model" / "FastSAM-x.pt"
DEFAULT_ANYGRASP_ROOT = MODULE_DIR.parent / "anygrasp_sdk" / "grasp_detection"
DEFAULT_ANYGRASP_CHECKPOINT = DEFAULT_ANYGRASP_ROOT / "log" / "checkpoint_detection.tar"
DEFAULT_TEMP_IMAGE_DIR = MODULE_DIR / "temp" / "image"
DEFAULT_TEMP_GRASP_DIR = MODULE_DIR / "temp" / "grasp"
MIN_MANUAL_BOX_SIZE_PX = 8
POINTCLOUD_PREVIEW_TRANSFORM_LABEL = "x,y,-z"
POINTCLOUD_PREVIEW_TRANSFORM = np.diag([1.0, 1.0, -1.0, 1.0])

ANYGRASP_ROOT = DEFAULT_ANYGRASP_ROOT
if str(ANYGRASP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYGRASP_ROOT))


@dataclass(frozen=True)
class VisionPipelineConfig:
    scene_xml_path: str = str(DEFAULT_SCENE_XML)
    camera_config_path: str = str(DEFAULT_CAMERA_CONFIG)
    rgb_camera_name: str = "rgb_camera"
    depth_camera_name: str = "depth_camera"
    yolo_model_path: str = str(DEFAULT_YOLO_MODEL)
    fastsam_model_path: str = str(DEFAULT_FASTSAM_MODEL)
    anygrasp_checkpoint_path: str = str(DEFAULT_ANYGRASP_CHECKPOINT)
    temp_image_dir: str = str(DEFAULT_TEMP_IMAGE_DIR)
    temp_grasp_dir: str = str(DEFAULT_TEMP_GRASP_DIR)
    depth_min_m: float = 0.0
    depth_max_m: float = 2.0
    anygrasp_max_gripper_width: float = 0.1
    anygrasp_gripper_height: float = 0.03
    anygrasp_top_down_grasp: bool = False
    anygrasp_debug: bool = False
    anygrasp_dense_grasp: bool = False
    anygrasp_collision_detection: bool = True
    anygrasp_apply_object_mask: bool = True


@dataclass
class CaptureResult:
    rgb_image: np.ndarray
    depth_buffer: np.ndarray
    depth_meters: np.ndarray
    rgb_camera_name: str
    depth_camera_name: str
    rgb_intrinsics: np.ndarray
    depth_intrinsics: np.ndarray
    rgb_resolution: tuple[int, int]
    depth_resolution: tuple[int, int]
    timestamp: str
    depth_to_rgb_optical: Optional[np.ndarray] = None


@dataclass
class DetectionResult:
    text: str
    timestamp: str
    prompt: str
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_names: list[str]
    annotated_image_path: str


@dataclass
class SegmentationResult:
    text: str
    timestamp: str
    selected_box_index: int
    selected_box_xyxy: np.ndarray
    mask_bool: np.ndarray
    overlay_image_path: str
    mask_image_path: str
    depth_image_path: str
    pointcloud_preview_path: str
    points: np.ndarray
    colors: np.ndarray
    point_cloud: o3d.geometry.PointCloud
    source: str = "auto"


@dataclass
class GraspResult:
    text: str
    timestamp: str
    best_translation: np.ndarray
    best_rotation_matrix: np.ndarray
    best_quaternion_wxyz: np.ndarray
    best_width: float
    best_score: float
    all_grasps_summary: list[dict[str, Any]]
    point_cloud_path: str
    grasp_json_path: str
    all_grasps_json_path: str
    visualization_path: Optional[str]
    point_cloud: o3d.geometry.PointCloud
    gripper_geometries: list[o3d.geometry.Geometry]


@dataclass
class SegmentationPreview:
    overlay: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    pointcloud: np.ndarray


@dataclass
class GraspPreview:
    projection: np.ndarray
    summary_text: str


class VisionPipeline:
    def __init__(self, config: VisionPipelineConfig | None = None) -> None:
        self.config = config or VisionPipelineConfig()
        self._detector = YOLOWorld(self.config.yolo_model_path)
        self._segmenter = FastSAM(self.config.fastsam_model_path)
        self._camera_config = self.load_camera_config(self.config.camera_config_path)
        self._glfw_window = None
        self._render_context = None
        self._render_scene = None
        self._render_model_id = None
        self._render_lock = threading.RLock()
        self._anygrasp = self._build_anygrasp()

        Path(self.config.temp_image_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_grasp_dir).mkdir(parents=True, exist_ok=True)

    def _build_anygrasp(self):
        checkpoint = Path(self.config.anygrasp_checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"AnyGrasp checkpoint not found: {checkpoint}. "
                "Please make sure grasp_detection/log/checkpoint_detection.tar exists."
            )
        from gsnet import AnyGrasp  # noqa: PLC0415

        class _Cfg:
            def __init__(self, cfg: VisionPipelineConfig) -> None:
                self.checkpoint_path = str(checkpoint)
                self.max_gripper_width = max(0.0, min(0.1, cfg.anygrasp_max_gripper_width))
                self.gripper_height = cfg.anygrasp_gripper_height
                self.top_down_grasp = cfg.anygrasp_top_down_grasp
                self.debug = cfg.anygrasp_debug

        anygrasp = AnyGrasp(_Cfg(self.config))
        anygrasp.load_net()
        return anygrasp

    @staticmethod
    def _safe_stem(text: str) -> str:
        cleaned = re.sub(r"\s+", "_", text.strip())
        cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "", cleaned)
        return cleaned or "unknown"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): VisionPipeline._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [VisionPipeline._json_ready(item) for item in value]
        return value

    @classmethod
    def write_metadata(cls, path: str | Path, **fields: Any) -> str:
        metadata_path = Path(path).with_suffix(Path(path).suffix + ".meta.json")
        metadata_path.write_text(
            json.dumps(cls._json_ready(fields), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(metadata_path)

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
    def preview_transform_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return points.reshape((-1, 3)).copy()
        reshaped = points.reshape((-1, 3))
        homogeneous = np.concatenate([reshaped, np.ones((reshaped.shape[0], 1), dtype=np.float64)], axis=1)
        transformed = homogeneous @ POINTCLOUD_PREVIEW_TRANSFORM.T
        transformed = transformed[:, :3]
        return transformed.reshape(points.shape)

    @staticmethod
    def _invert_transform(transform: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        inv = np.eye(4, dtype=np.float64)
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        inv[:3, :3] = rot.T
        inv[:3, 3] = -rot.T @ trans
        return inv

    @staticmethod
    def load_camera_config(path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def normalize_camera(name: str):
            cfg = raw[name]
            cam_name = cfg.get("camera_name", name)
            resolution = tuple(cfg["resolution"])
            fov_deg = tuple(cfg["fov_deg"])
            k = np.asarray(cfg["K"], dtype=np.float64)
            t = np.asarray(cfg["T_body_to_camera"], dtype=np.float64)
            return {
                "camera_name": cam_name,
                "resolution": resolution,
                "fov_deg": fov_deg,
                "K": k,
                "T_body_to_camera": t,
            }

        relations = raw.get("camera_relations", {}) or {}
        for key, value in list(relations.items()):
            relations[key] = np.asarray(value, dtype=np.float64)

        return {
            "rgb": normalize_camera("rgb_camera"),
            "depth": normalize_camera("depth_camera"),
            "relations": relations,
        }

    def _depth_to_rgb_optical(self) -> np.ndarray:
        relations = self._camera_config.get("relations", {})
        if "T_depth_camera_to_rgb_camera" in relations:
            return np.asarray(relations["T_depth_camera_to_rgb_camera"], dtype=np.float64)
        if "T_depth_to_rgb" in relations:
            return np.asarray(relations["T_depth_to_rgb"], dtype=np.float64)
        if "T_rgb_camera_to_depth_camera" in relations:
            return self._invert_transform(relations["T_rgb_camera_to_depth_camera"])
        if "T_rgb_to_depth" in relations:
            return self._invert_transform(relations["T_rgb_to_depth"])

        rgb_t = np.asarray(self._camera_config["rgb"]["T_body_to_camera"], dtype=np.float64)
        depth_t = np.asarray(self._camera_config["depth"]["T_body_to_camera"], dtype=np.float64)
        return self._invert_transform(rgb_t) @ depth_t

    @staticmethod
    def build_camera(model: mujoco.MjModel, cam_name: str) -> mujoco.MjvCamera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in XML.")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        return cam

    @staticmethod
    def depth_buffer_to_meters(model: mujoco.MjModel, depth_buffer: np.ndarray) -> np.ndarray:
        near = model.vis.map.znear * model.stat.extent
        far = model.vis.map.zfar * model.stat.extent
        return near / (1.0 - depth_buffer * (1.0 - near / far))

    def _ensure_renderer(self, model: mujoco.MjModel) -> tuple[mujoco.MjvScene, mujoco.MjrContext]:
        if self._render_model_id == id(model) and self._render_scene is not None and self._render_context is not None:
            return self._render_scene, self._render_context

        model.vis.global_.offwidth = max(
            self._camera_config["rgb"]["resolution"][0],
            self._camera_config["depth"]["resolution"][0],
        )
        model.vis.global_.offheight = max(
            self._camera_config["rgb"]["resolution"][1],
            self._camera_config["depth"]["resolution"][1],
        )

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._glfw_window = glfw.create_window(1200, 900, "mujoco-vision", None, None)
        if self._glfw_window is None:
            raise RuntimeError("Failed to create GLFW window.")
        glfw.make_context_current(self._glfw_window)

        self._render_scene = mujoco.MjvScene(model, maxgeom=10000)
        self._render_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._render_context)
        self._render_model_id = id(model)
        return self._render_scene, self._render_context

    def render_camera(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: mujoco.MjvCamera,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._render_lock:
            scene, context = self._ensure_renderer(model)
            if self._glfw_window is not None:
                glfw.make_context_current(self._glfw_window)
            try:
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    model,
                    data,
                    mujoco.MjvOption(),
                    None,
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    scene,
                )
                mujoco.mjr_render(viewport, scene, context)

                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                depth = np.zeros((height, width), dtype=np.float32)
                mujoco.mjr_readPixels(rgb, depth, viewport, context)
                return np.flipud(rgb), np.flipud(depth)
            finally:
                glfw.make_context_current(None)

    def capture_rgbd(self, model: mujoco.MjModel, data: mujoco.MjData) -> CaptureResult:
        mujoco.mj_forward(model, data)
        rgb_cfg = self._camera_config["rgb"]
        depth_cfg = self._camera_config["depth"]

        rgb_camera = self.build_camera(model, self.config.rgb_camera_name)
        depth_camera = self.build_camera(model, self.config.depth_camera_name)

        rgb_w, rgb_h = rgb_cfg["resolution"]
        depth_w, depth_h = depth_cfg["resolution"]

        rgb_image, _ = self.render_camera(model, data, rgb_camera, rgb_w, rgb_h)
        _, depth_buffer = self.render_camera(model, data, depth_camera, depth_w, depth_h)
        depth_meters = self.depth_buffer_to_meters(model, depth_buffer)

        return CaptureResult(
            rgb_image=rgb_image,
            depth_buffer=depth_buffer,
            depth_meters=depth_meters,
            rgb_camera_name=self.config.rgb_camera_name,
            depth_camera_name=self.config.depth_camera_name,
            rgb_intrinsics=np.asarray(rgb_cfg["K"], dtype=np.float64),
            depth_intrinsics=np.asarray(depth_cfg["K"], dtype=np.float64),
            rgb_resolution=(rgb_w, rgb_h),
            depth_resolution=(depth_w, depth_h),
            timestamp=self._timestamp(),
            depth_to_rgb_optical=self._depth_to_rgb_optical(),
        )

    def detect(self, text: str, capture_result: CaptureResult) -> DetectionResult:
        prompt = text.strip()
        if not prompt:
            raise ValueError("YOLOWorld prompt text is empty.")

        self._detector.set_classes([prompt])
        results = self._detector.predict(capture_result.rgb_image, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise RuntimeError(f"YOLOWorld did not detect any box for prompt '{prompt}'.")

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float64)
        confidences = results[0].boxes.conf.cpu().numpy().astype(np.float64)
        annotated_bgr = results[0].plot()

        stem = self._safe_stem(prompt)
        image_path = Path(self.config.temp_image_dir) / f"{stem}_yoloworld_{capture_result.timestamp}.png"
        cv2.imwrite(str(image_path), annotated_bgr)

        class_names = [prompt for _ in range(len(boxes))]
        result = DetectionResult(
            text=text,
            timestamp=capture_result.timestamp,
            prompt=prompt,
            boxes_xyxy=boxes,
            confidences=confidences,
            class_names=class_names,
            annotated_image_path=str(image_path),
        )
        self.write_metadata(
            image_path,
            kind="detection",
            status="created",
            text=text,
            timestamp=capture_result.timestamp,
            prompt=prompt,
            boxes_xyxy=boxes,
            confidences=confidences,
            class_names=class_names,
        )
        return result

    @staticmethod
    def select_best_box_index(detection_result: DetectionResult) -> int:
        if len(detection_result.boxes_xyxy) == 0:
            raise ValueError("Detection result does not contain any boxes.")
        if len(detection_result.boxes_xyxy) == 1:
            return 0
        return int(np.argmax(detection_result.confidences))

    def _select_box_index(self, detection_result: DetectionResult) -> int:
        if len(detection_result.boxes_xyxy) == 1:
            return 0
        self._print_info("YOLOWorld returned multiple boxes:")
        for idx, (box, score) in enumerate(zip(detection_result.boxes_xyxy, detection_result.confidences)):
            self._print_info(
                f"  [{idx}] score={score:.4f} box={np.round(box, 1).tolist()} class={detection_result.class_names[idx]}"
            )

        while True:
            raw = input("Select box index to continue: ").strip()
            try:
                index = int(raw)
            except ValueError:
                self._print_warn("Please enter a valid integer box index.")
                continue
            if 0 <= index < len(detection_result.boxes_xyxy):
                return index
            self._print_warn("Selected index is out of range.")

    def _depth_mask_preview(self, depth_meters: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        if mask_bool.shape != depth_meters.shape:
            raise ValueError(
                f"Depth mask shape {mask_bool.shape} does not match depth shape {depth_meters.shape}."
            )
        clipped = np.clip(depth_meters, self.config.depth_min_m, self.config.depth_max_m)
        depth_vis = ((clipped - self.config.depth_min_m) / max(1e-6, self.config.depth_max_m - self.config.depth_min_m) * 255.0).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_vis[~mask_bool] = 0
        return depth_vis

    @staticmethod
    def _resize_mask(mask_bool: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        if mask_bool.shape == (height, width):
            return mask_bool
        mask_u8 = mask_bool.astype(np.uint8) * 255
        resized = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
        return resized > 127

    @staticmethod
    def validate_box_xyxy(box_xyxy: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
        height, width = image_shape[:2]
        x1, y1, x2, y2 = box
        left = float(np.clip(min(x1, x2), 0, width - 1))
        right = float(np.clip(max(x1, x2), 0, width - 1))
        top = float(np.clip(min(y1, y2), 0, height - 1))
        bottom = float(np.clip(max(y1, y2), 0, height - 1))
        if right - left < MIN_MANUAL_BOX_SIZE_PX or bottom - top < MIN_MANUAL_BOX_SIZE_PX:
            raise ValueError(
                f"Manual box is too small: {[left, top, right, bottom]}. "
                f"Minimum size is {MIN_MANUAL_BOX_SIZE_PX}px."
            )
        return np.array([left, top, right, bottom], dtype=np.float64)

    @staticmethod
    def _make_detection_from_box(
        text: str,
        timestamp: str,
        box_xyxy: np.ndarray,
        source: str,
    ) -> DetectionResult:
        prompt = text.strip() or "manual"
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(1, 4)
        return DetectionResult(
            text=text,
            timestamp=timestamp,
            prompt=prompt,
            boxes_xyxy=box,
            confidences=np.array([1.0], dtype=np.float64),
            class_names=[source],
            annotated_image_path="",
        )

    def _mask_to_point_cloud(
        self,
        capture_result: CaptureResult,
        mask_bool: np.ndarray,
        rgb_lookup_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
        rgb = capture_result.rgb_image
        depth = capture_result.depth_meters
        if mask_bool.shape != depth.shape:
            raise ValueError(
                f"Point-cloud mask shape {mask_bool.shape} does not match depth shape {depth.shape}."
            )
        fx = capture_result.depth_intrinsics[0, 0]
        fy = capture_result.depth_intrinsics[1, 1]
        cx = capture_result.depth_intrinsics[0, 2]
        cy = capture_result.depth_intrinsics[1, 2]

        rgb_resized = cv2.resize(rgb, capture_result.depth_resolution, interpolation=cv2.INTER_LINEAR)
        h, w = depth.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (x_coords - cx) / fx * z
        y = (y_coords - cy) / fy * z

        valid = mask_bool & np.isfinite(z) & (z > 0) & (z < self.config.depth_max_m)
        if not np.any(valid):
            raise RuntimeError("FastSAM mask did not produce any valid depth points.")

        points = np.stack([x, y, z], axis=-1)[valid].astype(np.float32)
        if rgb_lookup_xy is None:
            colors = rgb_resized[valid].astype(np.float32) / 255.0
        else:
            lookup = np.rint(rgb_lookup_xy).astype(np.int32)
            rgb_h, rgb_w = rgb.shape[:2]
            lookup_x = np.clip(lookup[..., 0], 0, rgb_w - 1)
            lookup_y = np.clip(lookup[..., 1], 0, rgb_h - 1)
            colors = rgb[lookup_y[valid], lookup_x[valid]].astype(np.float32) / 255.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return points, colors, point_cloud

    @staticmethod
    def _project_depth_pixels_to_rgb(
        depth_meters: np.ndarray,
        depth_intrinsics: np.ndarray,
        rgb_intrinsics: np.ndarray,
        depth_to_rgb_optical: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        depth = np.asarray(depth_meters, dtype=np.float64)
        h, w = depth.shape
        fx_d = float(depth_intrinsics[0, 0])
        fy_d = float(depth_intrinsics[1, 1])
        cx_d = float(depth_intrinsics[0, 2])
        cy_d = float(depth_intrinsics[1, 2])
        fx_rgb = float(rgb_intrinsics[0, 0])
        fy_rgb = float(rgb_intrinsics[1, 1])
        cx_rgb = float(rgb_intrinsics[0, 2])
        cy_rgb = float(rgb_intrinsics[1, 2])

        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        z_d = depth
        x_d = (x_coords - cx_d) / fx_d * z_d
        y_d = (y_coords - cy_d) / fy_d * z_d
        points_d = np.stack([x_d, y_d, z_d, np.ones_like(z_d)], axis=-1)

        transform = np.eye(4, dtype=np.float64) if depth_to_rgb_optical is None else np.asarray(
            depth_to_rgb_optical,
            dtype=np.float64,
        ).reshape(4, 4)
        points_rgb = points_d @ transform.T
        z_rgb = points_rgb[..., 2]
        valid = np.isfinite(z_rgb) & (z_rgb > 1e-6) & np.isfinite(z_d) & (z_d > 0)

        x_over_z = np.divide(
            points_rgb[..., 0],
            z_rgb,
            out=np.full_like(z_rgb, np.nan),
            where=np.abs(z_rgb) > 1e-6,
        )
        y_over_z = np.divide(
            points_rgb[..., 1],
            z_rgb,
            out=np.full_like(z_rgb, np.nan),
            where=np.abs(z_rgb) > 1e-6,
        )
        u_rgb = fx_rgb * x_over_z + cx_rgb
        v_rgb = fy_rgb * y_over_z + cy_rgb
        lookup_xy = np.stack([u_rgb, v_rgb], axis=-1)
        return lookup_xy, valid

    def _rgb_mask_to_depth_mask(
        self,
        capture_result: CaptureResult,
        rgb_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb_h, rgb_w = rgb_mask.shape
        lookup_xy, projection_valid = self._project_depth_pixels_to_rgb(
            capture_result.depth_meters,
            capture_result.depth_intrinsics,
            capture_result.rgb_intrinsics,
            capture_result.depth_to_rgb_optical,
        )
        lookup = np.rint(np.nan_to_num(lookup_xy, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32)
        inside = (
            projection_valid
            & (lookup[..., 0] >= 0)
            & (lookup[..., 0] < rgb_w)
            & (lookup[..., 1] >= 0)
            & (lookup[..., 1] < rgb_h)
        )
        depth_mask = np.zeros(capture_result.depth_meters.shape, dtype=bool)
        depth_mask[inside] = rgb_mask[lookup[..., 1][inside], lookup[..., 0][inside]]
        return depth_mask, lookup_xy

    def segment(
        self,
        capture_result: CaptureResult,
        detection_result: DetectionResult,
        selected_box_index: int,
        *,
        source: str = "auto",
    ) -> SegmentationResult:
        if not (0 <= selected_box_index < len(detection_result.boxes_xyxy)):
            raise ValueError("selected_box_index is out of range.")

        box = detection_result.boxes_xyxy[selected_box_index].astype(np.float32)
        segment_results = self._segmenter.predict(
            capture_result.rgb_image,
            bboxes=[box.tolist()],
            verbose=False,
        )
        if not segment_results or segment_results[0].masks is None or len(segment_results[0].masks.data) == 0:
            raise RuntimeError("FastSAM did not return any mask for the selected box.")

        mask_data = segment_results[0].masks.data[0].cpu().numpy()
        mask_raw = mask_data > 0.5
        rgb_h, rgb_w = capture_result.rgb_image.shape[:2]
        rgb_mask = self._resize_mask(mask_raw, (rgb_w, rgb_h))
        depth_mask, rgb_lookup_xy = self._rgb_mask_to_depth_mask(capture_result, rgb_mask)

        overlay_rgb = capture_result.rgb_image.copy()
        overlay_rgb[rgb_mask] = (
            0.6 * overlay_rgb[rgb_mask] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)
        mask_img = (rgb_mask.astype(np.uint8) * 255)
        depth_preview = self._depth_mask_preview(capture_result.depth_meters, depth_mask)

        points, colors, point_cloud = self._mask_to_point_cloud(
            capture_result,
            depth_mask,
            rgb_lookup_xy,
        )

        source_tag = self._safe_stem(source)
        pointcloud_preview_path = Path(self.config.temp_image_dir) / (
            f"{self._safe_stem(detection_result.text)}_fastsam_{source_tag}_{capture_result.timestamp}_pointcloud.png"
        )
        self._save_pointcloud_projection(point_cloud, pointcloud_preview_path)

        overlay_path = Path(self.config.temp_image_dir) / (
            f"{self._safe_stem(detection_result.text)}_fastsam_{source_tag}_{capture_result.timestamp}_overlay.png"
        )
        mask_path = Path(self.config.temp_image_dir) / (
            f"{self._safe_stem(detection_result.text)}_fastsam_{source_tag}_{capture_result.timestamp}_mask.png"
        )
        depth_path = Path(self.config.temp_image_dir) / (
            f"{self._safe_stem(detection_result.text)}_fastsam_{source_tag}_{capture_result.timestamp}_depth.png"
        )

        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask_img)
        cv2.imwrite(str(depth_path), depth_preview)

        result = SegmentationResult(
            text=detection_result.text,
            timestamp=capture_result.timestamp,
            selected_box_index=selected_box_index,
            selected_box_xyxy=box.astype(np.float64),
            mask_bool=depth_mask,
            overlay_image_path=str(overlay_path),
            mask_image_path=str(mask_path),
            depth_image_path=str(depth_path),
            pointcloud_preview_path=str(pointcloud_preview_path),
            points=points,
            colors=colors,
            point_cloud=point_cloud,
            source=source,
        )
        for path in (overlay_path, mask_path, depth_path, pointcloud_preview_path):
            self.write_metadata(
                path,
                kind="segmentation",
                status="created",
                source=source,
                text=detection_result.text,
                timestamp=capture_result.timestamp,
                selected_box_index=selected_box_index,
                selected_box_xyxy=box.astype(np.float64),
                points_count=int(len(points)),
                preview_transform=POINTCLOUD_PREVIEW_TRANSFORM_LABEL,
                rendered_grippers=0,
            )
        return result

    def segment_manual_box(
        self,
        capture_result: CaptureResult,
        text: str,
        box_xyxy: np.ndarray,
    ) -> SegmentationResult:
        box = self.validate_box_xyxy(box_xyxy, capture_result.rgb_image.shape)
        detection = self._make_detection_from_box(text, capture_result.timestamp, box, "manual_box")
        return self.segment(capture_result, detection, 0, source="manual")

    def _save_pointcloud_projection(
        self,
        point_cloud: o3d.geometry.PointCloud,
        output_path: Path,
        *,
        gripper_geometries: list[o3d.geometry.Geometry] | None = None,
        title: str = "Segmented Point Cloud",
    ) -> int:
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        if points.size == 0:
            raise RuntimeError("Cannot save projection for an empty point cloud.")
        display_points = self.preview_transform_points(points)
        if colors.shape[0] != display_points.shape[0]:
            colors = np.tile(np.array([[0.25, 0.55, 0.95]], dtype=np.float64), (display_points.shape[0], 1))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2], c=colors, s=1)
        gripper_count = self._draw_gripper_geometries(ax, gripper_geometries or [])
        self._set_equal_3d_axes(ax, display_points, gripper_geometries or [])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=24, azim=-62)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return gripper_count

    def _draw_gripper_geometries(self, ax, gripper_geometries: list[o3d.geometry.Geometry]) -> int:
        rendered = 0
        for geometry in gripper_geometries:
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                vertices = np.asarray(geometry.vertices)
                triangles = np.asarray(geometry.triangles)
                if vertices.size == 0 or triangles.size == 0:
                    continue
                display_vertices = self.preview_transform_points(vertices)
                faces = display_vertices[triangles]
                mesh = Poly3DCollection(
                    faces,
                    facecolors=(1.0, 0.72, 0.05, 0.42),
                    edgecolors=(1.0, 0.52, 0.0, 0.95),
                    linewidths=0.45,
                )
                ax.add_collection3d(mesh)
                rendered += 1
            elif isinstance(geometry, o3d.geometry.LineSet):
                vertices = np.asarray(geometry.points)
                lines = np.asarray(geometry.lines)
                if vertices.size == 0 or lines.size == 0:
                    continue
                display_vertices = self.preview_transform_points(vertices)
                for start, end in lines:
                    segment = display_vertices[[start, end]]
                    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#ffb000", linewidth=1.0)
                rendered += 1
        return rendered

    def _set_equal_3d_axes(
        self,
        ax,
        display_points: np.ndarray,
        gripper_geometries: list[o3d.geometry.Geometry],
    ) -> None:
        point_sets = [np.asarray(display_points, dtype=np.float64).reshape((-1, 3))]
        for geometry in gripper_geometries:
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                vertices = np.asarray(geometry.vertices)
            elif isinstance(geometry, o3d.geometry.LineSet):
                vertices = np.asarray(geometry.points)
            else:
                continue
            if vertices.size:
                point_sets.append(self.preview_transform_points(vertices))

        all_points = np.concatenate([items for items in point_sets if items.size], axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float(np.max(maxs - mins) / 2.0)
        radius = max(radius, 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    @staticmethod
    def _read_rgb_image(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Preview image not found: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def build_depth_preview(self, depth_meters: np.ndarray) -> np.ndarray:
        clipped = np.clip(depth_meters, self.config.depth_min_m, self.config.depth_max_m)
        depth_u8 = (
            (clipped - self.config.depth_min_m)
            / max(1e-6, self.config.depth_max_m - self.config.depth_min_m)
            * 255.0
        ).astype(np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    def build_segmentation_preview(self, segmentation_result: SegmentationResult) -> SegmentationPreview:
        overlay = self._read_rgb_image(segmentation_result.overlay_image_path)
        mask = cv2.imread(segmentation_result.mask_image_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found: {segmentation_result.mask_image_path}")
        depth = self._read_rgb_image(segmentation_result.depth_image_path)
        pointcloud = self._read_rgb_image(segmentation_result.pointcloud_preview_path)
        return SegmentationPreview(
            overlay=overlay,
            mask=mask,
            depth=depth,
            pointcloud=pointcloud,
        )

    def build_grasp_preview(self, grasp_result: GraspResult) -> GraspPreview:
        if not grasp_result.visualization_path:
            raise ValueError("Grasp result does not contain a visualization path.")
        projection = self._read_rgb_image(grasp_result.visualization_path)
        translation = np.round(grasp_result.best_translation, 4).tolist()
        quaternion = np.round(grasp_result.best_quaternion_wxyz, 4).tolist()
        summary = (
            f"Score: {grasp_result.best_score:.4f}\n"
            f"Gripper width: {grasp_result.best_width:.4f} m\n"
            f"Position (depth camera): {translation}\n"
            f"Quaternion (wxyz): {quaternion}\n"
            f"Candidate count: {len(grasp_result.all_grasps_summary)}"
        )
        return GraspPreview(projection=projection, summary_text=summary)

    def mark_artifact_status(
        self,
        result: DetectionResult | SegmentationResult | GraspResult,
        status: str,
        **fields: Any,
    ) -> list[str]:
        paths: list[str] = []
        if isinstance(result, DetectionResult):
            if result.annotated_image_path:
                paths.append(result.annotated_image_path)
        elif isinstance(result, SegmentationResult):
            paths.extend(
                [
                    result.overlay_image_path,
                    result.mask_image_path,
                    result.depth_image_path,
                    result.pointcloud_preview_path,
                ]
            )
        elif isinstance(result, GraspResult):
            paths.extend(
                [
                    result.point_cloud_path,
                    result.grasp_json_path,
                    result.all_grasps_json_path,
                ]
            )
            if result.visualization_path:
                paths.append(result.visualization_path)
        else:
            raise TypeError(f"Unsupported result type: {type(result)!r}")

        metadata_paths: list[str] = []
        for path in paths:
            if path:
                metadata_paths.append(
                    self.write_metadata(
                        path,
                        status=status,
                        result_type=type(result).__name__,
                        **fields,
                    )
                )
        return metadata_paths

    def confirm_segmentation(self, segmentation_result: SegmentationResult) -> bool:
        overlay = cv2.cvtColor(cv2.imread(segmentation_result.overlay_image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(segmentation_result.mask_image_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.cvtColor(cv2.imread(segmentation_result.depth_image_path), cv2.COLOR_BGR2RGB)
        pc_img = cv2.cvtColor(cv2.imread(segmentation_result.pointcloud_preview_path), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(overlay)
        axes[0, 0].set_title("FastSAM Overlay")
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title("Mask")
        axes[1, 0].imshow(depth)
        axes[1, 0].set_title("Masked Depth")
        axes[1, 1].imshow(pc_img)
        axes[1, 1].set_title("Point Cloud Projection")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        plt.show(block=True)

        return True

    def _summarize_grasps(self, grasp_group) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for grasp in grasp_group:
            rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64)
            quat_xyzw = Rotation.from_matrix(rotation).as_quat()
            summary.append(
                {
                    "score": float(grasp.score),
                    "width": float(grasp.width),
                    "depth": float(grasp.depth),
                    "translation": np.asarray(grasp.translation, dtype=np.float64).tolist(),
                    "rotation_matrix": rotation.tolist(),
                    "quaternion_wxyz": np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64).tolist(),
                }
            )
        return summary

    def run_anygrasp(self, segmentation_result: SegmentationResult) -> GraspResult:
        points = segmentation_result.points.astype(np.float32)
        colors = segmentation_result.colors.astype(np.float32)
        lims = [
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
            float(points[:, 2].min()),
            float(points[:, 2].max()),
        ]

        grasp_group, filtered_cloud = self._anygrasp.get_grasp(
            points,
            colors,
            lims=lims,
            apply_object_mask=self.config.anygrasp_apply_object_mask,
            dense_grasp=self.config.anygrasp_dense_grasp,
            collision_detection=self.config.anygrasp_collision_detection,
        )
        if len(grasp_group) == 0:
            raise RuntimeError("AnyGrasp returned no grasp candidates.")

        grasp_group = grasp_group.nms().sort_by_score()
        best_grasp = grasp_group[0]
        all_summary = self._summarize_grasps(grasp_group)

        if filtered_cloud is None:
            grasp_cloud = segmentation_result.point_cloud
            cloud_source = "segmentation_fallback"
        else:
            grasp_cloud = filtered_cloud
            cloud_source = "anygrasp_filtered"

        if len(grasp_cloud.points) == 0:
            raise RuntimeError("AnyGrasp point cloud is empty after filtering.")

        grasp_cloud_path = Path(self.config.temp_grasp_dir) / (
            f"{self._safe_stem(segmentation_result.text)}_anygrasp_{segmentation_result.timestamp}_cloud.ply"
        )
        if not o3d.io.write_point_cloud(str(grasp_cloud_path), grasp_cloud):
            raise RuntimeError(f"Failed to write AnyGrasp point cloud: {grasp_cloud_path}")

        best_rotation = np.asarray(best_grasp.rotation_matrix, dtype=np.float64)
        best_quat_xyzw = Rotation.from_matrix(best_rotation).as_quat()
        best_quat_wxyz = np.array(
            [best_quat_xyzw[3], best_quat_xyzw[0], best_quat_xyzw[1], best_quat_xyzw[2]],
            dtype=np.float64,
        )

        best_json_path = Path(self.config.temp_grasp_dir) / (
            f"{self._safe_stem(segmentation_result.text)}_anygrasp_{segmentation_result.timestamp}_best_grasp.json"
        )
        all_json_path = Path(self.config.temp_grasp_dir) / (
            f"{self._safe_stem(segmentation_result.text)}_anygrasp_{segmentation_result.timestamp}_all_grasps.json"
        )

        best_payload = {
            "score": float(best_grasp.score),
            "width": float(best_grasp.width),
            "depth": float(best_grasp.depth),
            "translation": np.asarray(best_grasp.translation, dtype=np.float64).tolist(),
            "rotation_matrix": best_rotation.tolist(),
            "quaternion_wxyz": best_quat_wxyz.tolist(),
        }
        best_json_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        all_json_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        grippers = grasp_group[:20].to_open3d_geometry_list()
        best_grippers = grippers[:1]
        vis_path = Path(self.config.temp_grasp_dir) / (
            f"{self._safe_stem(segmentation_result.text)}_anygrasp_{segmentation_result.timestamp}_vis.png"
        )
        rendered_grippers = self._save_pointcloud_projection(
            grasp_cloud,
            vis_path,
            gripper_geometries=best_grippers,
            title="AnyGrasp Best Gripper",
        )

        result = GraspResult(
            text=segmentation_result.text,
            timestamp=segmentation_result.timestamp,
            best_translation=np.asarray(best_grasp.translation, dtype=np.float64),
            best_rotation_matrix=best_rotation,
            best_quaternion_wxyz=best_quat_wxyz,
            best_width=float(best_grasp.width),
            best_score=float(best_grasp.score),
            all_grasps_summary=all_summary,
            point_cloud_path=str(grasp_cloud_path),
            grasp_json_path=str(best_json_path),
            all_grasps_json_path=str(all_json_path),
            visualization_path=str(vis_path),
            point_cloud=grasp_cloud,
            gripper_geometries=grippers,
        )
        for path in (grasp_cloud_path, best_json_path, all_json_path, vis_path):
            self.write_metadata(
                path,
                kind="grasp",
                status="created",
                source=segmentation_result.source,
                text=segmentation_result.text,
                timestamp=segmentation_result.timestamp,
                best_score=float(best_grasp.score),
                best_width=float(best_grasp.width),
                cloud_source=cloud_source,
                preview_transform=POINTCLOUD_PREVIEW_TRANSFORM_LABEL,
                rendered_grippers=rendered_grippers if path == vis_path else 0,
            )
        return result

    def confirm_grasp(self, grasp_result: GraspResult) -> bool:
        preview = self.build_grasp_preview(grasp_result)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(preview.projection)
        axes[0].set_title("AnyGrasp Projection")
        axes[0].axis("off")
        axes[1].text(0.02, 0.98, preview.summary_text, va="top", ha="left", fontsize=11)
        axes[1].axis("off")
        fig.tight_layout()
        plt.show(block=True)
        return True

    def process(self, text: str, model: mujoco.MjModel, data: mujoco.MjData) -> GraspResult:
        capture = self.capture_rgbd(model, data)
        detection = self.detect(text, capture)
        selected_box_index = self.select_best_box_index(detection)
        segmentation = self.segment(capture, detection, selected_box_index)
        if not self.confirm_segmentation(segmentation):
            raise RuntimeError("Segmentation confirmation was rejected.")
        grasp = self.run_anygrasp(segmentation)
        if not self.confirm_grasp(grasp):
            raise RuntimeError("Grasp confirmation was rejected.")
        return grasp

    def shutdown(self) -> None:
        with self._render_lock:
            if self._glfw_window is not None:
                glfw.make_context_current(self._glfw_window)
            if self._render_context is not None:
                self._render_context.free()
                self._render_context = None
            self._render_scene = None
            self._render_model_id = None
            if self._glfw_window is not None:
                glfw.destroy_window(self._glfw_window)
                self._glfw_window = None
            glfw.make_context_current(None)
            try:
                glfw.terminate()
            except Exception:
                pass
