from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import mujoco


FrameName = Literal["world", "rgb_camera", "depth_camera", "rgb_camera_optical", "depth_camera_optical"]
SolverName = Literal["mujoco", "ikpy"]


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion norm must be non-zero.")
    return quat / norm


def _quat_wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_wxyz(quaternion)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion norm must be non-zero.")
    quat = quat / norm
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(_quat_wxyz_to_xyzw(quaternion_wxyz)).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def _rotation_error(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    delta = Rotation.from_matrix(target_rot @ current_rot.T)
    return delta.as_rotvec()


def _compose_full_ctrl(arm_qpos: np.ndarray, gripper_value: float) -> np.ndarray:
    full_ctrl = np.zeros(7, dtype=np.float64)
    full_ctrl[:6] = np.asarray(arm_qpos, dtype=np.float64).reshape(6)
    full_ctrl[6] = float(gripper_value)
    return full_ctrl


@dataclass
class IKResult:
    success: bool
    solver_name: str
    arm_qpos: np.ndarray
    full_ctrl: np.ndarray
    gripper_value: float
    pos_error: float
    rot_error: float
    iterations: int
    message: str


class _BasePiperIKSolver:
    def __init__(self, controller: "PiperIKController") -> None:
        self.controller = controller

    @property
    def model(self) -> mujoco.MjModel:
        return self.controller.model

    @property
    def data(self) -> mujoco.MjData:
        return self.controller.data

    def _gripper_value(self, gripper: Optional[float]) -> float:
        if gripper is None:
            return self.controller.get_gripper()
        return self.controller.clamp_gripper(gripper)

    def _make_result(
        self,
        *,
        success: bool,
        solver_name: str,
        arm_qpos: np.ndarray,
        gripper_value: float,
        pos_error: float,
        rot_error: float,
        iterations: int,
        message: str,
    ) -> IKResult:
        return IKResult(
            success=success,
            solver_name=solver_name,
            arm_qpos=np.asarray(arm_qpos, dtype=np.float64).reshape(6),
            full_ctrl=_compose_full_ctrl(arm_qpos, gripper_value),
            gripper_value=float(gripper_value),
            pos_error=float(pos_error),
            rot_error=float(rot_error),
            iterations=int(iterations),
            message=message,
        )


class MuJoCoIKSolver(_BasePiperIKSolver):
    def solve(
        self,
        target_position_world: np.ndarray,
        target_quaternion_world: np.ndarray,
        *,
        gripper: Optional[float] = None,
        seed_qpos: Optional[np.ndarray] = None,
        max_iters: int = 100,
        position_tol: float = 1e-3,
        rotation_tol: float = 1e-2,
        damping: float = 1e-3,
    ) -> IKResult:
        target_position = np.asarray(target_position_world, dtype=np.float64).reshape(3)
        target_quaternion = _normalize_quaternion_wxyz(target_quaternion_world)
        target_rot = Rotation.from_quat(_quat_wxyz_to_xyzw(target_quaternion)).as_matrix()
        gripper_value = self._gripper_value(gripper)

        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        original_act = self.data.act.copy() if self.model.na else None

        arm_qpos = (
            np.asarray(seed_qpos, dtype=np.float64).reshape(6)
            if seed_qpos is not None
            else self.controller.get_arm_qpos()
        )
        arm_qpos = self.controller.clamp_arm_qpos(arm_qpos)
        self.controller.set_arm_qpos(arm_qpos)

        success = False
        message = "least-squares did not converge"
        pos_error_norm = np.inf
        rot_error_norm = np.inf
        iterations = 0

        try:
            def residual(q: np.ndarray) -> np.ndarray:
                self.controller.set_arm_qpos(q)
                mujoco.mj_forward(self.model, self.data)
                current_position = self.data.site_xpos[self.controller.ee_site_id].copy()
                current_rot = self.data.site_xmat[self.controller.ee_site_id].reshape(3, 3).copy()
                pos_error_vec = target_position - current_position
                rot_error_vec = _rotation_error(target_rot, current_rot)
                return np.concatenate([pos_error_vec, rot_error_vec])

            result = least_squares(
                residual,
                arm_qpos,
                bounds=(
                    self.controller.arm_joint_ranges[:, 0],
                    self.controller.arm_joint_ranges[:, 1],
                ),
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=max_iters,
                method="trf",
                x_scale="jac",
            )
            arm_qpos = self.controller.clamp_arm_qpos(result.x)
            self.controller.set_arm_qpos(arm_qpos)
            mujoco.mj_forward(self.model, self.data)
            iterations = int(result.nfev)
            success = bool(result.success)
            message = str(result.message)

            mujoco.mj_forward(self.model, self.data)
            arm_qpos = self.controller.get_arm_qpos()
            current_position = self.data.site_xpos[self.controller.ee_site_id].copy()
            current_rot = self.data.site_xmat[self.controller.ee_site_id].reshape(3, 3).copy()
            pos_error_norm = float(np.linalg.norm(target_position - current_position))
            rot_error_norm = float(np.linalg.norm(_rotation_error(target_rot, current_rot)))
            success = success and pos_error_norm <= position_tol and rot_error_norm <= rotation_tol
            return self._make_result(
                success=success,
                solver_name="mujoco",
                arm_qpos=arm_qpos,
                gripper_value=gripper_value,
                pos_error=pos_error_norm,
                rot_error=rot_error_norm,
                iterations=iterations,
                message=message,
            )
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            if original_act is not None:
                self.data.act[:] = original_act
            mujoco.mj_forward(self.model, self.data)


class IKPySolver(_BasePiperIKSolver):
    INSTALL_HINT = "IKPy is not installed. Activate the 'mujoco' conda environment and run: pip install ikpy"

    def __init__(self, controller: "PiperIKController") -> None:
        super().__init__(controller)
        try:
            from ikpy.chain import Chain
            from ikpy.link import OriginLink, URDFLink
        except ModuleNotFoundError as exc:
            self._import_error = exc
            self._chain = None
            self._Chain = None
            self._OriginLink = None
            self._URDFLink = None
            return

        self._import_error = None
        self._Chain = Chain
        self._OriginLink = OriginLink
        self._URDFLink = URDFLink
        self._chain = self._build_chain()

    def is_available(self) -> bool:
        return self._import_error is None

    def _ensure_available(self) -> None:
        if self._import_error is not None:
            raise ModuleNotFoundError(self.INSTALL_HINT) from self._import_error

    def _build_chain(self):
        self._ensure_available()
        links = [self._OriginLink()]
        for body_name, joint_name in zip(
            self.controller.arm_body_names,
            self.controller.arm_joint_names,
        ):
            body_id = self.controller.body_ids[body_name]
            joint_id = self.controller.joint_ids[joint_name]
            translation = self.model.body_pos[body_id].copy()
            orientation = Rotation.from_quat(
                _quat_wxyz_to_xyzw(self.model.body_quat[body_id].copy())
            ).as_euler("xyz", degrees=False)
            axis = self.model.jnt_axis[joint_id].copy()
            bounds = tuple(self.model.jnt_range[joint_id].copy())
            links.append(
                self._URDFLink(
                    name=joint_name,
                    origin_translation=translation,
                    origin_orientation=orientation,
                    rotation=axis,
                    bounds=bounds,
                )
            )

        tool_pos = self.model.site_pos[self.controller.ee_site_id].copy()
        tool_quat = self.model.site_quat[self.controller.ee_site_id].copy()
        tool_orientation = Rotation.from_quat(_quat_wxyz_to_xyzw(tool_quat)).as_euler(
            "xyz", degrees=False
        )
        links.append(
            self._URDFLink(
                name="ee_grasp_center",
                origin_translation=tool_pos,
                origin_orientation=tool_orientation,
                rotation=None,
                joint_type="fixed",
            )
        )
        active_links_mask = [False] + [True] * 6 + [False]
        return self._Chain(
            name="piper_arm",
            links=links,
            active_links_mask=active_links_mask,
        )

    def solve(
        self,
        target_position_world: np.ndarray,
        target_quaternion_world: np.ndarray,
        *,
        gripper: Optional[float] = None,
        seed_qpos: Optional[np.ndarray] = None,
        max_iters: int = 100,
        position_tol: float = 1e-3,
        rotation_tol: float = 1e-2,
    ) -> IKResult:
        self._ensure_available()
        gripper_value = self._gripper_value(gripper)
        target_transform = _pose_matrix(target_position_world, target_quaternion_world)

        initial_position = np.zeros(len(self._chain.links), dtype=np.float64)
        initial_position[1:7] = (
            np.asarray(seed_qpos, dtype=np.float64).reshape(6)
            if seed_qpos is not None
            else self.controller.get_arm_qpos()
        )

        q_solution = self._chain.inverse_kinematics_frame(
            target_transform,
            initial_position=initial_position,
            max_iter=max_iters,
            orientation_mode="all",
        )
        arm_qpos = self.controller.clamp_arm_qpos(np.asarray(q_solution[1:7], dtype=np.float64))

        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        original_act = self.data.act.copy() if self.model.na else None
        try:
            self.controller.set_arm_qpos(arm_qpos)
            mujoco.mj_forward(self.model, self.data)
            current_position = self.data.site_xpos[self.controller.ee_site_id].copy()
            current_rot = self.data.site_xmat[self.controller.ee_site_id].reshape(3, 3).copy()
            target_rot = Rotation.from_quat(
                _quat_wxyz_to_xyzw(target_quaternion_world)
            ).as_matrix()
            pos_error = float(np.linalg.norm(np.asarray(target_position_world) - current_position))
            rot_error = float(np.linalg.norm(_rotation_error(target_rot, current_rot)))
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            if original_act is not None:
                self.data.act[:] = original_act
            mujoco.mj_forward(self.model, self.data)

        success = pos_error <= position_tol and rot_error <= rotation_tol
        message = "converged" if success else "ikpy returned a pose outside tolerance"
        return self._make_result(
            success=success,
            solver_name="ikpy",
            arm_qpos=arm_qpos,
            gripper_value=gripper_value,
            pos_error=pos_error,
            rot_error=rot_error,
            iterations=max_iters,
            message=message,
        )


class PiperIKController:
    ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
    ARM_BODY_NAMES = tuple(f"link{i}" for i in range(1, 7))
    GRIPPER_ACTUATOR_NAME = "gripper"
    CAMERA_SITE_NAMES = {
        "rgb_camera": "d435i_rgb_frame",
        "depth_camera": "d435i_depth_frame",
    }
    EE_SITE_NAME = "ee_grasp_center"
    GRIPPER_RANGE = np.array([-0.001, 0.035], dtype=np.float64)
    T_GRASP_TO_EE = np.eye(4, dtype=np.float64)
    T_OPTICAL_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.joint_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.ARM_JOINT_NAMES + ("joint7", "joint8")
        }
        self.body_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.ARM_BODY_NAMES + ("link7", "link8")
        }
        self.actuator_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ARM_JOINT_NAMES + (self.GRIPPER_ACTUATOR_NAME,)
        }
        self.site_ids = {
            self.EE_SITE_NAME: self._require_id(mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAME),
            **{
                frame: self._require_id(mujoco.mjtObj.mjOBJ_SITE, site_name)
                for frame, site_name in self.CAMERA_SITE_NAMES.items()
            },
        }
        self.camera_ids = {
            frame: self._require_id(mujoco.mjtObj.mjOBJ_CAMERA, frame)
            for frame in self.CAMERA_SITE_NAMES
        }

        self.ee_site_id = self.site_ids[self.EE_SITE_NAME]
        self.arm_joint_names = self.ARM_JOINT_NAMES
        self.arm_body_names = self.ARM_BODY_NAMES
        self.arm_joint_ids = np.array([self.joint_ids[name] for name in self.arm_joint_names], dtype=np.int32)
        self.arm_actuator_ids = np.array(
            [self.actuator_ids[name] for name in self.arm_joint_names],
            dtype=np.int32,
        )
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=np.int32)
        self.arm_joint_ranges = np.asarray(
            [self.model.jnt_range[jid].copy() for jid in self.arm_joint_ids],
            dtype=np.float64,
        )
        self.gripper_actuator_id = self.actuator_ids[self.GRIPPER_ACTUATOR_NAME]

        self.mujoco_solver = MuJoCoIKSolver(self)
        self.ikpy_solver = IKPySolver(self)

    @classmethod
    def from_xml_path(cls, xml_path: str | Path) -> "PiperIKController":
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return cls(model, data)

    def _require_id(self, obj_type: int, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Required object '{name}' of type {obj_type} not found in model.")
        return obj_id

    def clamp_arm_qpos(self, arm_qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(arm_qpos, dtype=np.float64).reshape(6)
        lower = self.arm_joint_ranges[:, 0]
        upper = self.arm_joint_ranges[:, 1]
        return np.clip(qpos, lower, upper)

    def clamp_gripper(self, opening: float) -> float:
        return float(np.clip(opening, self.GRIPPER_RANGE[0], self.GRIPPER_RANGE[1]))

    def get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_ids].copy()

    def set_arm_qpos(self, arm_qpos: np.ndarray) -> None:
        self.data.qpos[self.arm_qpos_ids] = self.clamp_arm_qpos(arm_qpos)

    def get_gripper(self) -> float:
        return self.clamp_gripper(self.data.ctrl[self.gripper_actuator_id])

    def set_gripper(self, opening: float) -> float:
        opening = self.clamp_gripper(opening)
        self.data.ctrl[self.gripper_actuator_id] = opening
        return opening

    def _site_transform_world(self, site_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[site_id]
        return transform

    def _camera_transform_world(self, camera_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3)
        transform[:3, 3] = self.data.cam_xpos[camera_id]
        return transform

    def _target_to_world(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        frame: FrameName,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_transform = _pose_matrix(position, quaternion) @ self.T_GRASP_TO_EE
        if frame == "world":
            world_transform = target_transform
        elif frame.endswith("_optical"):
            camera_frame = frame.removesuffix("_optical")
            if camera_frame not in self.camera_ids:
                raise ValueError(f"Unsupported optical camera frame '{frame}'.")
            mujoco.mj_forward(self.model, self.data)
            camera_world = self._camera_transform_world(self.camera_ids[camera_frame])
            world_transform = camera_world @ self.T_OPTICAL_TO_MUJOCO_CAMERA @ target_transform
        else:
            camera_site_id = self.site_ids[frame]
            mujoco.mj_forward(self.model, self.data)
            camera_world = self._site_transform_world(camera_site_id)
            world_transform = camera_world @ target_transform
        world_position = world_transform[:3, 3]
        world_quat = _quat_xyzw_to_wxyz(
            Rotation.from_matrix(world_transform[:3, :3]).as_quat()
        )
        return world_position, world_quat

    def target_to_world_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        frame: FrameName = "world",
    ) -> tuple[np.ndarray, np.ndarray]:
        if frame not in ("world", "rgb_camera", "depth_camera", "rgb_camera_optical", "depth_camera_optical"):
            raise ValueError(f"Unsupported frame '{frame}'.")
        return self._target_to_world(position, quaternion, frame)

    def solve(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        frame: FrameName = "world",
        solver: SolverName = "mujoco",
        gripper: Optional[float] = None,
        seed_qpos: Optional[np.ndarray] = None,
        max_iters: int = 100,
        position_tol: float = 1e-3,
        rotation_tol: float = 1e-2,
    ) -> IKResult:
        if frame not in ("world", "rgb_camera", "depth_camera", "rgb_camera_optical", "depth_camera_optical"):
            raise ValueError(f"Unsupported frame '{frame}'.")
        world_position, world_quaternion = self._target_to_world(position, quaternion, frame)
        if solver == "mujoco":
            return self.mujoco_solver.solve(
                world_position,
                world_quaternion,
                gripper=gripper,
                seed_qpos=seed_qpos,
                max_iters=max_iters,
                position_tol=position_tol,
                rotation_tol=rotation_tol,
            )
        if solver == "ikpy":
            return self.ikpy_solver.solve(
                world_position,
                world_quaternion,
                gripper=gripper,
                seed_qpos=seed_qpos,
                max_iters=max_iters,
                position_tol=position_tol,
                rotation_tol=rotation_tol,
            )
        raise ValueError(f"Unsupported solver '{solver}'.")

    def apply(self, result: IKResult, *, write_gripper: bool = True) -> None:
        if result.full_ctrl.shape != (7,):
            raise ValueError("Expected IKResult.full_ctrl to have shape (7,).")
        self.data.ctrl[self.arm_actuator_ids] = result.full_ctrl[:6]
        if write_gripper:
            self.data.ctrl[self.gripper_actuator_id] = self.clamp_gripper(result.gripper_value)

    def get_end_effector_pose(self, frame: FrameName = "world") -> tuple[np.ndarray, np.ndarray]:
        if frame not in ("world", "rgb_camera", "depth_camera", "rgb_camera_optical", "depth_camera_optical"):
            raise ValueError(f"Unsupported frame '{frame}'.")
        mujoco.mj_forward(self.model, self.data)
        ee_world = self._site_transform_world(self.ee_site_id)
        if frame == "world":
            transform = ee_world
        elif frame.endswith("_optical"):
            camera_frame = frame.removesuffix("_optical")
            if camera_frame not in self.camera_ids:
                raise ValueError(f"Unsupported optical camera frame '{frame}'.")
            camera_world = self._camera_transform_world(self.camera_ids[camera_frame])
            optical_world = camera_world @ self.T_OPTICAL_TO_MUJOCO_CAMERA
            transform = _invert_transform(optical_world) @ ee_world
        else:
            camera_world = self._site_transform_world(self.site_ids[frame])
            transform = _invert_transform(camera_world) @ ee_world
        position = transform[:3, 3].copy()
        quaternion = _quat_xyzw_to_wxyz(Rotation.from_matrix(transform[:3, :3]).as_quat())
        return position, quaternion
