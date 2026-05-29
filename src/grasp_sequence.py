from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.ik.piper_ik import IKResult, PiperIKController


ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
CTRL_ACTUATOR_NAMES = (*ARM_JOINT_NAMES, "gripper")
HOME_ARM_QPOS = np.array([0.0, 1.5, -1.45, 0.0, 1.22, 0.0], dtype=np.float64)
HOME_GRIPPER = 0.02
HOME_CTRL = np.array([*HOME_ARM_QPOS, HOME_GRIPPER], dtype=np.float64)

PLACE_JOINT1_RAD = -2.35619449
PLACE_EE_Z_M = 0.05
APPROACH_OFFSET_M = 0.08
LIFT_OFFSET_M = 0.12
MIN_GRASP_WORLD_Z_M = 0.03
GRASP_CAMERA_FRAME = "depth_camera_optical"
MAX_REACHABLE_GRASP_CANDIDATES = 20
OPEN_GRIPPER = 0.03
CLOSED_GRIPPER = -0.001

MOVE_DURATION_S = 9.0
CLOSE_GRIPPER_DELAY_S = 5.0
OPEN_GRIPPER_TIMEOUT_S = 3.0
OPEN_GRIPPER_REACHED_RATIO = 0.9
PLACE_TURN_DURATION_S = 7.5
HOME_RETURN_DURATION_S = 9.0
IK_MAX_ITERS = 200
APPROACH_POSITION_TOL = 0.08
GRASP_POSITION_TOL = 0.02
LIFT_POSITION_TOL = 0.08
PLACE_POSITION_TOL = 0.04
ROTATION_TOL = 0.05


def _require_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Required object '{name}' of type {obj_type} not found.")
    return obj_id


def _joint_qpos_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [
            model.jnt_qposadr[_require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_names
        ],
        dtype=np.int32,
    )


def _joint_dof_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [
            model.jnt_dofadr[_require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_names
        ],
        dtype=np.int32,
    )


def _actuator_ids(model: mujoco.MjModel, actuator_names: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [
            _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in actuator_names
        ],
        dtype=np.int32,
    )


def apply_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    arm_qpos_ids = _joint_qpos_addresses(model, ARM_JOINT_NAMES)
    arm_dof_ids = _joint_dof_addresses(model, ARM_JOINT_NAMES)
    ctrl_ids = _actuator_ids(model, CTRL_ACTUATOR_NAMES)

    data.qpos[arm_qpos_ids] = HOME_ARM_QPOS
    data.qvel[arm_dof_ids] = 0.0

    joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    joint8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint8")
    if joint7_id >= 0:
        data.qpos[model.jnt_qposadr[joint7_id]] = HOME_GRIPPER
        data.qvel[model.jnt_dofadr[joint7_id]] = 0.0
    if joint8_id >= 0:
        data.qpos[model.jnt_qposadr[joint8_id]] = -HOME_GRIPPER
        data.qvel[model.jnt_dofadr[joint8_id]] = 0.0

    data.ctrl[ctrl_ids] = HOME_CTRL
    mujoco.mj_forward(model, data)


@dataclass
class MotionRuntime:
    is_running: Any
    sync: Any
    stage: Any | None = None
    lock: Any | None = None


@dataclass(frozen=True)
class _GraspCandidate:
    index: int
    translation: np.ndarray
    quaternion_wxyz: np.ndarray
    score: float | None = None
    width: float | None = None


@dataclass(frozen=True)
class _SelectedGraspMotion:
    candidate: _GraspCandidate
    grasp_position: np.ndarray
    pre_grasp_position: np.ndarray
    lift_position: np.ndarray
    quaternion_wxyz: np.ndarray
    orientation_source: str


class ViewerMotionRuntime:
    def __init__(self, viewer) -> None:
        self.viewer = viewer

    def is_running(self) -> bool:
        return bool(self.viewer.is_running())

    def sync(self) -> None:
        self.viewer.sync()

    def lock(self):
        return self.viewer.lock()

    def stage(self, stage: str, position: np.ndarray | None = None) -> None:
        _print_stage(stage, position)


def _runtime_is_running(runtime: MotionRuntime | ViewerMotionRuntime) -> bool:
    return bool(runtime.is_running())


def _runtime_sync(runtime: MotionRuntime | ViewerMotionRuntime) -> None:
    runtime.sync()


def _runtime_lock(runtime: MotionRuntime | ViewerMotionRuntime):
    lock_callback = getattr(runtime, "lock", None)
    if lock_callback is None:
        return nullcontext()
    return lock_callback()


def _runtime_stage(
    runtime: MotionRuntime | ViewerMotionRuntime,
    stage: str,
    position: np.ndarray | None = None,
) -> None:
    stage_callback = getattr(runtime, "stage", None)
    if stage_callback is None:
        _print_stage(stage, position)
        return
    stage_callback(stage, position)


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _step_runtime(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    duration_s: float,
) -> bool:
    end_time = time.time() + duration_s
    while _runtime_is_running(runtime) and time.time() < end_time:
        step_start = time.time()
        with _runtime_lock(runtime):
            mujoco.mj_step(model, data)
        _runtime_sync(runtime)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    return _runtime_is_running(runtime)


def _step_with_callback(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    duration_s: float,
    step_callback: Any,
) -> bool:
    start_time = time.time()
    end_time = start_time + duration_s
    if duration_s <= 0.0:
        with _runtime_lock(runtime):
            step_callback(1.0)
            mujoco.mj_forward(model, data)
        _runtime_sync(runtime)
        return _runtime_is_running(runtime)

    while _runtime_is_running(runtime) and time.time() < end_time:
        step_start = time.time()
        alpha = (step_start - start_time) / duration_s
        with _runtime_lock(runtime):
            step_callback(_smoothstep(alpha))
            mujoco.mj_step(model, data)
        _runtime_sync(runtime)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if not _runtime_is_running(runtime):
        return False
    with _runtime_lock(runtime):
        step_callback(1.0)
        mujoco.mj_forward(model, data)
    _runtime_sync(runtime)
    return True


def _interpolate_full_ctrl(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
    target_full_ctrl: np.ndarray,
    duration_s: float,
) -> bool:
    start_full_ctrl = data.ctrl[:7].copy()
    target_full_ctrl = np.asarray(target_full_ctrl, dtype=np.float64).reshape(7)
    target_gripper = target_full_ctrl[6]

    def write_interpolated(alpha: float) -> None:
        full_ctrl = start_full_ctrl.copy()
        full_ctrl[:6] = start_full_ctrl[:6] + alpha * (target_full_ctrl[:6] - start_full_ctrl[:6])
        full_ctrl[6] = target_gripper
        _write_full_ctrl(ik_controller, full_ctrl)

    return _step_with_callback(model, data, runtime, duration_s, write_interpolated)


def _print_stage(stage: str, position: np.ndarray | None = None) -> None:
    if position is None:
        print(f"[INFO] Stage: {stage}")
        return
    rounded = np.round(position, 4).tolist()
    print(f"[INFO] Stage: {stage}, target_pos={rounded}")


def _raise_ik_error(stage: str, result: IKResult, position: np.ndarray) -> None:
    raise RuntimeError(
        f"IK failed during '{stage}': {result.message} "
        f"(target={np.round(position, 4).tolist()}, "
        f"pos_error={result.pos_error:.6f}, rot_error={result.rot_error:.6f})"
    )


def _solve_pose(
    *,
    ik_controller: PiperIKController,
    position: np.ndarray,
    quaternion: np.ndarray,
    gripper: float,
    position_tol: float,
    rotation_tol: float = ROTATION_TOL,
    seed_qpos: np.ndarray | None = None,
    runtime: MotionRuntime | ViewerMotionRuntime | None = None,
) -> IKResult:
    lock = _runtime_lock(runtime) if runtime is not None else nullcontext()
    with lock:
        return ik_controller.solve(
            position=position,
            quaternion=quaternion,
            frame="world",
            solver="mujoco",
            gripper=gripper,
            seed_qpos=seed_qpos,
            max_iters=IK_MAX_ITERS,
            position_tol=position_tol,
            rotation_tol=rotation_tol,
        )


def _solve_and_apply(
    *,
    stage: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
    position: np.ndarray,
    quaternion: np.ndarray,
    gripper: float,
    duration_s: float | None = None,
    position_tol: float = GRASP_POSITION_TOL,
    rotation_tol: float = ROTATION_TOL,
) -> bool:
    _runtime_stage(runtime, stage, position)
    result = _solve_pose(
        ik_controller=ik_controller,
        position=position,
        quaternion=quaternion,
        gripper=gripper,
        position_tol=position_tol,
        rotation_tol=rotation_tol,
        runtime=runtime,
    )
    if not result.success:
        _raise_ik_error(stage, result, position)

    print(
        f"[INFO] {stage} target_ctrl={np.round(result.full_ctrl, 4)} "
        f"pos_error={result.pos_error:.4f} rot_error={result.rot_error:.4f}"
    )
    return _interpolate_full_ctrl(
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        target_full_ctrl=result.full_ctrl,
        duration_s=MOVE_DURATION_S if duration_s is None else duration_s,
    )


def _quat_wxyz_from_rotation_matrix(rotation_matrix: Any) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(
        np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    ).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _candidate_from_summary(index: int, summary: Any) -> _GraspCandidate | None:
    if not isinstance(summary, dict):
        return None
    try:
        translation = np.asarray(summary["translation"], dtype=np.float64).reshape(3)
        quaternion = summary.get("quaternion_wxyz")
        if quaternion is None:
            quaternion = _quat_wxyz_from_rotation_matrix(summary["rotation_matrix"])
        quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
        score = summary.get("score")
        width = summary.get("width")
        return _GraspCandidate(
            index=index,
            translation=translation,
            quaternion_wxyz=quaternion,
            score=float(score) if score is not None else None,
            width=float(width) if width is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _best_candidate_from_result(grasp_result: Any) -> _GraspCandidate:
    score = getattr(grasp_result, "best_score", None)
    width = getattr(grasp_result, "best_width", None)
    return _GraspCandidate(
        index=0,
        translation=np.asarray(grasp_result.best_translation, dtype=np.float64).reshape(3),
        quaternion_wxyz=np.asarray(grasp_result.best_quaternion_wxyz, dtype=np.float64).reshape(4),
        score=float(score) if score is not None else None,
        width=float(width) if width is not None else None,
    )


def _iter_grasp_candidates(grasp_result: Any) -> list[_GraspCandidate]:
    candidates: list[_GraspCandidate] = []
    for index, summary in enumerate(getattr(grasp_result, "all_grasps_summary", None) or []):
        candidate = _candidate_from_summary(index, summary)
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= MAX_REACHABLE_GRASP_CANDIDATES:
            break
    if candidates:
        return candidates
    return [_best_candidate_from_result(grasp_result)]


def _motion_from_world_grasp(
    candidate: _GraspCandidate,
    grasp_position: np.ndarray,
    quaternion: np.ndarray,
    orientation_source: str,
) -> _SelectedGraspMotion:
    clamped_grasp_position = np.asarray(grasp_position, dtype=np.float64).reshape(3).copy()
    clamped_grasp_position[2] = max(clamped_grasp_position[2], MIN_GRASP_WORLD_Z_M)

    pre_grasp_position = clamped_grasp_position.copy()
    pre_grasp_position[2] += APPROACH_OFFSET_M
    lift_position = clamped_grasp_position.copy()
    lift_position[2] += LIFT_OFFSET_M
    return _SelectedGraspMotion(
        candidate=candidate,
        grasp_position=clamped_grasp_position,
        pre_grasp_position=pre_grasp_position,
        lift_position=lift_position,
        quaternion_wxyz=np.asarray(quaternion, dtype=np.float64).reshape(4),
        orientation_source=orientation_source,
    )


def _check_grasp_motion_reachable(
    ik_controller: PiperIKController,
    motion: _SelectedGraspMotion,
    runtime: MotionRuntime | ViewerMotionRuntime | None = None,
) -> tuple[bool, str]:
    pre_result = _solve_pose(
        ik_controller=ik_controller,
        position=motion.pre_grasp_position,
        quaternion=motion.quaternion_wxyz,
        gripper=OPEN_GRIPPER,
        position_tol=APPROACH_POSITION_TOL,
        runtime=runtime,
    )
    if not pre_result.success:
        return (
            False,
            f"pre pos={pre_result.pos_error:.4f} rot={pre_result.rot_error:.4f} "
            f"msg={pre_result.message}",
        )

    grasp_result = _solve_pose(
        ik_controller=ik_controller,
        position=motion.grasp_position,
        quaternion=motion.quaternion_wxyz,
        gripper=OPEN_GRIPPER,
        position_tol=GRASP_POSITION_TOL,
        seed_qpos=pre_result.arm_qpos,
        runtime=runtime,
    )
    if not grasp_result.success:
        return (
            False,
            f"grasp pos={grasp_result.pos_error:.4f} rot={grasp_result.rot_error:.4f} "
            f"msg={grasp_result.message}",
        )
    lift_result = _solve_pose(
        ik_controller=ik_controller,
        position=motion.lift_position,
        quaternion=motion.quaternion_wxyz,
        gripper=CLOSED_GRIPPER,
        position_tol=LIFT_POSITION_TOL,
        seed_qpos=grasp_result.arm_qpos,
        runtime=runtime,
    )
    if not lift_result.success:
        return (
            False,
            f"lift pos={lift_result.pos_error:.4f} rot={lift_result.rot_error:.4f} "
            f"msg={lift_result.message}",
        )
    return True, "ok"


def _select_reachable_grasp_motion(
    ik_controller: PiperIKController,
    grasp_result: Any,
    runtime: MotionRuntime | ViewerMotionRuntime | None = None,
) -> _SelectedGraspMotion:
    candidates = _iter_grasp_candidates(grasp_result)
    with _runtime_lock(runtime) if runtime is not None else nullcontext():
        _, fallback_quaternion = ik_controller.get_end_effector_pose(frame="world")
    failures: list[str] = []

    for candidate in candidates:
        with _runtime_lock(runtime) if runtime is not None else nullcontext():
            grasp_position, grasp_quaternion = ik_controller.target_to_world_pose(
                candidate.translation,
                candidate.quaternion_wxyz,
                frame=GRASP_CAMERA_FRAME,
            )
        motions = [
            _motion_from_world_grasp(candidate, grasp_position, grasp_quaternion, "anygrasp"),
            _motion_from_world_grasp(candidate, grasp_position, fallback_quaternion, "current-ee"),
        ]
        for motion in motions:
            ok, message = _check_grasp_motion_reachable(ik_controller, motion, runtime)
            if ok:
                print(
                    "[INFO] selected grasp "
                    f"candidate={candidate.index} score={candidate.score} "
                    f"orientation={motion.orientation_source} "
                    f"world_pos={np.round(motion.grasp_position, 4).tolist()}"
                )
                return motion
            failures.append(
                f"candidate={candidate.index} orientation={motion.orientation_source}: {message}"
            )

    failure_text = "; ".join(failures[:6])
    if len(failures) > 6:
        failure_text += f"; ... {len(failures) - 6} more"
    raise RuntimeError(
        "No reachable grasp candidate after camera optical transform. "
        f"Tried {len(candidates)} candidate(s). {failure_text}"
    )


def _write_full_ctrl(ik_controller: PiperIKController, full_ctrl: np.ndarray) -> None:
    ik_controller.data.ctrl[ik_controller.arm_actuator_ids] = full_ctrl[:6]
    ik_controller.data.ctrl[ik_controller.gripper_actuator_id] = ik_controller.clamp_gripper(full_ctrl[6])


def _gripper_joint_opening(ik_controller: PiperIKController) -> float | None:
    joint_ids = getattr(ik_controller, "joint_ids", None)
    data = getattr(ik_controller, "data", None)
    model = getattr(ik_controller, "model", None)
    if joint_ids is None or data is None or model is None:
        return None
    try:
        joint7_id = joint_ids["joint7"]
        joint8_id = joint_ids["joint8"]
        joint7_qpos = model.jnt_qposadr[joint7_id]
        joint8_qpos = model.jnt_qposadr[joint8_id]
        return float((data.qpos[joint7_qpos] - data.qpos[joint8_qpos]) * 0.5)
    except (KeyError, AttributeError, IndexError, TypeError):
        return None


def _wait_for_open_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
) -> bool:
    target_opening = OPEN_GRIPPER * OPEN_GRIPPER_REACHED_RATIO
    warned_missing_joints = False
    start_time = time.time()
    end_time = start_time + OPEN_GRIPPER_TIMEOUT_S

    while _runtime_is_running(runtime) and time.time() < end_time:
        step_start = time.time()
        with _runtime_lock(runtime):
            opening = _gripper_joint_opening(ik_controller)
            if opening is not None and opening >= target_opening:
                print(
                    f"[INFO] open-gripper reached opening={opening:.4f} "
                    f"target={target_opening:.4f}"
                )
                mujoco.mj_forward(model, data)
                return True
            if opening is None and not warned_missing_joints:
                print("[WARN] open-gripper joint state unavailable; falling back to timeout wait")
                warned_missing_joints = True
            mujoco.mj_step(model, data)
        _runtime_sync(runtime)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if not _runtime_is_running(runtime):
        return False
    opening = _gripper_joint_opening(ik_controller)
    print(
        "[WARN] open-gripper wait timed out "
        f"opening={None if opening is None else round(opening, 4)} "
        f"target={target_opening:.4f}"
    )
    return True


def _turn_joint1_for_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
) -> bool:
    _runtime_stage(runtime, "turn joint1 for place")
    target_ctrl = data.ctrl[:7].copy()
    target_ctrl[0] = PLACE_JOINT1_RAD
    target_ctrl[6] = CLOSED_GRIPPER
    print(f"[INFO] place-turn target_ctrl={np.round(target_ctrl, 4)}")
    return _interpolate_full_ctrl(
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        target_full_ctrl=target_ctrl,
        duration_s=PLACE_TURN_DURATION_S,
    )


def _open_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
) -> bool:
    _runtime_stage(runtime, "open gripper")
    with _runtime_lock(runtime):
        ik_controller.set_gripper(OPEN_GRIPPER)
        mujoco.mj_forward(model, data)
    _runtime_sync(runtime)
    print(f"[INFO] open-gripper ctrl={np.round(data.ctrl[:7], 4)}")
    return _wait_for_open_gripper(model, data, runtime, ik_controller)


def _close_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
) -> bool:
    _runtime_stage(runtime, "close gripper")
    with _runtime_lock(runtime):
        ik_controller.set_gripper(CLOSED_GRIPPER)
        mujoco.mj_forward(model, data)
    _runtime_sync(runtime)
    print(f"[INFO] close-gripper ctrl={np.round(data.ctrl[:7], 4)}")
    ok = _step_runtime(model, data, runtime, CLOSE_GRIPPER_DELAY_S)
    if ok:
        print(f"[INFO] close-gripper fixed delay complete duration={CLOSE_GRIPPER_DELAY_S:.2f}s")
    return ok


def _return_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
) -> bool:
    _runtime_stage(runtime, "return home")
    print(f"[INFO] home target_ctrl={np.round(HOME_CTRL, 4)}")
    if not _interpolate_full_ctrl(
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        target_full_ctrl=HOME_CTRL,
        duration_s=HOME_RETURN_DURATION_S,
    ):
        return False
    with _runtime_lock(runtime):
        apply_home_pose(model, data)
    _runtime_sync(runtime)
    return True


def execute_pick_place_sequence_with_runtime(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: MotionRuntime | ViewerMotionRuntime,
    ik_controller: PiperIKController,
    grasp_result: Any,
) -> bool:
    _runtime_stage(runtime, "select grasp candidate")
    motion = _select_reachable_grasp_motion(ik_controller, grasp_result, runtime)

    if not _solve_and_apply(
        stage="pre-grasp",
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        position=motion.pre_grasp_position,
        quaternion=motion.quaternion_wxyz,
        gripper=OPEN_GRIPPER,
        position_tol=APPROACH_POSITION_TOL,
    ):
        return False

    if not _solve_and_apply(
        stage="descend to grasp",
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        position=motion.grasp_position,
        quaternion=motion.quaternion_wxyz,
        gripper=OPEN_GRIPPER,
        position_tol=GRASP_POSITION_TOL,
    ):
        return False

    if not _close_gripper(model, data, runtime, ik_controller):
        return False

    if not _solve_and_apply(
        stage="lift object",
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        position=motion.lift_position,
        quaternion=motion.quaternion_wxyz,
        gripper=CLOSED_GRIPPER,
        position_tol=LIFT_POSITION_TOL,
    ):
        return False

    if not _turn_joint1_for_place(model, data, runtime, ik_controller):
        return False

    place_position, place_quaternion = ik_controller.get_end_effector_pose(frame="world")
    place_position[2] = PLACE_EE_Z_M
    if not _solve_and_apply(
        stage="lower to place",
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        position=place_position,
        quaternion=place_quaternion,
        gripper=CLOSED_GRIPPER,
        position_tol=PLACE_POSITION_TOL,
    ):
        return False

    if not _open_gripper(model, data, runtime, ik_controller):
        return False

    place_lift_position = place_position.copy()
    place_lift_position[2] += LIFT_OFFSET_M
    if not _solve_and_apply(
        stage="lift after release",
        model=model,
        data=data,
        runtime=runtime,
        ik_controller=ik_controller,
        position=place_lift_position,
        quaternion=place_quaternion,
        gripper=OPEN_GRIPPER,
        position_tol=LIFT_POSITION_TOL,
    ):
        return False

    return _return_home(model, data, runtime, ik_controller)


def execute_pick_place_sequence(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    ik_controller: PiperIKController,
    grasp_result: Any,
) -> bool:
    return execute_pick_place_sequence_with_runtime(
        model,
        data,
        ViewerMotionRuntime(viewer),
        ik_controller,
        grasp_result,
    )
