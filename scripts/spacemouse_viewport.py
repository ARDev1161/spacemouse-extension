import math
import omni.kit.app
import omni.kit.viewport.utility as viewport_utils
from pxr import UsdGeom, Gf, Sdf
import carb

from srl.spacemouse.spacemouse_extension import get_global_spacemouse
from srl.spacemouse.camera_settings import (
    apply_saved_camera_settings,
    save_camera_settings_if_changed,
    save_current_camera_settings,
)

# =========================
# TUNING
# =========================
TRANS_SPEED = 2.0        # units/sec at full deflection
ROT_SPEED_DEG = 120.0    # deg/sec at full deflection

# raw SpaceMouse axes: trans=[x,y,z], rot=[rx,ry,rz]
TRANS_MAP = (0, 2, 1)    # -> (tx,ty,tz)   (your mapping)
ROT_MAP   = (1, 2, 0)    # -> (rx,ry,rz)   (your mapping)

INVERT_TRANS = Gf.Vec3d( 1.0,  1.0,  1.0)
INVERT_ROT   = Gf.Vec3d(-1.0,  -1.0,  1.0)

# FPV order: yaw -> pitch -> roll
ROT_ORDER = "YPR"        # any perm of 'Y','P','R'

CAM_PATH = Sdf.Path("/World/SpacemouseCam")

DEADZONE = 1e-6
DT_MAX = 0.05            # clamp big dt on lag frames

DEBUG_EVERY_N = 60       # print once per N frames (0 disables)



def _v3(x, y, z):
    return Gf.Vec3d(float(x), float(y), float(z))


# ---------- quaternion helpers ----------
def quat_from_axis_angle(axis: Gf.Vec3d, angle_deg: float) -> Gf.Quatd:
    ax = Gf.Vec3d(axis)
    n = ax.GetLength()
    if n < 1e-12 or abs(angle_deg) < 1e-12:
        return Gf.Quatd(1.0, _v3(0, 0, 0))
    ax /= n
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    c = math.cos(half)
    return Gf.Quatd(c, _v3(ax[0] * s, ax[1] * s, ax[2] * s))


def quat_normalize(q: Gf.Quatd) -> Gf.Quatd:
    r = float(q.GetReal())
    im = q.GetImaginary()
    l2 = r * r + float(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
    if l2 < 1e-24:
        return Gf.Quatd(1.0, _v3(0, 0, 0))
    inv = 1.0 / math.sqrt(l2)
    return Gf.Quatd(r * inv, _v3(im[0] * inv, im[1] * inv, im[2] * inv))


def rotation_to_quat(rot: Gf.Rotation) -> Gf.Quatd:
    for name in ("GetQuat", "GetQuatd", "GetQuaternion"):
        if hasattr(rot, name):
            try:
                q = getattr(rot, name)()
                return Gf.Quatd(q)
            except Exception:
                pass
    return Gf.Quatd(1.0, _v3(0, 0, 0))


def compose_dq_local(pitch_deg, yaw_deg, roll_deg) -> Gf.Quatd:
    # Camera-local axes:
    # right = +X, up = +Y, forward = -Z (FPV)
    qP = quat_from_axis_angle(_v3(1, 0, 0),   pitch_deg)
    qY = quat_from_axis_angle(_v3(0, 1, 0),   yaw_deg)
    qR = quat_from_axis_angle(_v3(0, 0, -1),  roll_deg)

    def pick(ch):
        if ch == "P":
            return qP
        if ch == "Y":
            return qY
        if ch == "R":
            return qR
        raise ValueError("Bad ROT_ORDER")

    return pick(ROT_ORDER[0]) * pick(ROT_ORDER[1]) * pick(ROT_ORDER[2])


class SpaceMouseFPV:
    def __init__(self):
        self._app = omni.kit.app.get_app_interface()

        self._sp = get_global_spacemouse()
        if self._sp is None:
            raise RuntimeError("SpaceMouse is not active. Enable srl.spacemouse in the Extension Manager.")

        self._vp = viewport_utils.get_active_viewport()
        if self._vp is None:
            raise RuntimeError("No active viewport found. Click a viewport and run again.")

        self._stage = self._vp.stage
        if self._stage is None:
            raise RuntimeError("Stage is not ready yet. Wait for the stage to load and try again.")

        # capture current active camera transform before switching cameras
        active_cam_path = self._vp.camera_path
        if not active_cam_path:
            raise RuntimeError("Active camera is not ready yet. Wait and try again.")
        active_cam_prim = self._stage.GetPrimAtPath(active_cam_path)
        if not active_cam_prim.IsValid():
            raise RuntimeError("Active camera prim not found yet. Wait and try again.")
        active_cam_xf = UsdGeom.Xformable(active_cam_prim)
        M0 = active_cam_xf.ComputeLocalToWorldTransform(self._vp.time)
        active_clipping_attr = active_cam_prim.GetAttribute("clippingRange")
        active_clipping = active_clipping_attr.Get() if active_clipping_attr else None

        # ensure camera exists
        if not self._stage.GetPrimAtPath(CAM_PATH).IsValid():
            cam = UsdGeom.Camera.Define(self._stage, CAM_PATH)
            cam.GetFocalLengthAttr().Set(15.0)

        # make camera active
        self._vp.camera_path = str(CAM_PATH)

        cam_prim = self._stage.GetPrimAtPath(CAM_PATH)
        self._xformable = UsdGeom.Xformable(cam_prim)
        self._xformable.ClearXformOpOrder()
        self._op = self._xformable.AddTransformOp()

        # apply saved camera settings (focal length, apertures, clipping, etc.)
        apply_saved_camera_settings(cam_prim)
        # Always keep clipping plane behavior aligned with current active camera.
        # This avoids a too-far near clipping plane on SpacemouseCam.
        if active_clipping is not None:
            clipping_attr = cam_prim.GetAttribute("clippingRange")
            if clipping_attr:
                clipping_attr.Set(active_clipping)

        # init pose from previous viewport transform
        self._pos = M0.ExtractTranslation()
        rot0 = M0.ExtractRotation()
        self._q = quat_normalize(rotation_to_quat(rot0))

        self._frame = 0
        self._last_camera_settings = {}
        self._settings_save_stride = 30
        self._write_pose()

        stream = self._app.get_post_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update)

        carb.log_info(f"[SpaceMouseFPV] Started. camera={self._vp.camera_path}")

    def shutdown(self):
        if getattr(self, "_sub", None) is not None:
            try:
                self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None
        try:
            cam_prim = self._stage.GetPrimAtPath(CAM_PATH)
            if cam_prim.IsValid():
                save_current_camera_settings(cam_prim)
        except Exception:
            pass
        carb.log_info("[SpaceMouseFPV] Stopped")

    def _write_pose(self):
        R3 = Gf.Matrix3d(self._q)       # ok: Matrix3d(Quatd)
        M  = Gf.Matrix4d(R3, self._pos) # ok: Matrix4d(Matrix3d, Vec3d)
        self._op.Set(M, self._vp.time)

    def _on_update(self, e):
        # dt
        try:
            dt = float(e.payload.get("dt", 1.0 / 60.0))
        except Exception:
            dt = 1.0 / 60.0
        if dt <= 0.0:
            return
        if dt > DT_MAX:
            dt = DT_MAX

        stamp, trans, rot, buttons = self._sp.get_controller_state()

        sx = [float(trans[0]), float(trans[1]), float(trans[2])]
        sr = [float(rot[0]),   float(rot[1]),   float(rot[2])]

        tx, ty, tz = sx[TRANS_MAP[0]], sx[TRANS_MAP[1]], sx[TRANS_MAP[2]]
        rx, ry, rz = sr[ROT_MAP[0]],   sr[ROT_MAP[1]],   sr[ROT_MAP[2]]

        tx, ty, tz = tx * INVERT_TRANS[0], ty * INVERT_TRANS[1], tz * INVERT_TRANS[2]
        rx, ry, rz = rx * INVERT_ROT[0],   ry * INVERT_ROT[1],   rz * INVERT_ROT[2]

        if abs(tx) + abs(ty) + abs(tz) + abs(rx) + abs(ry) + abs(rz) < DEADZONE:
            return

        # ---- rotation (FPV body/local): q = q * dq ----
        pitch = rx * ROT_SPEED_DEG * dt
        yaw   = ry * ROT_SPEED_DEG * dt
        roll  = rz * ROT_SPEED_DEG * dt

        dq = compose_dq_local(pitch, yaw, roll)
        self._q = quat_normalize(self._q * dq)

        # ---- translation in camera-local axes ----
        # FPV: forward is -Z, so push tz into -Z explicitly
        local_move = _v3(tx, ty, -tz) * (TRANS_SPEED * dt)

        # Convert local direction -> world direction safely
        R3 = Gf.Matrix3d(self._q)
        R4 = Gf.Matrix4d(R3, _v3(0, 0, 0))  # rotation-only matrix
        world_move = R4.TransformDir(local_move)

        self._pos = self._pos + world_move

        self._write_pose()

        # optional debug
        self._frame += 1
        if self._frame % self._settings_save_stride == 0:
            try:
                cam_prim = self._stage.GetPrimAtPath(CAM_PATH)
                if cam_prim.IsValid():
                    self._last_camera_settings = save_camera_settings_if_changed(
                        cam_prim, self._last_camera_settings
                    )
            except Exception:
                pass
        if DEBUG_EVERY_N and (self._frame % DEBUG_EVERY_N == 0):
            # camera forward in world (local -Z)
            fwd_world = R4.TransformDir(_v3(0, 0, -1))
            carb.log_info(f"[FPV] local_move={local_move} world_move={world_move} fwd_world={fwd_world}")



# restart cleanly
if "SPACEMOUSE_FPV" in globals():
    try:
        SPACEMOUSE_FPV.shutdown()
    except Exception:
        pass

SPACEMOUSE_FPV = SpaceMouseFPV()
