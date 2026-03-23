#Basic integration of a TFMini Plus Lidar with the depth algorithm. The drone will stop and search for a better path when TFMINI_RANGE_THRESHOLD is < a value range. 

#Usage for SITL: python basic_pipelines/depth4d_4x4_mavlink.py --input rtsp://192.168.68.105:8554/live/flightgear --arch hailo8 --use-frame
#Usage for camera: python basic_pipelines/depth4d_4x4_mavlink_latest1.py  --arch hailo8 --use-frame -i rpi 

#need to switch to GUIDED mode, from the RC or SITL (mode GUIDED) before this script will be enabled. 

#use --bar to calibrate the barometer at startup

import sys  # Added for clean exit
import argparse
import gi
import numpy as np
import hailo
import cv2
import time
import math
from pymavlink import mavutil
from collections import deque
from hailo_apps_infra.hailo_rpi_common import app_callback_class
from hailo_apps_infra.depth_pipeline import GStreamerDepthApp
from gi.repository import GLib, Gst
import threading
import shutil

try:
    import serial
except Exception:
    serial = None

# Required for GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst


# ---------------- MAVLink Setup (SITL) ----------------
#master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
#master.wait_heartbeat()
#print("Connected to system")

# ---------------- MAVLink Setup (Real drone) ----------------
master = mavutil.mavlink_connection('/dev/serial0', baud=921600, autoreconnect=True)
master.wait_heartbeat()
print(f"Connected to system: sysid={master.target_system}, compid={master.target_component}")


# ---------------- Terminal HUD helpers ----------------
def fmt(val, unit="", prec=2):
    return f"{val:.{prec}f}{unit}" if val is not None else "?"


def update_line(text: str):
    """
    Update a single terminal line in-place (no scrolling).
    Also trims to terminal width to prevent line-wrapping (which looks like scrolling).
    """
    cols = shutil.get_terminal_size((120, 20)).columns
    text = text[:max(0, cols - 1)]  # prevent wrap
    sys.stdout.write("\r\033[2K" + text)  # \r start line, \033[2K clear line
    sys.stdout.flush()


# ---------------- Request telemetry messages ----------------
def set_msg_interval(msg_id: int, hz: float):
    interval_us = int(1e6 / hz) if hz > 0 else -1
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0
    )


def safe_mode_string(hb_msg):
    try:
        return mavutil.mode_string_v10(hb_msg)
    except Exception:
        return None


def extract_runtime_flags(argv):
    """
    Parse only script-specific flags and leave the rest for the Hailo app parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Run preflight barometer calibration before starting the vision app."
    )
    flags, remaining = parser.parse_known_args(argv[1:])
    return flags, [argv[0], *remaining]


# ---------------- Telemetry cache (ONLY reader thread updates these) ----------------
telemetry_lock = threading.Lock()
stop_mav_thread = False
stop_tfmini_thread = False

speed_ms = None
battery_v = None
rel_alt_m = None
roll_rad = None
pitch_rad = None
yaw_rad = None
mode_str = None
sats = None
hb_armed = False
last_hb_time = 0.0
landed_state = None  # EXTENDED_SYS_STATE.landed_state (optional)
tfmini_distance_m = None
tfmini_last_update = 0.0


# ---------------- Telemetry display (NO recv_match here) ----------------
last_telemetry_print = 0.0
telemetry_print_rate_hz = 5.0
telemetry_print_period = 1.0 / telemetry_print_rate_hz


def update_telemetry_display():
    global last_telemetry_print

    now = time.time()
    if now - last_telemetry_print < telemetry_print_period:
        return
    last_telemetry_print = now

    with telemetry_lock:
        spd = speed_ms
        bat = battery_v
        alt = rel_alt_m
        mode = mode_str
        s = sats
        tfmini_dist = tfmini_distance_m
        tfmini_ts = tfmini_last_update

    speed_kmh = spd * 3.6 if spd is not None else None
    tfmini_fresh = tfmini_dist is not None and (now - tfmini_ts) <= TFMINI_STALE_TIMEOUT_S
    tfmini_text = f"{tfmini_dist:.2f}m" if tfmini_fresh else "?"

    line = (
        f"Spd:{fmt(speed_kmh,'km/h')}  "
        f"Bat:{fmt(bat,'V')}  "
        f"Alt:{fmt(alt,'m')}  "
        f"Mode:{mode or '?'}  "
        f"Sats:{s if s is not None else '?'}  "
        f"TFmini:{tfmini_text}"
    )
    update_line(line)


# ---------------- MAVLink Reader Thread (ONLY place calling recv_match) ----------------
def mavlink_reader():
    global speed_ms, battery_v, rel_alt_m, roll_rad, pitch_rad, yaw_rad
    global mode_str, sats, hb_armed, last_hb_time, landed_state
    global stop_mav_thread

    # Request key messages at steady rates (ArduPilot)
    try:
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 2)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 5)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 15)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2)
    except Exception:
        pass

    while not stop_mav_thread:
        msg = master.recv_match(blocking=True, timeout=1)
        if not msg:
            continue
        if msg.get_type() == "BAD_DATA":
            continue

        mtype = msg.get_type()

        with telemetry_lock:
            if mtype == "HEARTBEAT":
                mode_str = safe_mode_string(msg)
                last_hb_time = time.time()
                hb_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

            elif mtype == "VFR_HUD":
                gs = getattr(msg, "groundspeed", None)
                if gs is not None:
                    speed_ms = gs

            elif mtype == "GPS_RAW_INT":
                sats = getattr(msg, "satellites_visible", sats)
                vel = getattr(msg, "vel", None)
                if vel is not None and vel != 0xFFFF and speed_ms is None:
                    speed_ms = vel / 100.0

            elif mtype == "SYS_STATUS":
                vb = getattr(msg, "voltage_battery", None)
                if vb is not None and vb != 0xFFFF:
                    battery_v = vb / 1000.0

            elif mtype == "GLOBAL_POSITION_INT":
                ra = getattr(msg, "relative_alt", None)
                if ra is not None:
                    rel_alt_m = ra / 1000.0

            elif mtype == "ATTITUDE":
                roll = getattr(msg, "roll", None)
                pitch = getattr(msg, "pitch", None)
                yaw = getattr(msg, "yaw", None)
                if roll is not None:
                    roll_rad = roll
                if pitch is not None:
                    pitch_rad = pitch
                if yaw is not None:
                    yaw_rad = yaw

            elif mtype == "EXTENDED_SYS_STATE":
                landed_state = getattr(msg, "landed_state", landed_state)


def read_tfmini_distance_m(sensor):
    """
    Parse one TFmini+ frame:
    0x59 0x59 Dist_L Dist_H Strength_L Strength_H Temp_L Temp_H Checksum
    """
    if sensor is None:
        return None

    first = sensor.read(1)
    if len(first) != 1 or first[0] != 0x59:
        return None

    second = sensor.read(1)
    if len(second) != 1 or second[0] != 0x59:
        return None

    payload = sensor.read(7)
    if len(payload) != 7:
        return None

    frame = bytes([0x59, 0x59]) + payload
    checksum = sum(frame[:8]) & 0xFF
    if checksum != frame[8]:
        return None

    distance_cm = frame[2] + frame[3] * 256
    if distance_cm <= 0:
        return None

    return distance_cm / 100.0


def tfmini_reader():
    global tfmini_distance_m, tfmini_last_update, stop_tfmini_thread

    if serial is None:
        print("[TFmini] pyserial not installed. TFmini disabled.")
        return

    try:
        sensor = serial.Serial(TFMINI_PORT, TFMINI_BAUD, timeout=TFMINI_TIMEOUT_S)
    except Exception as exc:
        print(f"[TFmini] Failed to open {TFMINI_PORT}: {exc}. TFmini disabled.")
        return

    print(f"[TFmini] Connected on {TFMINI_PORT} @ {TFMINI_BAUD} baud.")

    try:
        while not stop_tfmini_thread:
            dist_m = read_tfmini_distance_m(sensor)
            if dist_m is None:
                time.sleep(TFMINI_POLL_S)
                continue

            with telemetry_lock:
                tfmini_distance_m = dist_m
                tfmini_last_update = time.time()

            time.sleep(TFMINI_POLL_S)
    finally:
        try:
            sensor.close()
        except Exception:
            pass


def get_tfmini_distance_for_logic():
    now = time.time()
    with telemetry_lock:
        dist = tfmini_distance_m
        ts = tfmini_last_update

    if dist is None or (now - ts) > TFMINI_STALE_TIMEOUT_S:
        return None
    if dist > TFMINI_EVAL_MAX_M:
        return None
    return dist


# ---------------- Parameters ----------------
WALL_DEPTH_THRESHOLD = 34 # 36/38 for sim
MIN_SAFE_DEPTH_FOR_FLIGHT = 28 # 32 for sim
SMOOTHING_WINDOW = 30
TAKEOFF_ALTITUDE = 1  #meters
YAW_RATE = 0.05

searching_for_direction = False
search_start_time = None
search_timeout = 15

recent_directions = deque(maxlen=SMOOTHING_WINDOW)
armed_and_airborne = False

DEPTH_RANGE = 7
decay_rate = 1.5


#change to lower values for real flight (half is a good start)
LATERAL_P_GAIN = 0.15
VERTICAL_P_GAIN = 0.55

FWD_MIN_SPEED = 0.5
FWD_MAX_SPEED = 1.5  #in m/s -> 3.5 is around 10km/h
MAX_ASCENT_SPEED = 1.4
MAX_DESCENT_SPEED = 0.7

# ---------------- TFmini+ Parameters ----------------
TFMINI_PORT = "/dev/ttyAMA2"
TFMINI_BAUD = 115200
TFMINI_TIMEOUT_S = 0.05
TFMINI_POLL_S = 0.005
TFMINI_RANGE_THRESHOLD = 2.0
TFMINI_EVAL_MAX_M = 12.0
TFMINI_STALE_TIMEOUT_S = 1.0

last_tfmini_trigger_log = 0.0
BARO_CAL_ACK_TIMEOUT_S = 8.0


# ---------------- Safe shutdown ----------------
is_shutting_down = False
shutdown_lock = threading.Lock()


def safe_shutdown(app_instance):
    global is_shutting_down, stop_mav_thread, stop_tfmini_thread
    with shutdown_lock:
        if is_shutting_down:
            return
        is_shutting_down = True

    print("\n[CONTROL] Safely shutting down...")
    stop_mav_thread = True
    stop_tfmini_thread = True

    if app_instance:
        try:
            app_instance.shutdown()
        except Exception:
            pass

    try:
        master.close()
    except Exception:
        pass


# ---------------- RC flight mode override (NO recv_match here) ----------------
def check_rc_override(app_instance):
    global is_shutting_down
    if is_shutting_down:
        return False

    with telemetry_lock:
        mode = mode_str

    if mode in ['STABILIZE', 'LAND', 'ALT_HOLD' , "LOITER"]:
        print(f"\n[CONTROL] Override: {mode} detected.")
        GLib.idle_add(safe_shutdown, app_instance)
        return False

    return True


# ---------------- Mode helper (NO recv_match) ----------------
MODE_MAP = None


def init_mode_mapping():
    """
    Must be called BEFORE starting the reader thread, while it's safe.
    """
    global MODE_MAP
    try:
        MODE_MAP = master.mode_mapping()  # uses internal knowledge based on heartbeat/autopilot
    except Exception:
        MODE_MAP = None


def set_mode_no_recv(mode_name: str) -> bool:
    """
    Sends SET_MODE without any recv_match/waits.
    Then we confirm by watching cached mode_str.
    """
    if not MODE_MAP or mode_name not in MODE_MAP:
        # fallback: try master.set_mode (may use recv internally; avoid if possible)
        try:
            master.set_mode(mode_name)
            return True
        except Exception:
            return False

    custom_mode = MODE_MAP[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode
    )
    return True


def wait_for_mode_cached(mode_name: str, wait_s=10) -> bool:
    start = time.time()
    set_mode_no_recv(mode_name)

    while time.time() - start < wait_s:
        with telemetry_lock:
            m = mode_str
        if m == mode_name:
            return True
        time.sleep(0.1)
    return False


def wait_for_armed_cached(wait_s=15) -> bool:
    start = time.time()
    while time.time() - start < wait_s:
        with telemetry_lock:
            armed = hb_armed
        if armed:
            return True
        time.sleep(0.1)
    return False


def get_rel_alt_cached():
    with telemetry_lock:
        return rel_alt_m


def get_attitude_cached():
    with telemetry_lock:
        return roll_rad, pitch_rad, yaw_rad


def describe_mav_result(result_code):
    result_map = {}
    for name in (
        "MAV_RESULT_ACCEPTED",
        "MAV_RESULT_TEMPORARILY_REJECTED",
        "MAV_RESULT_DENIED",
        "MAV_RESULT_UNSUPPORTED",
        "MAV_RESULT_FAILED",
        "MAV_RESULT_IN_PROGRESS",
        "MAV_RESULT_CANCELLED",
    ):
        value = getattr(mavutil.mavlink, name, None)
        if value is not None:
            result_map[value] = name.removeprefix("MAV_RESULT_")
    return result_map.get(result_code, f"UNKNOWN({result_code})")


def wait_for_command_ack(command_id, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
        if msg is None:
            continue

        if getattr(msg, "command", None) != command_id:
            continue

        return msg

    return None


def calibrate_barometer(wait_ack_s=BARO_CAL_ACK_TIMEOUT_S):
    """
    Request preflight ground-pressure calibration.
    MAV_CMD_PREFLIGHT_CALIBRATION param3=1 is the standard barometer calibration trigger.
    """
    command_id = mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION
    print("[BARO] Requesting preflight barometer calibration...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command_id,
        0,
        0,  # param1: gyro/gyro temp
        0,  # param2: magnetometer
        1,  # param3: ground pressure / barometer
        0,  # param4: RC
        0,  # param5: accelerometer
        0,  # param6: compmot/airspeed
        0,  # param7: ESC/baro temp
    )

    ack = wait_for_command_ack(command_id, wait_ack_s)
    if ack is None:
        print(f"[BARO] No COMMAND_ACK received within {wait_ack_s:.1f}s.")
        return False

    result = getattr(ack, "result", None)
    result_text = describe_mav_result(result)
    if result in (
        mavutil.mavlink.MAV_RESULT_ACCEPTED,
        mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
    ):
        print(f"[BARO] Calibration command acknowledged: {result_text}.")
        time.sleep(1.0)
        return True

    print(
        f"[BARO] Calibration command rejected: {result_text}. "
        "Vehicle must be in preflight conditions and kept still."
    )
    return False


# ---------------- HUD Drawing ----------------
HUD_GREEN = (90, 255, 110)
HUD_GREEN_SOFT = (70, 170, 90)
HUD_AMBER = (0, 215, 255)
HUD_RED = (80, 80, 255)
HUD_PANEL = (20, 45, 20)


def blend_overlay(frame, alpha, drawer):
    overlay = frame.copy()
    drawer(overlay)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def rotate_points(points, angle_deg, origin):
    ox, oy = origin
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = []

    for x, y in points:
        dx = x - ox
        dy = y - oy
        rx = ox + dx * cos_a - dy * sin_a
        ry = oy + dx * sin_a + dy * cos_a
        rotated.append((int(round(rx)), int(round(ry))))

    return rotated


def draw_depth_grid(canvas, grid, tile_w, tile_h, final_idx_for_vis, dir_idx, is_path_safe_to_fly):
    height, width = canvas.shape[:2]

    for i in range(1, 5):
        y = i * tile_h
        cv2.line(canvas, (0, y), (width, y), HUD_GREEN_SOFT, 1, cv2.LINE_AA)
    for j in range(1, 5):
        x = j * tile_w
        cv2.line(canvas, (x, 0), (x, height), HUD_GREEN_SOFT, 1, cv2.LINE_AA)

    for i in range(5):
        for j in range(5):
            x1, y1 = j * tile_w, i * tile_h
            val = f"{grid[i, j]:.1f}"
            text_origin = (x1 + 6, y1 + 18)

            cv2.putText(
                canvas,
                val,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0,0,0),
                1,
                cv2.LINE_AA
            )


def draw_grid_highlights(canvas, tile_w, tile_h, final_idx_for_vis, dir_idx, is_path_safe_to_fly):
    if is_path_safe_to_fly and final_idx_for_vis is not None:
        row, col = final_idx_for_vis
        x1, y1 = col * tile_w, row * tile_h
        cv2.rectangle(canvas, (x1, y1), (x1 + tile_w, y1 + tile_h), HUD_GREEN, 2, cv2.LINE_AA)
    elif dir_idx is not None:
        row, col = dir_idx
        x1, y1 = col * tile_w, row * tile_h
        cv2.rectangle(canvas, (x1, y1), (x1 + tile_w, y1 + tile_h), HUD_AMBER, 2, cv2.LINE_AA)


def draw_horizon_backdrop(canvas):
    height, width = canvas.shape[:2]
    center = (width // 2, int(height * 0.52))
    radius = int(min(width, height) * 0.26)
    cv2.circle(canvas, center, radius, HUD_PANEL, -1, cv2.LINE_AA)


def draw_artificial_horizon(canvas, roll_value, pitch_value):
    height, width = canvas.shape[:2]
    cx = width // 2
    cy = int(height * 0.52)
    pitch_deg = math.degrees(pitch_value) if pitch_value is not None else 0.0
    roll_deg = math.degrees(roll_value) if roll_value is not None else 0.0

    pitch_deg = float(np.clip(pitch_deg, -30.0, 30.0))
    roll_deg = float(np.clip(roll_deg, -70.0, 70.0))

    pixels_per_deg = height / 55.0
    rotation_deg = -roll_deg
    center = (cx, cy)

    for mark_deg in range(-20, 25, 5):
        line_y = cy + (pitch_deg - mark_deg) * pixels_per_deg
        if line_y < -30 or line_y > height + 30:
            continue

        half_len = 55 if mark_deg == 0 else (34 if mark_deg % 10 == 0 else 20)
        thickness = 2 if mark_deg == 0 else 1
        line_points = rotate_points(
            [(cx - half_len, line_y), (cx + half_len, line_y)],
            rotation_deg,
            center
        )
        cv2.line(canvas, line_points[0], line_points[1], HUD_GREEN, thickness, cv2.LINE_AA)

        if mark_deg != 0 and mark_deg % 10 == 0:
            label = f"{abs(mark_deg)}"
            left_label = rotate_points([(cx - half_len - 18, line_y + 4)], rotation_deg, center)[0]
            right_label = rotate_points([(cx + half_len + 5, line_y + 4)], rotation_deg, center)[0]
            cv2.putText(canvas, label, left_label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_GREEN, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_GREEN, 1, cv2.LINE_AA)

    bank_radius = int(min(width, height) * 0.18)
    arc_center = (cx, cy - 6)
    cv2.ellipse(canvas, arc_center, (bank_radius, bank_radius), 0, 210, 330, HUD_GREEN_SOFT, 1, cv2.LINE_AA)
    for bank_mark in (-60, -45, -30, -20, -10, 10, 20, 30, 45, 60):
        tick_angle = math.radians(bank_mark - 90)
        x_outer = arc_center[0] + bank_radius * math.cos(tick_angle)
        y_outer = arc_center[1] + bank_radius * math.sin(tick_angle)
        tick_len = 9 if abs(bank_mark) in (30, 60) else 6
        x_inner = arc_center[0] + (bank_radius - tick_len) * math.cos(tick_angle)
        y_inner = arc_center[1] + (bank_radius - tick_len) * math.sin(tick_angle)
        cv2.line(
            canvas,
            (int(round(x_inner)), int(round(y_inner))),
            (int(round(x_outer)), int(round(y_outer))),
            HUD_GREEN_SOFT,
            1,
            cv2.LINE_AA
        )

    aircraft_left = [(cx - 36, cy), (cx - 12, cy), (cx - 5, cy + 6)]
    aircraft_right = [(cx + 36, cy), (cx + 12, cy), (cx + 5, cy + 6)]
    cv2.polylines(canvas, [np.array(aircraft_left, dtype=np.int32)], False, HUD_GREEN, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(aircraft_right, dtype=np.int32)], False, HUD_GREEN, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, HUD_GREEN, -1, cv2.LINE_AA)


def draw_altitude_tape_backdrop(canvas):
    height, width = canvas.shape[:2]
    tape_width = 54
    x2 = width - 10
    x1 = x2 - tape_width
    top = 18
    bottom = height - 18
    center_y = height // 2

    cv2.rectangle(canvas, (x1, top), (x2, bottom), HUD_PANEL, -1)
    cv2.rectangle(canvas, (x1 - 10, center_y - 13), (x2, center_y + 13), HUD_PANEL, -1)


def draw_altitude_tape(canvas, alt_m):
    height, width = canvas.shape[:2]
    tape_width = 54
    x2 = width - 10
    x1 = x2 - tape_width
    top = 18
    bottom = height - 18
    center_y = height // 2

    cv2.rectangle(canvas, (x1, top), (x2, bottom), HUD_GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ALT", (x1 + 8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_GREEN, 1, cv2.LINE_AA)

    marker = np.array([(x1 - 9, center_y), (x1, center_y - 8), (x1, center_y + 8)], dtype=np.int32)
    cv2.fillConvexPoly(canvas, marker, HUD_GREEN)
    cv2.rectangle(canvas, (x1, center_y - 13), (x2, center_y + 13), HUD_GREEN, 1, cv2.LINE_AA)

    if alt_m is None:
        cv2.putText(canvas, "--.-", (x1 + 4, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GREEN, 1, cv2.LINE_AA)
        return

    major_step_m = 1.0
    minor_step_m = 0.5
    pixels_per_meter = 26.0
    visible_range_m = (bottom - top) / (2.0 * pixels_per_meter)
    first_tick = math.floor((alt_m - visible_range_m) / minor_step_m) * minor_step_m
    tick = first_tick

    while tick <= alt_m + visible_range_m + 1e-6:
        y = int(round(center_y - (tick - alt_m) * pixels_per_meter))
        if top <= y <= bottom:
            major_tick = abs((tick / major_step_m) - round(tick / major_step_m)) < 1e-6
            tick_color = HUD_GREEN if major_tick else HUD_GREEN_SOFT
            tick_len = 15 if major_tick else 8
            cv2.line(canvas, (x2 - tick_len - 4, y), (x2 - 4, y), tick_color, 1, cv2.LINE_AA)

            if major_tick and abs(y - center_y) > 15:
                if abs(tick - round(tick)) < 1e-6:
                    label = f"{int(round(tick))}"
                else:
                    label = f"{tick:.1f}"
                cv2.putText(canvas, label, (x1 + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_GREEN, 1, cv2.LINE_AA)

        tick += minor_step_m

    cv2.putText(canvas, f"{alt_m:.1f}", (x1 + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GREEN, 1, cv2.LINE_AA)


def draw_status_text(canvas, status_text, status_color):
    height = canvas.shape[0]
    cv2.putText(canvas, status_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)


# ---------------- Drone Control ----------------
def arm_and_takeoff(target_alt=None, timeout_s=90, stabilize_before=False):
    global armed_and_airborne

    if target_alt is None:
        target_alt = TAKEOFF_ALTITUDE

    print(f"[TAKEOFF] Target altitude: {target_alt:.1f} m AGL")

    # Ensure we have heartbeat recently (cached)
    with telemetry_lock:
        hb_age = time.time() - last_hb_time if last_hb_time > 0 else 999

    if hb_age > 3:
        print("[TAKEOFF] WARNING: Heartbeat looks stale. Telemetry link may be down.")

    if stabilize_before:
        wait_for_mode_cached('STABILIZE', wait_s=5)

    if not wait_for_mode_cached('GUIDED', wait_s=10):
        print("[TAKEOFF] ERROR: Failed to switch to GUIDED.")
        return False
    print("[TAKEOFF] Mode GUIDED confirmed.")

    print("[TAKEOFF] Arming motors...")
    master.arducopter_arm()

    if not wait_for_armed_cached(wait_s=15):
        print("[TAKEOFF] ERROR: Arm timeout.")
        return False
    print("[TAKEOFF] Motors armed.")

    print(f"[TAKEOFF] Commanding takeoff to {target_alt:.1f} m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, float(target_alt)
    )

    ALT_TOL = max(0.15, target_alt * 0.1)
    start = time.time()

    while time.time() - start < timeout_s:
        alt = get_rel_alt_cached()
        if alt is None:
            print("[TAKEOFF] Waiting for altitude data...")
        else:
            if alt >= (target_alt - ALT_TOL):
                print(f"[TAKEOFF] Altitude reached: {alt:.2f} m (cached GLOBAL_POSITION_INT.relative_alt).")
                armed_and_airborne = True
                return True
            print(f"[TAKEOFF] Climbing... {alt:.2f} m, goal {target_alt:.1f}-{ALT_TOL:.1f} m")

        time.sleep(0.2)

    print(f"[TAKEOFF] TIMEOUT after {timeout_s}s.")
    return False


def send_ned_velocity(vx, vy, vz, mode=3576, yaw_rate=0):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        mode,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )


def fly_in_direction(direction_idx, grid):
    row, col = direction_idx
    depth = grid[row, col]

    free_depth = max(0.0, depth - WALL_DEPTH_THRESHOLD)
    t = np.clip(free_depth / DEPTH_RANGE, 0.0, 1.0)
    t = 1.0 - np.exp(-decay_rate * t)

    forward_speed = FWD_MIN_SPEED + (FWD_MAX_SPEED - FWD_MIN_SPEED) * t
    speed_scale = forward_speed / FWD_MAX_SPEED

    error_col = col - 2.0
    error_row = row - 2.0

    vx = forward_speed
    vy = LATERAL_P_GAIN * error_col * speed_scale

    raw_vz = VERTICAL_P_GAIN * error_row
    vz = np.clip(raw_vz, -MAX_ASCENT_SPEED, MAX_DESCENT_SPEED)

    send_ned_velocity(vx, vy, vz, 3527, 0)


# ---------------- Direction Logic ----------------
def choose_direction(grid):
    valid_mask = grid > WALL_DEPTH_THRESHOLD
    if not np.any(valid_mask):
        return None, None

    center_row, center_col = 2, 2
    center_depth = grid[center_row, center_col]

    threshold = 0 if center_depth <= 0 else 0.25 * center_depth

    best_overall_idx = np.unravel_index(np.argmax(np.where(valid_mask, grid, -np.inf)), grid.shape)
    best_overall_depth = grid[best_overall_idx]

    if valid_mask[center_row, center_col] and abs(best_overall_depth - center_depth) <= threshold:
        return f"R{center_row+1}C{center_col+1}", (center_row, center_col)
    else:
        return f"R{best_overall_idx[0]+1}C{best_overall_idx[1]+1}", best_overall_idx


def smoothed_direction(new_idx):
    recent_directions.append(new_idx)
    if len(recent_directions) < SMOOTHING_WINDOW:
        return new_idx

    counts = {}
    for item in recent_directions:
        counts[item] = counts.get(item, 0) + 1

    max_count = max(counts.values())
    most_frequent = [k for k, v in counts.items() if v == max_count]

    return (2, 2) if (2, 2) in most_frequent else most_frequent[0]


# ---------------- HAILO8 Callback Class ----------------
class user_app_callback_class(app_callback_class):
    def __init__(self, app_ref=None):
        super().__init__()
        self.use_frame = True
        self.frame_to_show = None
        self.app_ref = app_ref

    def draw_frame(self):
        if self.frame_to_show is not None:
            return self.frame_to_show
        return np.zeros((256, 320, 3), dtype=np.uint8)


# ---------------- Main App Callback ----------------
def app_callback(pad, info, user_data):
    # Update telemetry display (NO blocking, NO recv_match)
    update_telemetry_display()

    # RC override check (NO recv_match)
    if not check_rc_override(user_data.app_ref):
        return Gst.PadProbeReturn.HANDLED

    global armed_and_airborne
    global searching_for_direction, search_start_time
    global last_tfmini_trigger_log

    user_data.increment()
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    depth_mat = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    if len(depth_mat) == 0:
        return Gst.PadProbeReturn.OK

    depth_data = depth_mat[0].get_data()
    depth_array = np.array(depth_data)
    width, height = 320, 256

    if depth_array.size != width * height:
        return Gst.PadProbeReturn.OK

    depth_image = depth_array.reshape((height, width))
    tile_h, tile_w = height // 5, width // 5

    grid = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            tile = depth_image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            grid[i, j] = np.mean(tile)

    direction, dir_idx = choose_direction(grid)

    if not armed_and_airborne:
        success = arm_and_takeoff()
        if not success:
            print("Takeoff failed or timed out!")
            return Gst.PadProbeReturn.OK
        else:
            print("Drone is airborne and at target altitude.")

    current_time = time.time()
    tfmini_dist = get_tfmini_distance_for_logic()
    tfmini_trigger = tfmini_dist is not None and tfmini_dist <= TFMINI_RANGE_THRESHOLD

    # Step 1: Determine if the found path is actually safe to fly.
    is_path_safe_to_fly = False
    if direction is not None and not tfmini_trigger:
        best_path_depth = grid[dir_idx]
        if best_path_depth >= MIN_SAFE_DEPTH_FOR_FLIGHT:
            is_path_safe_to_fly = True
        else:
            print(f"Path {direction} found, but depth {best_path_depth:.1f} is below safe minimum {MIN_SAFE_DEPTH_FOR_FLIGHT}.")
    elif tfmini_trigger and (current_time - last_tfmini_trigger_log) > 1.0:
        print(
            f"[TFmini] Distance {tfmini_dist:.2f}m <= {TFMINI_RANGE_THRESHOLD:.2f}m. "
            "Initiating direction change."
        )
        last_tfmini_trigger_log = current_time

    # Step 2: Act based on whether the path is safe.
    smoothed_idx = None
    if is_path_safe_to_fly:
        if searching_for_direction:
            print(f"Safe direction found: {direction}. Stopping search.")
            send_ned_velocity(0, 0, 0, 1479, 0)
            searching_for_direction = False

        if armed_and_airborne:
            smoothed_idx = smoothed_direction(dir_idx)
            fly_in_direction(smoothed_idx, grid)
    else:
        if not searching_for_direction:
            print("No clear or safe direction. Starting yaw search.")
            searching_for_direction = True
            search_start_time = current_time

        if searching_for_direction:
            if current_time - search_start_time >= search_timeout:
                print("Search timed out. Hovering.")
                send_ned_velocity(0, 0, 0, 1479, 0)
                searching_for_direction = False
            else:
                send_ned_velocity(0, 0, 0, 1479, YAW_RATE)

    # Visualization
    frame_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)

    status_text = "Status: Flying"
    status_color = HUD_GREEN
    if searching_for_direction:
        status_text = "Status: Searching for path..."
        status_color = HUD_AMBER
    elif not is_path_safe_to_fly:
        status_text = "Status: Path unsafe, hovering."
        status_color = HUD_RED

    with telemetry_lock:
        roll_vis = roll_rad
        pitch_vis = pitch_rad
        alt_vis = rel_alt_m
        tfmini_vis_dist = tfmini_distance_m
        tfmini_vis_ts = tfmini_last_update
    if tfmini_vis_dist is not None and (current_time - tfmini_vis_ts) <= TFMINI_STALE_TIMEOUT_S:
        status_text += f" | TFmini {tfmini_vis_dist:.2f}m"

    final_idx_for_vis = smoothed_idx if smoothed_idx is not None else (dir_idx if dir_idx is not None else (2, 2))

    blend_overlay(frame_vis, 0.18, draw_horizon_backdrop)
    blend_overlay(frame_vis, 0.16, draw_altitude_tape_backdrop)
    blend_overlay(
        frame_vis,
        0.28,
        lambda canvas: draw_depth_grid(canvas, grid, tile_w, tile_h, final_idx_for_vis, dir_idx, is_path_safe_to_fly)
    )
    blend_overlay(
        frame_vis,
        0.75,
        lambda canvas: draw_grid_highlights(canvas, tile_w, tile_h, final_idx_for_vis, dir_idx, is_path_safe_to_fly)
    )
    blend_overlay(frame_vis, 0.82, lambda canvas: draw_artificial_horizon(canvas, roll_vis, pitch_vis))
    blend_overlay(frame_vis, 0.84, lambda canvas: draw_altitude_tape(canvas, alt_vis))
    blend_overlay(frame_vis, 0.72, lambda canvas: draw_status_text(canvas, status_text, status_color))

    if user_data.use_frame:
        user_data.frame_to_show = frame_vis.copy()

    user_data.set_frame(frame_vis)
    return Gst.PadProbeReturn.OK


# ---------------- Main Entry ----------------
def main():
    global stop_mav_thread, stop_tfmini_thread
    flags, filtered_argv = extract_runtime_flags(sys.argv)
    sys.argv = filtered_argv

    if flags.bar and not calibrate_barometer():
        print("[BARO] Startup aborted because barometer calibration was not confirmed.")
        return 1

    # IMPORTANT: init mode map BEFORE reader thread starts
    init_mode_mapping()

    # Start MAVLink reader thread (ONLY place recv_match is called)
    mav_thread = threading.Thread(target=mavlink_reader, daemon=True)
    mav_thread.start()

    tfmini_thread = threading.Thread(target=tfmini_reader, daemon=True)
    tfmini_thread.start()

    user_callback = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_callback)
    user_callback.app_ref = app

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stop_mav_thread = True
        stop_tfmini_thread = True
        print("\nShutdown complete.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
