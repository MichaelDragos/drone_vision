import sys  # Added for clean exit
import gi
import numpy as np
import hailo
import cv2
import time
from pymavlink import mavutil
from collections import deque
from hailo_apps_infra.hailo_rpi_common import app_callback_class
from hailo_apps_infra.depth_pipeline import GStreamerDepthApp
from gi.repository import GLib, Gst
import threading
import shutil

# Required for GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst


# ---------------- MAVLink Setup (SITL) ----------------
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
master.wait_heartbeat()
print("Connected to system")

# ---------------- MAVLink Setup (Real drone) ----------------
# master = mavutil.mavlink_connection('/dev/serial0', baud=921600, autoreconnect=True)
# master.wait_heartbeat()
# print(f"Connected to system: sysid={master.target_system}, compid={master.target_component}")


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


# ---------------- Telemetry cache (ONLY reader thread updates these) ----------------
telemetry_lock = threading.Lock()
stop_mav_thread = False

speed_ms = None
battery_v = None
rel_alt_m = None
mode_str = None
sats = None
hb_armed = False
last_hb_time = 0.0
landed_state = None  # EXTENDED_SYS_STATE.landed_state (optional)


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

    speed_kmh = spd * 3.6 if spd is not None else None

    line = (
        f"Spd:{fmt(speed_kmh,'km/h')}  "
        f"Bat:{fmt(bat,'V')}  "
        f"Alt:{fmt(alt,'m')}  "
        f"Mode:{mode or '?'}  "
        f"Sats:{s if s is not None else '?'}"
    )
    update_line(line)


# ---------------- MAVLink Reader Thread (ONLY place calling recv_match) ----------------
def mavlink_reader():
    global speed_ms, battery_v, rel_alt_m, mode_str, sats, hb_armed, last_hb_time, landed_state
    global stop_mav_thread

    # Request key messages at steady rates (ArduPilot)
    try:
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 2)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 5)
        set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
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

            elif mtype == "EXTENDED_SYS_STATE":
                landed_state = getattr(msg, "landed_state", landed_state)


# ---------------- Parameters ----------------
WALL_DEPTH_THRESHOLD = 36
MIN_SAFE_DEPTH_FOR_FLIGHT = 32
SMOOTHING_WINDOW = 30
TAKEOFF_ALTITUDE = 1.5
YAW_RATE = 0.05

searching_for_direction = False
search_start_time = None
search_timeout = 15

recent_directions = deque(maxlen=SMOOTHING_WINDOW)
armed_and_airborne = False

DEPTH_RANGE = 7
decay_rate = 1.5

LATERAL_P_GAIN = 0.15
VERTICAL_P_GAIN = 0.55

FWD_MIN_SPEED = 0.5
FWD_MAX_SPEED = 3.5
MAX_ASCENT_SPEED = 1.4
MAX_DESCENT_SPEED = 0.7


# ---------------- Safe shutdown ----------------
is_shutting_down = False
shutdown_lock = threading.Lock()


def safe_shutdown(app_instance):
    global is_shutting_down, stop_mav_thread
    with shutdown_lock:
        if is_shutting_down:
            return
        is_shutting_down = True

    print("\n[CONTROL] Safely shutting down...")
    stop_mav_thread = True

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

    if mode in ['STABILIZE', 'LAND', 'ALT_HOLD']:
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
            print(f"[TAKEOFF] Climbing... {alt:.2f} m, goal {target_alt:.1f}±{ALT_TOL:.1f} m")

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

    # Step 1: Determine if the found path is actually safe to fly.
    is_path_safe_to_fly = False
    if direction is not None:
        best_path_depth = grid[dir_idx]
        if best_path_depth >= MIN_SAFE_DEPTH_FOR_FLIGHT:
            is_path_safe_to_fly = True
        else:
            print(f"Path {direction} found, but depth {best_path_depth:.1f} is below safe minimum {MIN_SAFE_DEPTH_FOR_FLIGHT}.")

    # Step 2: Act based on whether the path is safe.
    if is_path_safe_to_fly:
        if searching_for_direction:
            print(f"Safe direction found: {direction}. Stopping search.")
            send_ned_velocity(0, 0, 0, 1479, 0)
            searching_for_direction = False

        if armed_and_airborne:
            final_idx = smoothed_direction(dir_idx)
            fly_in_direction(final_idx, grid)
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
    final_idx_for_vis = dir_idx if dir_idx is not None else (2, 2)
    if is_path_safe_to_fly and armed_and_airborne and dir_idx is not None:
        final_idx_for_vis = smoothed_direction(dir_idx)

    frame_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)

    for i in range(1, 5):
        cv2.line(frame_vis, (0, i * tile_h), (width, i * tile_h), (100, 100, 100), 1)
    for j in range(1, 5):
        cv2.line(frame_vis, (j * tile_w, 0), (j * tile_w, height), (100, 100, 100), 1)

    for i in range(5):
        for j in range(5):
            x1, y1 = j * tile_w, i * tile_h
            val = f"{grid[i, j]:.1f}"
            text_color = (0, 0, 0)

            if (i, j) == final_idx_for_vis and is_path_safe_to_fly:
                cv2.rectangle(frame_vis, (x1, y1), (x1 + tile_w, y1 + tile_h), (0, 255, 0), 2)
            elif (i, j) == dir_idx and not is_path_safe_to_fly and direction is not None:
                cv2.rectangle(frame_vis, (x1, y1), (x1 + tile_w, y1 + tile_h), (0, 165, 255), 2)

            cv2.putText(frame_vis, val, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    status_text = "Status: Flying"
    status_color = (0, 255, 0)
    if searching_for_direction:
        status_text = "Status: Searching for path..."
        status_color = (0, 255, 255)
    elif not is_path_safe_to_fly:
        status_text = "Status: Path unsafe, hovering."
        status_color = (0, 0, 255)

    cv2.putText(frame_vis, status_text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if user_data.use_frame:
        user_data.frame_to_show = frame_vis.copy()

    user_data.set_frame(frame_vis)
    return Gst.PadProbeReturn.OK


# ---------------- Main Entry ----------------
def main():
    global stop_mav_thread

    # IMPORTANT: init mode map BEFORE reader thread starts
    init_mode_mapping()

    # Start MAVLink reader thread (ONLY place recv_match is called)
    mav_thread = threading.Thread(target=mavlink_reader, daemon=True)
    mav_thread.start()

    user_callback = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_callback)
    user_callback.app_ref = app

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stop_mav_thread = True
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
