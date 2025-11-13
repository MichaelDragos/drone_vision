#Usage for SITL: python basic_pipelines/depth4d_4x4_mavlink.py --input rtsp://192.168.68.105:8554/live/flightgear --arch hailo8 --use-frame
#Usage for camera: python basic_pipelines/depth4d_4x4_mavlink_latest1.py  --arch hailo8 --use-frame -i rpi 

import gi
import numpy as np
import hailo
import cv2
import time
from pymavlink import mavutil
from collections import deque
from hailo_apps_infra.hailo_rpi_common import app_callback_class
from hailo_apps_infra.depth_pipeline import GStreamerDepthApp

# Required for GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ---------------- MAVLink Setup simulator ----------------
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
master.wait_heartbeat()
print("Connected to system")


# ---------------- MAVLink Setup drone ----------------
#master = mavutil.mavlink_connection('/dev/serial0', baud=921600, autoreconnect=True)
#master.wait_heartbeat()
#print(f"Connected to system: sysid={master.target_system}, compid={master.target_component}")



# ---------------- Parameters ----------------
WALL_DEPTH_THRESHOLD = 38  #depth value when to change direction. Highly dependant on light conditions  
#maybe this value can be offset with a light intensity sensor -> e.g. if lighting is low, decrease the depth_thresold 
MIN_SAFE_DEPTH_FOR_FLIGHT = 32   #no flight past this value -> begin to rotate to search a better path
SMOOTHING_WINDOW = 50 #ms
TAKEOFF_ALTITUDE = 1 #meters
YAW_RATE = 0.05  # rad/s

LATERAL_P_GAIN = 0.07   # How strongly to react side-to-side
VERTICAL_P_GAIN = 0.35    # How strongly to react up/down

searching_for_direction = False
search_start_time = None
search_timeout = 10  # seconds


recent_directions = deque(maxlen=SMOOTHING_WINDOW)
armed_and_airborne = False

MIN_SPEED = 5       
DEPTH_RANGE = 7     # how much extra depth (above MIN_SAFE) to reach full speed
decay_rate = 2.0

# ---------------- Drone Control ----------------


# ---------------- Drone Control ----------------
def arm_and_takeoff(target_alt=None, timeout_s=90, stabilize_before=False):
    """Arms and takes off to target_alt (meters AGL), waiting for completion."""
    global armed_and_airborne

    if target_alt is None:
        target_alt = TAKEOFF_ALTITUDE


    def _get_rel_alt_m():

        msg = master.recv_match(
            type=['GLOBAL_POSITION_INT','LOCAL_POSITION_NED','ALTITUDE','VFR_HUD'],
            blocking=True, timeout=0.2
            )
        if not msg:
            return None, None

        mtype = msg.get_type()

        if mtype == 'GLOBAL_POSITION_INT':
            return msg.relative_alt / 1000.0, 'GLOBAL_POSITION_INT.relative_alt'

        if mtype == 'LOCAL_POSITION_NED':
            # z is meters, positive down; negative z means above origin (up)
            return -float(msg.z), 'LOCAL_POSITION_NED.z (neg up)'

        if mtype == 'ALTITUDE':
            rel = getattr(msg, 'relative', None)
            if rel is not None:
                return float(rel), 'ALTITUDE.relative'
            alt_local = getattr(msg, 'altitude_local', None)
            if alt_local is not None:
                return float(alt_local), 'ALTITUDE.altitude_local'

        if mtype == 'VFR_HUD':
            # AMSL; not strictly AGL, but acceptable fallback in SITL
            return float(msg.alt), 'VFR_HUD.alt (AMSL)'

        return None, None


    def _wait_for_mode(mode_name, wait_s=10):
        start = time.time()
        master.set_mode(mode_name)
        while time.time() - start < wait_s:
            hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if hb:
                try:
                    mode = mavutil.mode_string_v10(hb)
                except Exception:
                    mode = None
                if mode == mode_name:
                    return True
        return False

    print(f"[TAKEOFF] Target altitude: {target_alt:.1f} m AGL")

    master.wait_heartbeat()
    print("[TAKEOFF] Heartbeat OK.")

    if stabilize_before:
        _wait_for_mode('STABILIZE', wait_s=5)

    if not _wait_for_mode('GUIDED', wait_s=10):
        print("[TAKEOFF] ERROR: Failed to switch to GUIDED.")
        return False
    print("[TAKEOFF] Mode GUIDED confirmed.")

    print("[TAKEOFF] Arming motors...")
    master.arducopter_arm()
    try:
        master.motors_armed_wait()
    except Exception:
        start = time.time()
        while time.time() - start < 15:
            hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                break
        else:
            print("[TAKEOFF] ERROR: Arm timeout.")
            return False
    print("[TAKEOFF] Motors armed.")

    print(f"[TAKEOFF] Commanding takeoff to {target_alt:.1f} m...")

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, float(target_alt)
    )

    ALT_TOL = max(0.15, target_alt * 0.1)  # tolerance (10% or at least 0.15 m)
    airborne_seen = False
    source_used = None
    start = time.time()

    while time.time() - start < timeout_s:
        ess = master.recv_match(type='EXTENDED_SYS_STATE', blocking=False)
        if ess and getattr(ess, 'landed_state', None) == mavutil.mavlink.MAV_LANDED_STATE_IN_AIR:
            airborne_seen = True

        alt, src = _get_rel_alt_m()
        if alt is not None:
            source_used = src
            if alt > 0.5:
                airborne_seen = True
            if alt >= (target_alt - ALT_TOL):
                print(f"[TAKEOFF] Altitude reached: {alt:.2f} m ({src}).")
                armed_and_airborne = True
                return True

            print(f"[TAKEOFF] Climbing... {alt:.2f} m ({src}), goal {target_alt:.1f}Â±{ALT_TOL:.1f} m")
        else:
            print("[TAKEOFF] Waiting for altitude data...")

        time.sleep(0.2)

    print(f"[TAKEOFF] TIMEOUT after {timeout_s}s. Airborne={airborne_seen}, last_alt_source={source_used}")
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
    """
    Calculates drone velocity using proportional control based on the chosen grid cell.
    BODY_NED: x=forward, y=right, z=down. Negative vz = climb, positive vz = descend.
    """
    row, col = direction_idx
    depth = grid[row, col]

    # Speed scaling based on free space
    excess = max(0.0, depth - MIN_SAFE_DEPTH_FOR_FLIGHT)
    t = min(1.0, excess / DEPTH_RANGE)
    #t = t * t * (3 - 2 * t)  # smoothstep
    #t = t * t * t * (t * (t * 6 - 15) + 10)  # Ken Perlin's improved smoothstep
    
    # Exponential response: fast when far, slow when close
    t = 1.0 - np.exp(-decay_rate * t)
    
    speed_factor = MIN_SPEED + (1.0 - MIN_SPEED) * t
    base_speed = 1.0 * speed_factor

    # Error relative to center cell (2,2)
    error_col = col - 2.0   # left<0, right>0 -> vy sign matches NED y (right positive)
    error_row = row - 2.0   # up<0, down>0

    # Proportional velocities
    vx = base_speed
    vy = LATERAL_P_GAIN  * error_col * speed_factor
    # IMPORTANT: For up (error_row<0), vz must be negative in BODY_NED -> multiply directly
    vz = VERTICAL_P_GAIN * error_row * speed_factor

    # (Optional) clamp lateral/vertical speeds for safety
    # vy = max(min(vy, 1.0), -1.0)
    # vz = max(min(vz, 0.8), -0.8)

    print(f"Proportional Move: err_col={error_col:.1f}, err_row={error_row:.1f} -> vy={vy:.2f}, vz={vz:.2f}")
    print("SpeedFactor:", speed_factor)

    # Type mask 3527 keeps only velocity+yaw_rate enabled, as in your original call
    send_ned_velocity(vx, vy, vz, 3527, 0)


# ---------------- Direction Logic ----------------
def choose_direction(grid):
    valid_mask = grid > WALL_DEPTH_THRESHOLD
    if not np.any(valid_mask):
        return None, None

    # Identify the center cell
    center_row, center_col = 2, 2
    center_depth = grid[center_row, center_col]

    # Calculate the 5% depth threshold relative to the center
    # Ensure center_depth is not zero to avoid division by zero for threshold calculation
    if center_depth <= 0:
        threshold = 0  # If center depth is zero or negative, no threshold for comparison
    else:
       
        threshold = 0.15 * center_depth #was 0.08

    # Find the best direction (highest depth) among all valid regions
    best_overall_idx = np.unravel_index(np.argmax(np.where(valid_mask, grid, -np.inf)), grid.shape)
    best_overall_depth = grid[best_overall_idx]

    # Check if the best overall direction is within 5% of the center depth
    # If the center depth itself is not valid, we can't default to it based on its depth.
    if valid_mask[center_row, center_col] and abs(best_overall_depth - center_depth) <= threshold:
        # If the best direction is within 5% of the center,
        # and the center itself is a valid (not a wall) depth,
        # then prioritize the center as the chosen direction.
        return f"R{center_row+1}C{center_col+1}", (center_row, center_col)
    else:
        # Otherwise, stick with the overall best direction found
        return f"R{best_overall_idx[0]+1}C{best_overall_idx[1]+1}", best_overall_idx

def smoothed_direction(new_idx):
    recent_directions.append(new_idx)
    if len(recent_directions) < SMOOTHING_WINDOW:
        return new_idx
    # Use the center as a tie-breaker if multiple directions have the same count
    counts = {}
    for item in recent_directions:
        counts[item] = counts.get(item, 0) + 1

    max_count = 0
    most_frequent_directions = []
    for item, count in counts.items():
        if count > max_count:
            max_count = count
            most_frequent_directions = [item]
        elif count == max_count:
            most_frequent_directions.append(item)

    # If the center (2,2) is among the most frequent, prioritize it
    if (2, 2) in most_frequent_directions:
        return (2, 2)
    else:
        # Otherwise, return the first one (or you could have another tie-breaking rule)
        return most_frequent_directions[0]



# ---------------- HAILO8 Callback Class ----------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.frame_to_show = None

    def draw_frame(self):
        if self.frame_to_show is not None:
            return self.frame_to_show
        return np.zeros((256, 320, 3), dtype=np.uint8)

# ---------------- Main App Callback ----------------
def app_callback(pad, info, user_data):
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
            return Gst.PadProbeReturn.OK # Stop further processing if takeoff fails
        else:
            print("Drone is airborne and at target altitude.")

  # ###############################################################
    # ##               NEW AND IMPROVED LOGIC START                ##
    # ###############################################################

    current_time = time.time()

    # Step 1: Determine if the found path is actually safe to fly.
    is_path_safe_to_fly = False
    if direction is not None:
        best_path_depth = grid[dir_idx]
        if best_path_depth >= MIN_SAFE_DEPTH_FOR_FLIGHT:
            is_path_safe_to_fly = True
        else:
            # A path exists, but it's too close for comfort. Treat as unsafe.
            print(f"Path {direction} found, but depth {best_path_depth:.1f} is below safe minimum {MIN_SAFE_DEPTH_FOR_FLIGHT}.")

    # Step 2: Act based on whether the path is safe.
    if is_path_safe_to_fly:
        # A safe path was found, so we can fly.
        if searching_for_direction:
            print(f"Safe direction found: {direction}. Stopping search.")
            send_ned_velocity(0, 0, 0, 1479, 0)  # Stop yawing
            searching_for_direction = False

        if armed_and_airborne:
            final_idx = smoothed_direction(dir_idx)
            fly_in_direction(final_idx, grid)
    else:
        # No path was found OR the best path was not safe enough.
        # Enter/continue search mode.
        if not searching_for_direction:
            print("No clear or safe direction. Starting yaw search.")
            searching_for_direction = True
            search_start_time = current_time

        if searching_for_direction:
            if current_time - search_start_time >= search_timeout:
                print("Search timed out. Hovering.")
                send_ned_velocity(0, 0, 0, 1479, 0) # Stop yaw and hover
                searching_for_direction = False # Stop searching to avoid constant messages
            else:
                # Still searching, continue to yaw.
                send_ned_velocity(0, 0, 0, 1479, YAW_RATE)

      # ###############################################################
    # ##                NEW AND IMPROVED LOGIC END                 ##
    # ###############################################################

    # Visualization
    # Create a separate variable for visualization to prevent crashes and show current state.
    final_idx_for_vis = dir_idx if dir_idx is not None else (2, 2)
    if is_path_safe_to_fly and armed_and_airborne:
        # If we are flying, the visualized index should be the final smoothed one.
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
            
            # Color-code the chosen rectangle
            if (i, j) == final_idx_for_vis and is_path_safe_to_fly:
                # Green for a safe, active path
                cv2.rectangle(frame_vis, (x1, y1), (x1 + tile_w, y1 + tile_h), (0, 255, 0), 2)
            elif (i, j) == dir_idx and not is_path_safe_to_fly and direction is not None:
                # Orange for a path that exists but is too close (caution)
                cv2.rectangle(frame_vis, (x1, y1), (x1 + tile_w, y1 + tile_h), (0, 165, 255), 2)
            
            cv2.putText(frame_vis, val, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)


 # Display the current status on the frame
    status_text = "Status: Flying"
    status_color = (0, 255, 0) # Green
    if searching_for_direction:
        status_text = "Status: Searching for path..."
        status_color = (0, 255, 255) # Yellow
    elif not is_path_safe_to_fly:
        status_text = "Status: Path unsafe, hovering."
        status_color = (0, 0, 255) # Red

    cv2.putText(frame_vis, status_text, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if user_data.use_frame:
        user_data.frame_to_show = frame_vis.copy()

    user_data.set_frame(frame_vis)

    return Gst.PadProbeReturn.OK



# ---------------- Main Entry ----------------
def main():
    user_callback = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_callback)
    app.run()

if __name__ == '__main__':
    main()