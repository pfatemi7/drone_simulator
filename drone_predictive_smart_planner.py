import cv2
import numpy as np
import math
from collections import deque

WIDTH, HEIGHT = 800, 600
DT = 0.06

DRONE_LENGTH = 20
DRONE_WIDTH  = 12

MAX_YAW_RATE  = 4.0
MAX_YAW_ACCEL = 2.0
ANGLE_SMOOTH  = 0.25

BASE_SPEED = 3.5
MIN_SPEED  = 1.5
MAX_SPEED  = 5.5

K_GOAL   = 1.2
K_REP    = 400.0
REP_CUTOFF = 140.0
EPS_DIST = 1e-6

NEAR_ON   = 60.0
NEAR_OFF  = 85.0
WALL_TANG_GAIN = 0.9
WALL_NORM_PUSH = 0.3

TIGHT_SLOW = 55.0

SPAWN_GOAL_DIST_MIN = 220
SPAWN_GOAL_DIST_MAX = 320
GOAL_LATERAL_JITTER = 80
OBST_AHEAD_DIST_MIN = 180
OBST_AHEAD_DIST_MAX = 420
OBST_SPAWN_COOLDOWN = 1.8
OBST_W_MIN, OBST_W_MAX = 18, 36
OBST_H_MIN, OBST_H_MAX = 120, 180
PRUNE_OBST_BEHIND_DIST = 700

SPAWN_ON_GOAL = 3
GOAL_CYCLE = 0

recent_positions = []
circling_detected = False

GOAL_SLOW_RADIUS = 80.0
GOAL_CAPTURE_DIST = 12.0
GOAL_CAPTURE_HEAD = 15.0
TURN_IN_PLACE_ERR = 45.0
TURN_IN_PLACE_SPEED = 1.5
HE_ERR_SLOW_K = 0.08

BASE_YAW_RATE = 4.0
BASE_YAW_ACCEL = 2.0
GOAL_YAW_RATE = 12.0
GOAL_YAW_ACCEL = 6.0
NEAR_GOAL_YAW_RATE = 16.0
NEAR_GOAL_YAW_ACCEL = 8.0

obstacles = [
    (200, 200, 600, 230),
    (100, 300, 120, 500),
    (680, 100, 700, 400),
    (350, 350, 450, 370),
]

R_BODY = max(18.0, 0.5 * ((DRONE_LENGTH**2 + DRONE_WIDTH**2) ** 0.5))

def inflate_rect(x1, y1, x2, y2, r):
    return (int(x1 - r), int(y1 - r), int(x2 + r), int(y2 + r))

OBST_INF = [inflate_rect(*r, R_BODY) for r in obstacles]

X_MIN, Y_MIN = R_BODY, R_BODY
X_MAX, Y_MAX = WIDTH - R_BODY, HEIGHT - R_BODY

def spawn_goal_ahead(pos, heading_rad):
    fwd = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
    base = pos + fwd * np.random.uniform(SPAWN_GOAL_DIST_MIN, SPAWN_GOAL_DIST_MAX)
    jitter = left * np.random.uniform(-GOAL_LATERAL_JITTER, GOAL_LATERAL_JITTER)
    cand = base + jitter
    cand[0] = np.clip(cand[0], X_MIN, X_MAX)
    cand[1] = np.clip(cand[1], Y_MIN, Y_MAX)
    return cand

def check_corridor_overlap(rect, pos, goal):
    x1, y1, x2, y2 = rect
    corridor_width = 12.5
    
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return False
    
    unit_x = dx / length
    unit_y = dy / length
    perp_x = -unit_y
    perp_y = unit_x
    
    rect_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    to_rect = rect_center - pos
    
    proj_along = to_rect[0] * unit_x + to_rect[1] * unit_y
    proj_perp = to_rect[0] * perp_x + to_rect[1] * perp_y
    
    if proj_along < 0 or proj_along > length:
        return False
    
    return abs(proj_perp) < corridor_width + max(x2-x1, y2-y1) / 2

def spawn_obstacle_ahead(pos, heading_rad, active_goal):
    for _ in range(5):
        fwd = np.array([math.cos(heading_rad), math.sin(heading_rad)])
        left = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
        
        dist = np.random.uniform(OBST_AHEAD_DIST_MIN, OBST_AHEAD_DIST_MAX)
        lateral = np.random.uniform(-90, 90)
        
        center = pos + fwd * dist + left * lateral
        
        if np.random.random() < 0.5:
            w = np.random.uniform(OBST_W_MIN, OBST_W_MAX)
            h = np.random.uniform(OBST_H_MIN, OBST_H_MAX)
        else:
            h = np.random.uniform(OBST_W_MIN, OBST_W_MAX)
            w = np.random.uniform(OBST_H_MIN, OBST_H_MAX)
        
        x1 = center[0] - w/2
        y1 = center[1] - h/2
        x2 = center[0] + w/2
        y2 = center[1] + h/2
        
        if x1 < X_MIN or x2 > X_MAX or y1 < Y_MIN or y2 > Y_MAX:
            continue
            
        rect = (int(x1), int(y1), int(x2), int(y2))
        
        if check_corridor_overlap(rect, pos, active_goal):
            continue
            
        drone_dist = distance_point_rect(pos[0], pos[1], x1, y1, x2, y2)[0]
        if drone_dist < R_BODY + 5:
            continue
            
        overlap = False
        for obs in obstacles:
            ox1, oy1, ox2, oy2 = obs
            if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                overlap = True
                break
                
        if not overlap:
            return rect
    
    return None

goals = deque([])
pos   = np.array([WIDTH//2, HEIGHT-60], dtype=float)
theta = math.radians(-90)
yaw_rate = 0.0
target_heading_deg = math.degrees(theta)
path_pts = []
wall_follow = None
goal_reached = False
simulation_running = True

def angnorm(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def get_triangle(p, ang):
    front = p + DRONE_LENGTH * np.array([math.cos(ang), math.sin(ang)])
    left  = p + DRONE_WIDTH  * np.array([math.cos(ang + math.radians(130)), math.sin(ang + math.radians(130))])
    right = p + DRONE_WIDTH  * np.array([math.cos(ang - math.radians(130)), math.sin(ang - math.radians(130))])
    return np.array([front, left, right], dtype=int)

def draw_world(img):
    img[:] = 255
    for x1, y1, x2, y2 in obstacles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    for i, goal_pos in enumerate(goals):
        if i == 0:
            cv2.circle(img, goal_pos.astype(int), 10, (255, 0, 0), -1)
        else:
            alpha = max(0.3, 1.0 - i * 0.3)
            color = (int(255 * alpha), 0, 0)
            cv2.circle(img, goal_pos.astype(int), 6, color, -1)

def draw_start_stop_button(img, is_running):
    button_x, button_y = WIDTH - 120, 20
    button_w, button_h = 100, 40
    
    color = (0, 255, 0) if is_running else (0, 0, 255)
    cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h), color, -1)
    cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 0, 0), 2)
    
    text = "STOP" if is_running else "START"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = button_x + (button_w - text_size[0]) // 2
    text_y = button_y + (button_h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return (button_x, button_y, button_x + button_w, button_y + button_h)

def draw_hud(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    hud_text = f"Cycle: {GOAL_CYCLE} | Obstacles: {len(obstacles)}"
    cv2.putText(img, hud_text, (10, 30), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, hud_text, (10, 30), font, font_scale, (255, 255, 255), thickness)
    
    if len(goals) > 0:
        active_goal = goals[0]
        distance_to_goal = np.linalg.norm(pos - active_goal)
        current_heading_deg = math.degrees(theta)
        field, _, _ = compute_field(pos, theta, wall_follow, active_goal)
        desired_heading = desired_heading_from_field(field)
        heading_err = heading_error_deg(current_heading_deg, desired_heading)
        
        yaw_info = f"Dist: {distance_to_goal:.1f} | Err: {heading_err:.1f}° | Yaw: {yaw_rate:.1f}°/f"
        cv2.putText(img, yaw_info, (10, 50), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, yaw_info, (10, 50), font, font_scale, (255, 255, 255), thickness)

def handle_mouse_click(event, x, y, flags, param):
    global simulation_running
    if event == cv2.EVENT_LBUTTONDOWN:
        button_x, button_y = WIDTH - 120, 20
        button_w, button_h = 100, 40
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            simulation_running = not simulation_running
            print(f"Simulation {'STARTED' if simulation_running else 'STOPPED'}")

def draw_drone(img, p, ang):
    if not hasattr(draw_drone, 'drone_img'):
        draw_drone.drone_img = cv2.imread('drone.png', cv2.IMREAD_UNCHANGED)
        if draw_drone.drone_img is None:
            print("Warning: Could not load drone.png, falling back to triangle")
            draw_drone.drone_img = None
    
    if draw_drone.drone_img is None:
        tri = get_triangle(p, ang)
        cv2.drawContours(img, [tri], 0, (0, 255, 0), -1)
        c = np.mean(tri, axis=0).astype(int)
        cv2.circle(img, tuple(c), int(R_BODY), (0, 255, 255), 1)
        return
    
    drone_h, drone_w = draw_drone.drone_img.shape[:2]
    
    target_width = int(DRONE_LENGTH * 2)
    target_height = int(DRONE_WIDTH * 2)
    
    drone_resized = cv2.resize(draw_drone.drone_img, (target_width, target_height))
    
    center = (target_width // 2, target_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, math.degrees(ang), 1.0)
    
    drone_rotated = cv2.warpAffine(drone_resized, rotation_matrix, (target_width, target_height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    x_offset = int(p[0] - target_width // 2)
    y_offset = int(p[1] - target_height // 2)
    
    if drone_rotated.shape[2] == 4:
        alpha = drone_rotated[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        drone_rgb = drone_rotated[:, :, :3]
        
        x1, y1 = max(0, x_offset), max(0, y_offset)
        x2, y2 = min(WIDTH, x_offset + target_width), min(HEIGHT, y_offset + target_height)
        
        if x1 < x2 and y1 < y2:
            drone_x1 = max(0, -x_offset)
            drone_y1 = max(0, -y_offset)
            drone_x2 = drone_x1 + (x2 - x1)
            drone_y2 = drone_y1 + (y2 - y1)
            
            roi = img[y1:y2, x1:x2]
            drone_roi = drone_rgb[drone_y1:drone_y2, drone_x1:drone_x2]
            alpha_roi = alpha[drone_y1:drone_y2, drone_x1:drone_x2]
            
            blended = roi * (1 - alpha_roi) + drone_roi * alpha_roi
            img[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        x1, y1 = max(0, x_offset), max(0, y_offset)
        x2, y2 = min(WIDTH, x_offset + target_width), min(HEIGHT, y_offset + target_height)
        
        if x1 < x2 and y1 < y2:
            drone_x1 = max(0, -x_offset)
            drone_y1 = max(0, -y_offset)
            drone_x2 = drone_x1 + (x2 - x1)
            drone_y2 = drone_y1 + (y2 - y1)
            
            img[y1:y2, x1:x2] = drone_rotated[drone_y1:drone_y2, drone_x1:drone_x2, :3]
    
    c = p.astype(int)
    cv2.circle(img, tuple(c), int(R_BODY), (0, 255, 255), 1)

def distance_point_rect(px, py, x1, y1, x2, y2):
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return math.hypot(px - cx, py - cy), np.array([cx, cy])

def nearest_obstacle_info(p):
    x, y = p
    d_bounds = min(x - X_MIN, X_MAX - x, y - Y_MIN, Y_MAX - y)
    d_best, cp_best = d_bounds, np.array([np.clip(x, X_MIN, X_MAX), np.clip(y, Y_MIN, Y_MAX)])
    for x1, y1, x2, y2 in OBST_INF:
        d, cp = distance_point_rect(x, y, x1, y1, x2, y2)
        if d < d_best:
            d_best, cp_best = d, cp
    return d_best, cp_best

def triangle_collision(tri):
    c = np.mean(tri, axis=0)
    x, y = c
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        return True
    for x1, y1, x2, y2 in OBST_INF:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        for pt in tri:
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return True
    return False

def compute_field(p, th, wall_follow_state, active_goal):
    g_vec = active_goal - p
    g_dist = np.linalg.norm(g_vec) + EPS_DIST
    a_goal = (g_vec / g_dist) * K_GOAL

    d_obs, cp = nearest_obstacle_info(p)
    r_vec = p - cp
    r_norm = np.linalg.norm(r_vec) + EPS_DIST

    if d_obs < REP_CUTOFF:
        rep_mag = K_REP / (d_obs * d_obs)
        a_rep = (r_vec / r_norm) * rep_mag
    else:
        a_rep = np.zeros(2)

    field = a_goal + a_rep

    new_state = wall_follow_state
    if wall_follow_state is None and d_obs < NEAR_ON:
        n = (r_vec / r_norm)
        t_left  = np.array([-n[1], n[0]])
        t_right = -t_left
        goal_dir = g_vec / g_dist
        new_state = "L" if np.dot(t_left, goal_dir) > np.dot(t_right, goal_dir) else "R"

    if wall_follow_state is not None:
        if d_obs > NEAR_OFF:
            new_state = None
        else:
            n = (r_vec / r_norm)
            t = np.array([-n[1], n[0]]) if wall_follow_state == "L" else np.array([n[1], -n[0]])
            field += t * WALL_TANG_GAIN + n * WALL_NORM_PUSH

    if np.linalg.norm(field) < 1e-9:
        field = np.array([math.cos(th), math.sin(th)]) * 1e-3

    return field, new_state, d_obs

def desired_heading_from_field(field):
    return math.degrees(math.atan2(field[1], field[0]))

def speed_from_clearance(clearance):
    if clearance < TIGHT_SLOW:
        return max(MIN_SPEED, MAX_SPEED * 0.55)
    if clearance > 110:
        return min(MAX_SPEED, BASE_SPEED * 1.1)
    return BASE_SPEED

def rebuild_inflated_obstacles():
    global OBST_INF
    OBST_INF = [inflate_rect(*r, R_BODY) for r in obstacles]

def heading_error_deg(current_heading_deg, target_heading_deg):
    err = ((target_heading_deg - current_heading_deg + 180) % 360) - 180
    return err

def yaw_limits_for_radius(distance_to_goal):
    if distance_to_goal <= GOAL_CAPTURE_DIST * 2:
        return NEAR_GOAL_YAW_RATE, NEAR_GOAL_YAW_ACCEL
    elif distance_to_goal <= GOAL_SLOW_RADIUS:
        return GOAL_YAW_RATE, GOAL_YAW_ACCEL
    else:
        return BASE_YAW_RATE, BASE_YAW_ACCEL

def detect_circling(pos, goal_pos, distance_to_goal):
    global recent_positions, circling_detected
    
    if distance_to_goal > GOAL_CAPTURE_DIST * 3:
        recent_positions.clear()
        circling_detected = False
        return False
    
    recent_positions.append(pos.copy())
    if len(recent_positions) > 20:
        recent_positions.pop(0)
    
    if len(recent_positions) < 10:
        return False
    
    center = np.mean(recent_positions, axis=0)
    goal_dist = np.linalg.norm(center - goal_pos)
    
    if goal_dist < distance_to_goal * 0.8:
        circling_detected = True
        return True
    
    return False

def speed_from_heading(distance_to_goal, heading_err_deg, base_speed):
    speed = base_speed
    
    speed -= HE_ERR_SLOW_K * abs(heading_err_deg)
    
    if abs(heading_err_deg) >= TURN_IN_PLACE_ERR:
        speed = min(speed, TURN_IN_PLACE_SPEED)
        if abs(heading_err_deg) >= TURN_IN_PLACE_ERR and speed > 0.5:
            print(f"🔄 Turn-in-place: heading_err={heading_err_deg:.1f}°")
    
    if distance_to_goal <= GOAL_CAPTURE_DIST * 2:
        speed = min(speed, BASE_SPEED * 0.6)
        if abs(heading_err_deg) > 20:
            speed = min(speed, TURN_IN_PLACE_SPEED * 0.8)
    elif distance_to_goal <= GOAL_SLOW_RADIUS:
        speed = min(speed, BASE_SPEED)
    
    speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
    return speed

def spawn_obstacles_for_new_segment(min_n=2, max_n=3):
    global obstacles, pos, theta
    
    n_target = np.random.randint(min_n, max_n + 1)
    print(f"[Spawn] Target: {n_target} obstacles, current obstacles: {len(obstacles)}")
    
    new_rects = []
    attempts_per_obstacle = 0
    max_attempts_per_obstacle = 50
    total_attempts = 0
    max_total_attempts = 200
    
    while len(new_rects) < n_target:
        attempts_per_obstacle += 1
        total_attempts += 1
        
        if total_attempts > max_total_attempts:
            print(f"[Emergency] Max attempts reached, forcing minimal spawn")
            break
            
        if attempts_per_obstacle > max_attempts_per_obstacle:
            attempts_per_obstacle = 0
            print(f"[Fallback] Using deterministic placement, attempts: {total_attempts}")
            rect = deterministic_fallback_placement()
            if rect:
                new_rects.append(rect)
                obstacles.append(rect)
                print(f"[Fallback] Successfully placed obstacle {len(new_rects)}")
            else:
                print(f"[Fallback] Deterministic placement failed")
            continue
        
        fwd = np.array([math.cos(theta), math.sin(theta)])
        left = np.array([-math.sin(theta), math.cos(theta)])
        
        stage = "A"
        if attempts_per_obstacle > 30:
            stage = "B"
        elif attempts_per_obstacle > 60:
            stage = "C"
        elif attempts_per_obstacle > 80:
            stage = "D"
        
        if stage == "A":
            dist = np.random.uniform(OBST_AHEAD_DIST_MIN, OBST_AHEAD_DIST_MAX)
            lateral = np.random.uniform(-90, 90)
            corridor_padding = 12.5
        elif stage == "B":
            dist = np.random.uniform(OBST_AHEAD_DIST_MIN, OBST_AHEAD_DIST_MAX + 150)
            lateral = np.random.uniform(-120, 120)
            corridor_padding = 8.0
        elif stage == "C":
            dist = np.random.uniform(OBST_AHEAD_DIST_MIN, WIDTH * 0.7)
            lateral = np.random.uniform(-150, 150)
            corridor_padding = 5.0
        else:
            dist = np.random.uniform(80, 350)
            lateral = np.random.uniform(-250, 250)
            corridor_padding = 0.0
        
        center = pos + fwd * dist + left * lateral
        
        if np.random.random() < 0.5:
            w = np.random.uniform(OBST_W_MIN, OBST_W_MAX)
            h = np.random.uniform(OBST_H_MIN, OBST_H_MAX)
        else:
            h = np.random.uniform(OBST_W_MIN, OBST_W_MAX)
            w = np.random.uniform(OBST_H_MIN, OBST_H_MAX)
        
        x1 = center[0] - w/2
        y1 = center[1] - h/2
        x2 = center[0] + w/2
        y2 = center[1] + h/2
        
        if x1 < X_MIN or x2 > X_MAX or y1 < Y_MIN or y2 > Y_MAX:
            continue
            
        rect = (int(x1), int(y1), int(x2), int(y2))
        
        if corridor_padding > 0 and check_corridor_overlap(rect, pos, goals[0]):
            continue
            
        drone_dist = distance_point_rect(pos[0], pos[1], x1, y1, x2, y2)[0]
        if drone_dist < R_BODY + 3:
            continue
            
        goal_dist = distance_point_rect(goals[0][0], goals[0][1], x1, y1, x2, y2)[0]
        if goal_dist < 15:
            continue
            
        overlap = False
        for obs in new_rects:
            ox1, oy1, ox2, oy2 = obs
            if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                overlap = True
                break
                
        if not overlap:
            new_rects.append(rect)
            obstacles.append(rect)
            attempts_per_obstacle = 0
            print(f"[Spawn] Successfully placed obstacle {len(new_rects)} at ({x1},{y1})-({x2},{y2})")
    
    if len(new_rects) < 2:
        print(f"[Warning] Only spawned {len(new_rects)} obstacles, forcing fallback")
        while len(new_rects) < 2:
            rect = deterministic_fallback_placement()
            if rect:
                new_rects.append(rect)
                obstacles.append(rect)
            else:
                break
    
    if len(new_rects) == 0:
        print(f"[Emergency] No obstacles spawned, creating minimal obstacle")
        fwd = np.array([math.cos(theta), math.sin(theta)])
        left = np.array([-math.sin(theta), math.cos(theta)])
        
        for attempt in range(10):
            center = pos + fwd * (100 + attempt * 40) + left * (30 + attempt * 20)
            rect = (int(center[0] - 15), int(center[1] - 50), int(center[0] + 15), int(center[1] + 50))
            
            if (X_MIN <= rect[0] <= X_MAX and Y_MIN <= rect[1] <= Y_MAX and 
                X_MIN <= rect[2] <= X_MAX and Y_MIN <= rect[3] <= Y_MAX):
                obstacles.append(rect)
                new_rects.append(rect)
                print(f"[Emergency] Created obstacle at attempt {attempt + 1}")
                break
    
    if len(new_rects) < 2:
        print(f"[Emergency] Still need more obstacles, creating additional ones")
        fwd = np.array([math.cos(theta), math.sin(theta)])
        left = np.array([-math.sin(theta), math.cos(theta)])
        
        for i in range(2 - len(new_rects)):
            center = pos + fwd * (200 + i * 50) + left * (100 + i * 30)
            rect = (int(center[0] - 20), int(center[1] - 60), int(center[0] + 20), int(center[1] + 60))
            
            if (X_MIN <= rect[0] <= X_MAX and Y_MIN <= rect[1] <= Y_MAX and 
                X_MIN <= rect[2] <= X_MAX and Y_MIN <= rect[3] <= Y_MAX):
                obstacles.append(rect)
                new_rects.append(rect)
                print(f"[Emergency] Created additional obstacle {len(new_rects)}")
    
    print(f"[Spawn] Successfully created {len(new_rects)} obstacles")
    return len(new_rects)

def deterministic_fallback_placement():
    fwd = np.array([math.cos(theta), math.sin(theta)])
    left = np.array([-math.sin(theta), math.cos(theta)])
    
    attempts = 0
    for offset in [60, 100, 140, 180, 220, 260]:
        for side in [-1, 1]:
            for dist in [80, 120, 160, 200, 240, 280]:
                attempts += 1
                center = pos + fwd * dist + left * (side * offset)
                
                w = 25
                h = 140
                
                x1 = center[0] - w/2
                y1 = center[1] - h/2
                x2 = center[0] + w/2
                y2 = center[1] + h/2
                
                if x1 < X_MIN or x2 > X_MAX or y1 < Y_MIN or y2 > Y_MAX:
                    continue
                    
                rect = (int(x1), int(y1), int(x2), int(y2))
                
                drone_dist = distance_point_rect(pos[0], pos[1], x1, y1, x2, y2)[0]
                if drone_dist < R_BODY + 2:
                    continue
                    
                if len(goals) > 0:
                    goal_dist = distance_point_rect(goals[0][0], goals[0][1], x1, y1, x2, y2)[0]
                    if goal_dist < 12:
                        continue
                    
                overlap = False
                for obs in obstacles:
                    ox1, oy1, ox2, oy2 = obs
                    if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                        overlap = True
                        break
                        
                if not overlap:
                    print(f"[Fallback] Success at attempt {attempts}: offset={offset}, side={side}, dist={dist}")
                    return rect
    
    print(f"[Fallback] All {attempts} attempts failed")
    return None

def handle_goal_reached():
    global goals, obstacles, GOAL_CYCLE
    
    GOAL_CYCLE += 1
    print(f"[Goal] === CYCLE {GOAL_CYCLE} START ===")
    
    obstacles.clear()
    print(f"[Goal] Cleared {len(obstacles)} obstacles")
    
    goals.popleft()
    print(f"[Goal] Popped goal, {len(goals)} goals remaining")
    
    while len(goals) < 2:
        new_goal = spawn_goal_ahead(pos, theta)
        goals.append(new_goal)
        print(f"[Cycle {GOAL_CYCLE}] Added backup goal at {new_goal}")
    
    print(f"[Goal] Starting obstacle spawn with {len(goals)} goals available")
    spawned_count = spawn_obstacles_for_new_segment(2, 3)
    rebuild_inflated_obstacles()
    
    print(f"[Cycle {GOAL_CYCLE}] Spawned {spawned_count} obstacles, {len(goals)} goals remaining")
    print(f"[Goal] === CYCLE {GOAL_CYCLE} END ===")

def prune_obstacles_behind(pos):
    global obstacles, OBST_INF
    to_remove = []
    for i, (x1, y1, x2, y2) in enumerate(obstacles):
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        dist = np.linalg.norm(center - pos)
        if dist > PRUNE_OBST_BEHIND_DIST:
            to_remove.append(i)
    
    for i in reversed(to_remove):
        obstacles.pop(i)
    
    rebuild_inflated_obstacles()

def main():
    global pos, theta, yaw_rate, target_heading_deg, wall_follow, path_pts, goal_reached, simulation_running

    if len(goals) == 0:
        goals.append(np.array([WIDTH//2 + np.random.uniform(-100, 100),
                              60 + np.random.uniform(-20, 20)], dtype=float))

    win = "Smooth Vector-Field Drone (Inflated Obstacles + Substeps)"
    canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, handle_mouse_click)

    while True:
        if len(goals) == 0:
            goals.append(spawn_goal_ahead(pos, theta))
            print(f"[Emergency] Added goal, now {len(goals)} goals")
        
        active_goal = goals[0]
            
        draw_world(canvas)
        draw_start_stop_button(canvas, simulation_running)

        distance_to_goal = np.linalg.norm(pos - active_goal)
        heading_err = heading_error_deg(math.degrees(theta), desired_heading_from_field(compute_field(pos, theta, wall_follow, active_goal)[0]))
        
        is_circling = detect_circling(pos, active_goal, distance_to_goal)
        
        if distance_to_goal <= GOAL_CAPTURE_DIST and abs(heading_err) <= GOAL_CAPTURE_HEAD:
            print(f"[Main] Goal reached! Distance: {distance_to_goal:.1f}, Heading err: {heading_err:.1f}")
            handle_goal_reached()
        elif distance_to_goal <= GOAL_CAPTURE_DIST:
            print(f"[Main] Close to goal but heading wrong: Distance: {distance_to_goal:.1f}, Heading err: {heading_err:.1f}")
        elif abs(heading_err) <= GOAL_CAPTURE_HEAD:
            print(f"[Main] Heading aligned but too far: Distance: {distance_to_goal:.1f}, Heading err: {heading_err:.1f}")
        
        if is_circling:
            print(f"[Anti-circle] Circling detected! Distance: {distance_to_goal:.1f}")

        if not simulation_running:
            path_pts.append(tuple(pos.astype(int)))
            for pt in path_pts[-2000:]:
                cv2.circle(canvas, pt, 1, (180, 180, 180), -1)
            draw_drone(canvas, pos, theta)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                break
            continue

        field, wall_follow, d_obs = compute_field(pos, theta, wall_follow, active_goal)

        desired_heading = desired_heading_from_field(field)

        err_to_target = heading_error_deg(target_heading_deg, desired_heading)
        target_heading_deg = target_heading_deg + ANGLE_SMOOTH * err_to_target

        current_heading_deg = math.degrees(theta)
        heading_err = heading_error_deg(current_heading_deg, target_heading_deg)
        
        max_rate, max_accel = yaw_limits_for_radius(distance_to_goal)
        
        if is_circling:
            max_rate *= 2.0
            max_accel *= 2.0
            print(f"[Anti-circle] Enhanced yaw: rate={max_rate:.1f}, accel={max_accel:.1f}")
        
        desired_rate = np.clip(heading_err, -max_rate, max_rate)
        delta = np.clip(desired_rate - yaw_rate, -max_accel, max_accel)
        yaw_rate = np.clip(yaw_rate + delta, -max_rate, max_rate)

        theta = angnorm(theta + math.radians(yaw_rate))

        clearance_speed = speed_from_clearance(d_obs)
        spd = speed_from_heading(distance_to_goal, heading_err, clearance_speed)
        
        if is_circling:
            spd = min(spd, TURN_IN_PLACE_SPEED * 0.5)
            print(f"[Anti-circle] Reduced speed to {spd:.1f}")

        step_len = 2.0
        total_dist = spd * (DT / 0.06)
        n_steps = max(1, int(math.ceil(total_dist / step_len)))
        sub_d = total_dist / n_steps

        collided = False
        for _ in range(n_steps):
            trial = pos + sub_d * np.array([math.cos(theta), math.sin(theta)])
            tri = get_triangle(trial, theta)
            if triangle_collision(tri):
                collided = True
                break
            pos = trial

        if collided:
            side = np.array([-math.sin(theta), math.cos(theta)]) * 1.5
            candidates = (
                pos + side,
                pos - side,
                pos - 0.5 * sub_d * np.array([math.cos(theta), math.sin(theta)]),
            )
            for cand in candidates:
                tri = get_triangle(cand, theta)
                if not triangle_collision(tri):
                    pos = cand
                    break

        if len(goals) < 3:
            goals.append(spawn_goal_ahead(pos, theta))

        path_pts.append(tuple(pos.astype(int)))
        for pt in path_pts[-2000:]:
            cv2.circle(canvas, pt, 1, (180, 180, 180), -1)
        draw_drone(canvas, pos, theta)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
