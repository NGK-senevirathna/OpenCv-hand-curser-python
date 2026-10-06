import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import math
import time
import os
import urllib.request

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ================== Settings ==================
wCam, hCam = 640, 480
frameR = 40
smoothening = 6

CLICK_DIST = 35
SCROLL_DEADZONE = 3
SCROLL_SCALE = 12
DOUBLE_CLICK_THRESHOLD = 0.35
DRAG_ACTIVATE_FRAMES = 4
SCROLL_ACTIVATE_FRAMES = 3

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ================== Download model if missing ==================
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model... (one time only)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print("Model download failed:", e)
        print("Please download manually from:")
        print(MODEL_URL)
        print("and put the file in the same folder as this script.")
        exit()

# ================== MediaPipe HandLandmarker (New API) ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)

# ================== Camera & Screen ==================
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
dragging = False
click_down = False
last_click_time = 0.0
prev_scroll_y = None
scroll_frames = 0
drag_frames = 0
mode_text = "MOVE"
start_time = time.time()

def fingers_up(landmarks, w, h):
    """Return [thumb, index, middle, ring, pinky] 1=up 0=down"""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (x comparison - works for right hand mirrored)
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (y comparison)
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Detect
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mode_text = "MOVE"
        scroll_mode = False

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]  # first hand

            # Convert normalized landmarks to pixel coordinates
            lm_list = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Key points
            x_index, y_index = lm_list[8]
            x_middle, y_middle = lm_list[12]
            x_thumb, y_thumb = lm_list[4]

            # Draw usable area
            cv2.rectangle(frame, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

            # Draw landmarks (simple)
            for pt in lm_list:
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)

            fingers = fingers_up(landmarks, w, h)
            thumb_f, index_f, middle_f, ring_f, pinky_f = fingers
            num_up = sum(fingers)

            # Patterns
            scroll_pattern = (thumb_f == 0 and index_f == 1 and middle_f == 1 and ring_f == 1 and pinky_f == 0)
            hand_closed = (num_up == 0)
            hand_open = (num_up >= 3)

            # Debounce
            if scroll_pattern:
                scroll_frames += 1
            else:
                scroll_frames = 0
            scroll_mode = scroll_frames >= SCROLL_ACTIVATE_FRAMES

            if hand_closed:
                drag_frames += 1
            else:
                drag_frames = 0

            # ========== MOVE ==========
            x3 = np.interp(x_index, (frameR, w - frameR), (0, screen_w))
            y3 = np.interp(y_index, (frameR, h - frameR), (0, screen_h))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            cv2.circle(frame, (x_index, y_index), 12, (255, 0, 255), cv2.FILLED)

            # ========== SCROLL ==========
            if scroll_mode:
                if prev_scroll_y is None:
                    prev_scroll_y = y_index
                dy = y_index - prev_scroll_y
                if abs(dy) > SCROLL_DEADZONE:
                    scroll_amount = int(-dy * SCROLL_SCALE)
                    pyautogui.scroll(scroll_amount)
                    prev_scroll_y = y_index
                mode_text = "SCROLL"
            else:
                prev_scroll_y = None

            # ========== CLICK / DOUBLE CLICK (Index + Middle pinch) ==========
            if (not scroll_mode and index_f == 1 and middle_f == 1 and not dragging):
                dist_im = math.hypot(x_middle - x_index, y_middle - y_index)
                cv2.line(frame, (x_index, y_index), (x_middle, y_middle), (255, 0, 255), 2)

                if dist_im < CLICK_DIST:
                    cx = (x_index + x_middle) // 2
                    cy = (y_index + y_middle) // 2
                    cv2.circle(frame, (cx, cy), 14, (0, 255, 0), cv2.FILLED)

                    if not click_down:
                        now = time.time()
                        if now - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                            mode_text = "DOUBLE CLICK"
                        else:
                            mode_text = "CLICK"
                        pyautogui.click()
                        last_click_time = now
                        click_down = True
                else:
                    click_down = False
            else:
                click_down = False

            # ========== DRAG (Fist) ==========
            if drag_frames >= DRAG_ACTIVATE_FRAMES and not dragging:
                pyautogui.mouseDown()
                dragging = True

            if hand_open and dragging:
                pyautogui.mouseUp()
                dragging = False

            if dragging:
                mode_text = "DRAG"
                cv2.putText(frame, "DRAGGING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # No hand → release drag
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            click_down = False
            prev_scroll_y = None
            scroll_frames = 0
            drag_frames = 0

        # Status
        cv2.putText(frame, f"Mode: {mode_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        instructions = [
            "Index: Move",
            "Index+Middle pinch: Click / Double",
            "Fist: Drag  |  Open hand: Drop",
            "3 fingers (I+M+R): Scroll",
            "Press Q to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 80 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Virtual Mouse (New MediaPipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()