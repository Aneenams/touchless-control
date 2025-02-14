import cv2 
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pyautogui
import win32api
import win32con
import math
import pyperclip

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )

        # Screen and window setup
        self.whiteboard_width = 1280
        self.whiteboard_height = 720
        
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Mouse control settings
        self.mouse_smoothing_factor = 0.5
        self.mouse_speed = 1.5
        self.last_mouse_pos = (0, 0)

        # Drawing setup
        self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
        self.drawing = False
        self.draw_color = (0, 0, 0)
        self.draw_thickness = 2
        
        self.drawing_points = deque(maxlen=50)
        self.smoothing_factor = 0.15
        
        self.erasing = False
        self.eraser_size = 30
        self.eraser_color = (255, 255, 255)
        
        # State tracking
        self.prev_hand_landmarks = None
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.whiteboard_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.whiteboard_height)
        
        # Mouse state
        self.mouse_pressed = False
        self.last_click_time = 0
        self.click_cooldown = 0.3

        # Gesture states
        self.whiteboard_active = True
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        self.toggle_gesture_duration = 0
        self.toggle_threshold = 15  # Frames to hold gesture
        self.last_gesture = None

    def get_finger_states(self, hand_landmarks):
        """Get the state of each finger (up/down) with improved accuracy"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        # For thumb, compare with metacarpal instead of PIP
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        
        # Check if thumb is extended
        thumb_extended = (hand_landmarks.landmark[finger_tips[0]].x - wrist.x) * (thumb_cmc.x - wrist.x) > 0
        
        # Check other fingers
        fingers_extended = [thumb_extended]
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            tip_y = hand_landmarks.landmark[tip].y
            pip_y = hand_landmarks.landmark[pip].y
            fingers_extended.append(tip_y < pip_y)
            
        return fingers_extended

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection with improved reliability"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        
        # Get specific landmark positions
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate thumb-index distance for pinch detection
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )

        # Define gestures based on finger states
        gestures = {
            'pinch': thumb_index_dist < 0.05,
            'eraser': fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]),
            'copy': fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[4],
            'paste': all(fingers_extended[1:]),
            'toggle': not any(fingers_extended[:4]) and fingers_extended[4]  # Only pinky raised
        }
        
        return gestures

    def handle_mouse_and_drawing(self, hand_landmarks, frame_width, frame_height):
        """Enhanced handler for mouse, drawing, and gestures"""
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert to screen coordinates
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)
        
        # Apply mouse smoothing
        smooth_x, smooth_y = self.smooth_mouse_movement(x, y)
        
        # Convert to screen coordinates
        screen_x = int((smooth_x / frame_width) * self.screen_width)
        screen_y = int((smooth_y / frame_height) * self.screen_height)
        
        # Ensure coordinates are within screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        # Move mouse cursor
        win32api.SetCursorPos((screen_x, screen_y))
        
        # Detect gestures
        gestures = self.detect_gestures(hand_landmarks)
        current_time = time.time()

        # Handle toggle gesture with duration check
        if gestures['toggle']:
            if self.last_gesture != 'toggle':
                self.toggle_gesture_duration = 1
            else:
                self.toggle_gesture_duration += 1
                
            if self.toggle_gesture_duration >= self.toggle_threshold:
                if current_time - self.last_gesture_time > self.gesture_cooldown:
                    self.whiteboard_active = not self.whiteboard_active
                    self.last_gesture_time = current_time
                    print(f"Whiteboard {'activated' if self.whiteboard_active else 'deactivated'}!")
                    self.toggle_gesture_duration = 0
        else:
            self.toggle_gesture_duration = 0
        
        self.last_gesture = 'toggle' if gestures['toggle'] else None
        
        # Handle copy/paste gestures
        if current_time - self.last_gesture_time > self.gesture_cooldown:
            if gestures['copy']:
                pyautogui.hotkey('ctrl', 'c')
                self.last_gesture_time = current_time
                print("Copy gesture detected!")
            elif gestures['paste']:
                pyautogui.hotkey('ctrl', 'v')
                self.last_gesture_time = current_time
                print("Paste gesture detected!")
        
        # Handle mouse click and drawing
        if gestures['pinch'] and not self.mouse_pressed and (current_time - self.last_click_time) > self.click_cooldown:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
            self.mouse_pressed = True
            self.last_click_time = current_time
            
            if self.whiteboard_active:
                self.drawing = True
                self.drawing_points.append((x, y))
            
        elif not gestures['pinch'] and self.mouse_pressed:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
            pyautogui.leftClick
            self.mouse_pressed = False
            self.drawing = False
        
        # Handle drawing if whiteboard is active
        if self.whiteboard_active:
            if self.drawing:
                self.drawing_points.append((x, y))
                if len(self.drawing_points) >= 2:
                    smoothed_points = self.smooth_drawing(list(self.drawing_points))
                    for i in range(len(smoothed_points) - 1):
                        cv2.line(self.whiteboard, 
                                smoothed_points[i], 
                                smoothed_points[i + 1], 
                                self.draw_color, 
                                self.draw_thickness, 
                                cv2.LINE_AA)
                    self.drawing_points.clear()
                    self.drawing_points.append((x, y))
            
            if gestures['eraser']:
                cv2.circle(self.whiteboard, (x, y), self.eraser_size, self.eraser_color, -1)
        
        return x, y

    def smooth_mouse_movement(self, x, y):
        if self.last_mouse_pos == (0, 0):
            self.last_mouse_pos = (x, y)
            return x, y
            
        smoothed_x = int(x * self.mouse_smoothing_factor + self.last_mouse_pos[0] * (1 - self.mouse_smoothing_factor))
        smoothed_y = int(y * self.mouse_smoothing_factor + self.last_mouse_pos[1] * (1 - self.mouse_smoothing_factor))
        
        self.last_mouse_pos = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def smooth_drawing(self, points):
        if len(points) < 3:
            return points
            
        smoothed = []
        for i in range(1, len(points) - 1):
            x = int(points[i-1][0] * 0.25 + points[i][0] * 0.5 + points[i+1][0] * 0.25)
            y = int(points[i-1][1] * 0.25 + points[i][1] * 0.5 + points[i+1][1] * 0.25)
            smoothed.append((x, y))
            
        return smoothed

    def process_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return False

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if self.whiteboard_active:
                display = self.whiteboard.copy()
            else:
                display = frame.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    x, y = self.handle_mouse_and_drawing(hand_landmarks, frame.shape[1], frame.shape[0])
                    
                    if self.whiteboard_active:
                        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Display FPS and toggle gesture progress
            cv2.putText(frame, f'FPS: {int(avg_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.toggle_gesture_duration > 0:
                progress = min(self.toggle_gesture_duration / self.toggle_threshold * 100, 100)
                cv2.putText(frame, f'Toggle: {int(progress)}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Whiteboard" if self.whiteboard_active else "Camera Feed", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
            
            return True

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return True

    def run(self):
        print("Starting Enhanced Gesture Controller")
        print("\nGesture Controls:")
        print("1. Basic Controls:")
        print("   - Move mouse: Point with index finger")
        print("   - Click/Draw: Pinch thumb and index finger")
        print("   - Erase: Hold index and middle fingers together")
        print("\n2. Advanced Controls:")
        print("   - Copy: Raise index, middle, and ring fingers (keep pinky down)")
        print("   - Paste: Raise all fingers")
        print("   - Toggle Whiteboard: Raise ONLY pinky finger and hold for 1 second")
        print("\n3. Keyboard Controls:")
        print("   - Clear whiteboard: Press 'c'")
        print("   - Quit program: Press 'q'")
        print("\nNote: For toggle gesture, keep all fingers down except pinky and hold the position")
        
        self.running = True
        try:
            while self.running:
                if not self.process_frame():
                    break
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureController()
    controller.run()
    #yevan sett 
    # select copy paste  vm whiteboard
    # nee kollavallodi kallipenne