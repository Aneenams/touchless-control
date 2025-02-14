
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from math import sqrt

class VirtualMouse:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Smoothing settings
        self.smooth_factor = 0.2
        self.prev_x, self.prev_y = 0, 0

        # Gesture settings
        self.action_delay = 0.3
        self.last_action_time = time.time()
        self.scroll_speed = 20
        self.gesture_threshold = 0.1

    def smooth_movement(self, x, y):
        """Apply stronger smoothing for slower cursor movement"""
        smoothed_x = int(self.smooth_factor * x + (1 - self.smooth_factor) * self.prev_x)
        smoothed_y = int(self.smooth_factor * y + (1 - self.smooth_factor) * self.prev_y)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

    def get_normalized_coordinates(self, hand_landmarks):
        """Convert hand landmarks to screen coordinates"""
        index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        x = int(np.interp(index_finger.x, (0.1, 0.9), (0, self.screen_width)))
        y = int(np.interp(index_finger.y, (0.1, 0.9), (0, self.screen_height)))
        
        return self.smooth_movement(x, y)

    def handle_mouse_events(self, hand_landmarks):
        """Handle mouse events based on gestures"""
        try:
            x, y = self.get_normalized_coordinates(hand_landmarks)
            current_time = time.time()
            
            # Move cursor using index finger only
            pyautogui.moveTo(x, y)
            
            # Get finger status
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            thumb_index_dist = sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            # Selection (when thumb and index tips touch)
            if thumb_index_dist < self.gesture_threshold and current_time - self.last_action_time > self.action_delay:
                pyautogui.click()
                self.last_action_time = current_time
            
        except Exception as e:
            print(f"Error handling mouse events: {e}")

    def run(self):
        """Main loop for virtual mouse control"""
        print("Virtual Mouse Controller Started")
        print("Press 'q' to quit")

        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        self.handle_mouse_events(hand_landmarks)

                cv2.imshow("Virtual Mouse", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_mouse = VirtualMouse()
    virtual_mouse.run()
