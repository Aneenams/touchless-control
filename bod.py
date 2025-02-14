import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque


class GestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.85,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )

        # Screen and window setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.whiteboard_width = 1280
        self.whiteboard_height = 720
        pyautogui.FAILSAFE = False

        # Drawing setup
        self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
        self.drawing = False
        self.draw_color = (0, 0, 0)
        self.draw_thickness = 2
        
        # Path visualization
        self.path_overlay = np.zeros_like(self.whiteboard)
        self.path_points = deque(maxlen=30)
        
        # Smoothing
        self.position_history = deque(maxlen=5)
        
        # Drawing state
        self.last_point = None
        self.erasing = False
        self.eraser_size = 20
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.whiteboard_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.whiteboard_height)

    def smooth_position(self, x, y):
        """Smooth the cursor position using moving average"""
        self.position_history.append((x, y))
        if len(self.position_history) < 2:
            return x, y
        
        x_smooth = sum(p[0] for p in self.position_history) / len(self.position_history)
        y_smooth = sum(p[1] for p in self.position_history) / len(self.position_history)
        return int(x_smooth), int(y_smooth)

    def update_path_visualization(self, x, y):
        """Update the path visualization with current cursor position"""
        self.path_points.append((x, y))
        
        # Fade existing path
        self.path_overlay = (self.path_overlay * 0.9).astype(np.uint8)
        
        # Draw new path segment
        if len(self.path_points) > 1:
            cv2.line(self.path_overlay,
                    self.path_points[-2],
                    self.path_points[-1],
                    (0, 255, 0),
                    1)

    def get_hand_position(self, hand_landmarks):
        """Get cursor position from hand landmarks"""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * self.whiteboard_width)
        y = int(index_tip.y * self.whiteboard_height)
        return self.smooth_position(x, y)

    def get_finger_distances(self, hand_landmarks):
        """Calculate distances between fingers"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        index_middle_dist = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5
        
        return thumb_index_dist, index_middle_dist

    def process_frame(self):
        """Process each frame from the webcam"""
        ret, frame = self.cap.read()
        if not ret:
            return False

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Clear path overlay
        self.path_overlay = np.zeros_like(self.whiteboard)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get cursor position
                x, y = self.get_hand_position(hand_landmarks)
                
                # Update path visualization
                self.update_path_visualization(x, y)
                
                # Get finger distances
                thumb_index_dist, index_middle_dist = self.get_finger_distances(hand_landmarks)
                
                # Drawing mode
                if thumb_index_dist < 0.1:  # Thumb and index finger close
                    if not self.drawing:
                        self.drawing = True
                        self.last_point = (x, y)
                    else:
                        cv2.line(self.whiteboard,
                                self.last_point,
                                (x, y),
                                self.draw_color,
                                self.draw_thickness)
                        self.last_point = (x, y)
                else:
                    self.drawing = False
                
                # Erasing mode
                if index_middle_dist < 0.05:  # Index and middle finger close
                    cv2.circle(self.whiteboard,
                             (x, y),
                             self.eraser_size,
                             (255, 255, 255),
                             -1)

        # Combine whiteboard with path overlay
        display = cv2.addWeighted(self.whiteboard, 1, self.path_overlay, 0.5, 0)
        
        # Draw cursor point
        if results.multi_hand_landmarks:
            x, y = self.get_hand_position(hand_landmarks)
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        # Show the frames
        cv2.imshow("Hand Tracking", frame)
        cv2.imshow("Whiteboard", display)

        return True

    def run(self):
        """Main loop"""
        try:
            while True:
                if not self.process_frame():
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = GestureController()
    controller.run()
    # whiteboard onlydef handle_scroll(self, hand_landmarks):
    """Simple and reliable scroll handler"""
    # Check for scroll gesture (index and middle fingers up, others down)
    fingers_extended = self.get_finger_states(hand_landmarks)
    is_scroll_gesture = (fingers_extended[1] and  # index finger
                        fingers_extended[2] and    # middle finger
                        not any(fingers_extended[3:]) and  # ring and pinky down
                        not fingers_extended[0])   # thumb down
    
    # Get middle finger tip position
    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    current_y = middle_tip.y
    
    if is_scroll_gesture:
        if not self.scroll_active:
            # Start new scroll
            self.scroll_active = True
            self.last_scroll_y = current_y
        else:
            # Calculate scroll
            if self.last_scroll_y is not None:
                # Calculate movement
                y_diff = current_y - self.last_scroll_y
                
                # Simple threshold check
                if abs(y_diff) > 0.01:  # Increased threshold for stability
                    # Convert to scroll amount
                    scroll_amount = int(y_diff * 1000)  # Simplified calculation
                    
                    # Limit maximum scroll
                    scroll_amount = max(min(scroll_amount, 50), -50)
                    
                    # Perform scroll
                    win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                
                self.last_scroll_y = current_y
    else:
        # Reset when gesture ends
        self.scroll_active = False
        self.last_scroll_y = Nonedef handle_scroll(self, hand_landmarks):
    """Simple and reliable scroll handler"""
    # Check for scroll gesture (index and middle fingers up, others down)
    fingers_extended = self.get_finger_states(hand_landmarks)
    is_scroll_gesture = (fingers_extended[1] and  # index finger
                        fingers_extended[2] and    # middle finger
                        not any(fingers_extended[3:]) and  # ring and pinky down
                        not fingers_extended[0])   # thumb down
    
    # Get middle finger tip position
    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    current_y = middle_tip.y
    
    if is_scroll_gesture:
        if not self.scroll_active:
            # Start new scroll
            self.scroll_active = True
            self.last_scroll_y = current_y
        else:
            # Calculate scroll
            if self.last_scroll_y is not None:
                # Calculate movement
                y_diff = current_y - self.last_scroll_y
                
                # Simple threshold check
                if abs(y_diff) > 0.01:  # Increased threshold for stability
                    # Convert to scroll amount
                    scroll_amount = int(y_diff * 1000)  # Simplified calculation
                    
                    # Limit maximum scroll
                    scroll_amount = max(min(scroll_amount, 50), -50)
                    
                    # Perform scroll
                    win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                
                self.last_scroll_y = current_y
    else:
        # Reset when gesture ends
        self.scroll_active = False
        self.last_scroll_y = Nonedef handle_scroll(self, hand_landmarks):
    """Simple and reliable scroll handler"""
    # Check for scroll gesture (index and middle fingers up, others down)
    fingers_extended = self.get_finger_states(hand_landmarks)
    is_scroll_gesture = (fingers_extended[1] and  # index finger
                        fingers_extended[2] and    # middle finger
                        not any(fingers_extended[3:]) and  # ring and pinky down
                        not fingers_extended[0])   # thumb down
    
    # Get middle finger tip position
    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    current_y = middle_tip.y
    
    if is_scroll_gesture:
        if not self.scroll_active:
            # Start new scroll
            self.scroll_active = True
            self.last_scroll_y = current_y
        else:
            # Calculate scroll
            if self.last_scroll_y is not None:
                # Calculate movement
                y_diff = current_y - self.last_scroll_y
                
                # Simple threshold check
                if abs(y_diff) > 0.01:  # Increased threshold for stability
                    # Convert to scroll amount
                    scroll_amount = int(y_diff * 1000)  # Simplified calculation
                    
                    # Limit maximum scroll
                    scroll_amount = max(min(scroll_amount, 50), -50)
                    
                    # Perform scroll
                    win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                
                self.last_scroll_y = current_y
    else:
        # Reset when gesture ends
        self.scroll_active = False
        self.last_scroll_y = Nonedef handle_scroll(self, hand_landmarks):
    """Simple and reliable scroll handler"""
    # Check for scroll gesture (index and middle fingers up, others down)
    fingers_extended = self.get_finger_states(hand_landmarks)
    is_scroll_gesture = (fingers_extended[1] and  # index finger
                        fingers_extended[2] and    # middle finger
                        not any(fingers_extended[3:]) and  # ring and pinky down
                        not fingers_extended[0])   # thumb down
    
    # Get middle finger tip position
    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    current_y = middle_tip.y
    
    if is_scroll_gesture:
        if not self.scroll_active:
            # Start new scroll
            self.scroll_active = True
            self.last_scroll_y = current_y
        else:
            # Calculate scroll
            if self.last_scroll_y is not None:
                # Calculate movement
                y_diff = current_y - self.last_scroll_y
                
                # Simple threshold check
                if abs(y_diff) > 0.01:  # Increased threshold for stability
                    # Convert to scroll amount
                    scroll_amount = int(y_diff * 1000)  # Simplified calculation
                    
                    # Limit maximum scroll
                    scroll_amount = max(min(scroll_amount, 50), -50)
                    
                    # Perform scroll
                    win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                
                self.last_scroll_y = current_y
    else:
        # Reset when gesture ends
        self.scroll_active = False
        self.last_scroll_y = None