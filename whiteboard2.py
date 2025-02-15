import cv2
import mediapipe as mp
import numpy as np
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
        self.whiteboard_width = 1280
        self.whiteboard_height = 720

        # Drawing setup
        self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
        self.drawing = False
        self.draw_color = (0, 0, 0)
        self.draw_thickness = 2
        
        # Path visualization
        self.path_overlay = np.zeros_like(self.whiteboard)
        self.path_points = deque(maxlen=60)
        
        # Smoothing
        self.position_history = deque(maxlen=8)
        
        # Drawing state
        self.last_point = None
        self.erasing = False
        self.eraser_size = 20
        
        # Cursor trail effect
        self.cursor_trail = deque(maxlen=10)
        
        # Control flags
        self.running = True
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.whiteboard_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.whiteboard_height)

    def smooth_position(self, x, y):
        """Enhanced smooth cursor position using weighted moving average"""
        self.position_history.append((x, y))
        if len(self.position_history) < 2:
            return x, y
        
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights = weights / weights.sum()
        
        x_smooth = sum(p[0] * w for p, w in zip(self.position_history, weights))
        y_smooth = sum(p[1] * w for p, w in zip(self.position_history, weights))
        
        return int(x_smooth), int(y_smooth)

    def update_path_visualization(self, x, y):
        """Enhanced path visualization with smooth fading"""
        try:
            self.path_points.append((x, y))
            self.path_overlay = (self.path_overlay * 0.85).astype(np.uint8)
            
            if len(self.path_points) > 1:
                cv2.line(self.path_overlay,
                        self.path_points[-2],
                        self.path_points[-1],
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
        except Exception as e:
            print(f"Error in path visualization: {e}")

    def draw_cursor_trail(self, display, x, y):
        """Draw a smooth cursor trail with error handling"""
        try:
            self.cursor_trail.append((x, y))
            
            for i in range(len(self.cursor_trail) - 1):
                alpha = (i + 1) / len(self.cursor_trail)
                thickness = max(1, min(3, int(alpha * 3)))  # Ensure thickness is between 1 and 3
                
                start_point = self.cursor_trail[i]
                end_point = self.cursor_trail[i + 1]
                
                # Ensure points are within image boundaries
                if (0 <= start_point[0] < self.whiteboard_width and 
                    0 <= start_point[1] < self.whiteboard_height and
                    0 <= end_point[0] < self.whiteboard_width and 
                    0 <= end_point[1] < self.whiteboard_height):
                    cv2.line(display,
                            start_point,
                            end_point,
                            (0, 0, 255),
                            thickness,
                            cv2.LINE_AA)
        except Exception as e:
            print(f"Error in cursor trail: {e}")

    def get_hand_position(self, hand_landmarks):
        """Get smoothed cursor position from hand landmarks"""
        try:
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_tip.x * self.whiteboard_width)
            y = int(index_tip.y * self.whiteboard_height)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.whiteboard_width - 1))
            y = max(0, min(y, self.whiteboard_height - 1))
            
            return self.smooth_position(x, y)
        except Exception as e:
            print(f"Error in hand position: {e}")
            return (0, 0)

    def get_finger_distances(self, hand_landmarks):
        """Calculate distances between fingers"""
        try:
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            index_middle_dist = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5
            
            return thumb_index_dist, index_middle_dist
        except Exception as e:
            print(f"Error in finger distances: {e}")
            return (1.0, 1.0)  # Return large distances to prevent unwanted actions

    def process_frame(self):
        """Process each frame with enhanced visualization and error handling"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return False

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self.path_overlay = np.zeros_like(self.whiteboard)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    x, y = self.get_hand_position(hand_landmarks)
                    self.update_path_visualization(x, y)
                    
                    thumb_index_dist, index_middle_dist = self.get_finger_distances(hand_landmarks)
                    
                    if thumb_index_dist < 0.1:
                        if not self.drawing:
                            self.drawing = True
                            self.last_point = (x, y)
                        else:
                            cv2.line(self.whiteboard,
                                    self.last_point,
                                    (x, y),
                                    self.draw_color,
                                    self.draw_thickness,
                                    cv2.LINE_AA)
                            self.last_point = (x, y)
                    else:
                        self.drawing = False
                    
                    if index_middle_dist < 0.05:
                        cv2.circle(self.whiteboard,
                                 (x, y),
                                 self.eraser_size,
                                 (255, 255, 255),
                                 -1,
                                 cv2.LINE_AA)

            display = cv2.addWeighted(self.whiteboard, 1, self.path_overlay, 0.5, 0)
            
            if results.multi_hand_landmarks:
                x, y = self.get_hand_position(hand_landmarks)
                self.draw_cursor_trail(display, x, y)
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Whiteboard", display)

            # Add keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                self.running = False
            elif key == ord('c'):  # Clear whiteboard
                self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
            
            return self.running

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return True  # Continue running despite errors

    def run(self):
        """Main loop with error handling"""
        print("Starting Enhanced Gesture Controller")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'c' to clear whiteboard")
        print("- Pinch thumb and index finger to draw")
        print("- Pinch index and middle finger to erase")
        
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
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = GestureController()
    controller.run()
    # whiteboard mathrm ollu
    # valolla cursor onddalloo