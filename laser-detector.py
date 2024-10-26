import cv2
import numpy as np
from collections import deque

class HighPrecisionLaserDetector:
    def __init__(self):
        # Initialize camera object
        self.cap = cv2.VideoCapture(0)  # 0 indicates default camera
        # Set camera resolution to 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # Set frame rate to 60fps to increase sampling frequency
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # Disable auto-exposure (0.25 is the magic number for disabling)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Set fixed exposure value to -8 to enhance laser point contrast against background
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)

        # Define detection parameters dictionary
        self.params = {
            # Red color range in HSV color space (red spans across both ends of HSV wheel)
            # H: Hue, S: Saturation, V: Value
            # First red range: 0-15 degrees
            'red_lower1': np.array([0, 120, 160]),    # Higher saturation and value thresholds to reduce interference
            'red_upper1': np.array([15, 255, 255]),   # Upper limit set to maximum
            # Second red range: 160-180 degrees
            'red_lower2': np.array([160, 120, 160]),  # Same saturation and value requirements
            'red_upper2': np.array([180, 255, 255]),  # 180 is maximum H value in HSV
            'min_area': 3,       # Minimum area of laser point (in pixels), set small for high sensitivity
            'max_area': 150,     # Maximum area to prevent false detection of large red objects
            'min_intensity': 200 # Minimum brightness threshold, laser points are typically very bright
        }

        # Smoothing parameters
        # Use deque (double-ended queue) to store positions from last 3 frames for smoothing
        self.position_buffer = deque(maxlen=3)
        # Store last valid detection result
        self.last_detection = None
        # Define position jump threshold (in pixels), jumps larger than this are considered false detections
        self.stability_threshold = 50

        # Define display colors (BGR format)
        self.colors = {
            'target': (0, 255, 0),  # Green, for detection box and crosshair
            'info': (255, 255, 255) # White, for text information
        }

    def preprocess_frame(self, frame):
        """
        Preprocess input frame to enhance laser point visibility
        Args:
            frame: Input image frame
        Returns:
            enhanced: Enhanced image frame
        """
        # Convert image to LAB color space, L channel represents brightness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        l, a, b = cv2.split(lab)
        # Create CLAHE object (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # Apply CLAHE to brightness channel
        cl = clahe.apply(l)
        # Merge processed channels
        enhanced = cv2.merge((cl,a,b))
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_laser(self, frame):
        """
        Detect laser point in the image
        Args:
            frame: Input image frame
        Returns:
            best_detection: Detection result dictionary containing position, area, score, etc.; 
                          None if no detection
        """
        try:
            # Enhance image through preprocessing
            enhanced = self.preprocess_frame(frame)
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Create masks for both red regions
            mask1 = cv2.inRange(hsv, self.params['red_lower1'], self.params['red_upper1'])
            mask2 = cv2.inRange(hsv, self.params['red_lower2'], self.params['red_upper2'])
            # Combine both red masks
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Extract brightness channel
            v_channel = hsv[:, :, 2]
            # Create brightness mask
            _, bright_mask = cv2.threshold(v_channel, 
                                        self.params['min_intensity'], 
                                        255, 
                                        cv2.THRESH_BINARY)

            # Combine red and brightness masks - areas must satisfy both color and brightness conditions
            final_mask = cv2.bitwise_and(red_mask, bright_mask)

            # Use morphological operations to remove noise
            kernel = np.ones((3,3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(final_mask, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)

            # Initialize best detection results
            best_detection = None
            max_score = 0

            # Iterate through all contours to find best candidate
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                # Check if area is within valid range
                if self.params['min_area'] <= area <= self.params['max_area']:
                    # Calculate contour perimeter
                    perimeter = cv2.arcLength(contour, True)
                    # Calculate circularity (perfect circle = 1)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                    # Create mask for intensity calculation
                    mask = np.zeros_like(v_channel)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    # Calculate average brightness of the region
                    mean_intensity = cv2.mean(v_channel, mask=mask)[0]
                    
                    # Calculate contour's bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    # Calculate density (ratio of actual area to bounding box area)
                    density = area / (w * h) if w * h > 0 else 0

                    # Calculate composite score
                    score = (circularity * 0.3 +        # Circularity weight
                            (mean_intensity/255) * 0.4 + # Brightness weight
                            density * 0.3)               # Density weight

                    # Update best detection result
                    if score > max_score:
                        # Calculate contour moments
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            # Calculate contour center point
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            max_score = score
                            best_detection = {
                                'center': (cx, cy),
                                'area': area,
                                'score': score,
                                'intensity': mean_intensity
                            }

            return best_detection

        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None

    def stabilize_detection(self, detection):
        """
        Apply smoothing to detection results to reduce jitter
        Args:
            detection: Current frame detection result
        Returns:
            Processed detection result
        """
        if detection is None:
            # Clear buffer if no detection in current frame
            self.position_buffer.clear()
            return self.last_detection

        # Get current detection position
        current_pos = detection['center']

        # Accept new detection directly if buffer is empty
        if not self.position_buffer:
            self.position_buffer.append(current_pos)
            self.last_detection = detection
            return detection

        # Get last position
        last_pos = self.position_buffer[-1]
        # Calculate Euclidean distance from last position
        distance = np.sqrt((current_pos[0]-last_pos[0])**2 + 
                         (current_pos[1]-last_pos[1])**2)

        # Likely false detection or interference if position change is too large
        if distance > self.stability_threshold:
            return self.last_detection

        # Update position buffer
        self.position_buffer.append(current_pos)
        
        # Apply smoothing if enough history positions are available
        if len(self.position_buffer) >= 2:
            # Use weighted average of last two frames, giving more weight to current frame
            weights = [0.7, 0.3]  # Current frame: 0.7, previous frame: 0.3
            # Calculate weighted average for x coordinate
            x = int(np.average([p[0] for p in list(self.position_buffer)[-2:]], 
                             weights=weights))
            # Calculate weighted average for y coordinate
            y = int(np.average([p[1] for p in list(self.position_buffer)[-2:]], 
                             weights=weights))
            # Update position in detection result
            detection['center'] = (x, y)

        # Save current detection result
        self.last_detection = detection
        return detection

    def draw_detection(self, frame, detection):
        """
        Draw detection results on the image
        Args:
            frame: Input image frame
            detection: Detection result
        Returns:
            Processed image frame
        """
        if detection is None:
            return frame

        try:
            # Get detection point coordinates and score
            x, y = detection['center']
            score = detection.get('score', 0)

            # Dynamically adjust display box size based on detection score
            box_size = int(20 + score * 10)

            # 1. Draw crosshair
            cv2.line(frame, (x-10, y), (x+10, y), self.colors['target'], 2)
            cv2.line(frame, (x, y-10), (x, y+10), self.colors['target'], 2)

            # 2. Draw detection box
            cv2.rectangle(frame,
                         (x - box_size, y - box_size),
                         (x + box_size, y + box_size),
                         self.colors['target'], 2)

            # 3. Display detection information
            info_text = f"Position: ({x}, {y})"  # Position information
            score_text = f"Score: {score:.2f}"   # Score information

            # Draw text information on image
            cv2.putText(frame, info_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.colors['info'], 2)
            cv2.putText(frame, score_text, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.colors['info'], 2)

        except Exception as e:
            print(f"Drawing error: {str(e)}")

        return frame

    def run(self):
        """Main detection program loop"""
        print("Press 'q' to quit")
        print("Press 'r' to reset detection")

        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Perform laser point detection
                detection = self.detect_laser(frame)
                
                # Apply smoothing to high confidence detections
                if detection and detection.get('score', 0) > 0.5:
                    detection = self.stabilize_detection(detection)
                
                # Draw detection results on image
                frame = self.draw_detection(frame, detection)
                
                # Display processed image
                cv2.imshow('High Precision Laser Detection', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press q to quit
                    break
                elif key == ord('r'):  # Press r to reset detection
                    self.position_buffer.clear()
                    self.last_detection = None

        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    detector = HighPrecisionLaserDetector()
    detector.run()