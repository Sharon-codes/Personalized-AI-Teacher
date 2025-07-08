import cv2
import numpy as np
import time
import threading
import platform
import subprocess
import math
import pickle
import os
from datetime import datetime
from collections import deque

class AdvancedNavigationSystem:
    def __init__(self, calibration_file="navigation_calibration.pkl"):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Basic parameters
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        
        # Load or set default calibration values
        self.load_calibration(calibration_file)
        
        # Voice command system
        self.last_command_time = datetime.now()
        self.command_cooldown = 2.0
        self.is_running = True
        self.currently_speaking = False
        self.speak_lock = threading.Lock()
        self.system = platform.system()
        self.voice_enabled = True
        self.last_command = ""
        
        # Motion tracking variables
        self.prev_gray = None
        self.track_points = None
        self.min_distance = 10
        self.trajectory_history = deque(maxlen=30)  # Store recent motion
        
        # Obstacle tracking
        self.obstacle_history = {}  # Dictionary to track obstacles across frames
        self.next_obstacle_id = 0
        self.obstacle_persistence = 5  # Frames to keep tracking an obstacle
        
        # Path planning
        self.path_clear_threshold = 100  # Minimum width for a clear path
        self.emergency_stop_distance = 30  # Critical distance for stopping
        
        # Setup classifier for common objects (basic implementation)
        self.setup_object_classifier()
        
        # Announce system startup with voice
        self.speak("Advanced navigation system initialized")
    
    def load_calibration(self, calibration_file):
        """Load calibration or set defaults"""
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'rb') as f:
                    calibration = pickle.load(f)
                    
                self.extremely_close_threshold = calibration.get('extremely_close_threshold', 80)
                self.close_threshold = calibration.get('close_threshold', 150)
                self.medium_threshold = calibration.get('medium_threshold', 250)
                self.focal_length = calibration.get('focal_length', 500)
                self.avg_human_height = calibration.get('avg_human_height', 170)  # cm
                self.floor_level = calibration.get('floor_level', 450)  # y-coord of floor
                
                print("Calibration loaded successfully")
            except Exception as e:
                print(f"Error loading calibration: {e}")
                self.set_default_calibration()
        else:
            self.set_default_calibration()
    
    def set_default_calibration(self):
        """Set default calibration values"""
        # Distance thresholds (in pixels, later translated to real-world units)
        self.extremely_close_threshold = 80
        self.close_threshold = 150
        self.medium_threshold = 250
        
        # Camera parameters (will be used for better distance estimation)
        self.focal_length = 500  # approximated focal length
        self.avg_human_height = 170  # cm, used for scaling
        self.floor_level = 450  # y-coordinate of the floor in the image
        
        print("Using default calibration values")
    
    def save_calibration(self, calibration_file="navigation_calibration.pkl"):
        """Save current calibration values"""
        calibration = {
            'extremely_close_threshold': self.extremely_close_threshold,
            'close_threshold': self.close_threshold,
            'medium_threshold': self.medium_threshold,
            'focal_length': self.focal_length,
            'avg_human_height': self.avg_human_height,
            'floor_level': self.floor_level
        }
        
        try:
            with open(calibration_file, 'wb') as f:
                pickle.dump(calibration, f)
            print("Calibration saved successfully")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def calibrate_system(self, frames=30):
        """Interactive calibration procedure"""
        self.speak("Starting system calibration")
        print("Place a known object (like a person) at 1 meter distance and press 'c'")
        
        calibration_frames = []
        
        while len(calibration_frames) < frames:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Collecting calibration data ({len(calibration_frames)}/{frames})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'c' to capture frame", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                calibration_frames.append(frame)
                print(f"Frame {len(calibration_frames)}/{frames} captured")
                
            elif key == ord('q'):
                print("Calibration aborted")
                cv2.destroyWindow('Calibration')
                return False
        
        # Process calibration frames
        total_edges = 0
        edge_distances = []
        
        for frame in calibration_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find where edges are prominent
            edge_rows = np.where(np.sum(edges, axis=1) > 50)[0]
            if len(edge_rows) > 0:
                avg_edge_row = int(np.mean(edge_rows))
                distance_from_bottom = self.frame_height - avg_edge_row
                edge_distances.append(distance_from_bottom)
        
        if edge_distances:
            # Use the mode of edge distances for more robustness
            edge_distances.sort()
            median_distance = edge_distances[len(edge_distances) // 2]
            
            # Update calibration - this is for 1 meter distance
            self.close_threshold = median_distance
            self.extremely_close_threshold = int(median_distance * 0.5)  # 50cm
            self.medium_threshold = int(median_distance * 1.5)  # 1.5m
            
            cv2.destroyWindow('Calibration')
            self.speak("Calibration complete")
            self.save_calibration()
            return True
        else:
            cv2.destroyWindow('Calibration')
            self.speak("Calibration failed")
            return False
    
    def setup_object_classifier(self):
        """Setup basic object classifier using Haar cascades"""
        # Load pre-trained classifiers
        try:
            self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            print("Object classifiers loaded")
        except Exception as e:
            print(f"Error loading classifiers: {e}")
            self.person_cascade = None
            self.full_body_cascade = None
    
    def speak(self, text):
        """Speak the given text using platform-specific commands"""
        # Use a lock to prevent multiple speech commands at once
        if not self.voice_enabled:
            return
            
        with self.speak_lock:
            if self.currently_speaking:
                return
                
            self.currently_speaking = True
            
            try:
                if self.system == "Windows":
                    process = subprocess.Popen(
                        ["powershell", "-Command", 
                         f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}');"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                elif self.system == "Darwin":  # macOS
                    process = subprocess.Popen(["say", text], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                elif self.system == "Linux":
                    process = subprocess.Popen(["espeak", text],
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                
                # Wait for speech to complete
                process.wait()
            except Exception as e:
                print(f"Voice error: {e}")
            finally:
                self.currently_speaking = False
    
    def give_command(self, command, priority=False):
        """Give a navigation command if it's different from the last one or has priority"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_command_time).total_seconds()
        
        # Logic for when to issue commands
        should_issue = (
            (command != self.last_command) or  # New command
            (priority and time_diff >= 1.0) or  # Priority command with minimal cooldown
            (time_diff >= self.command_cooldown)  # Regular cooldown passed
        )
        
        if should_issue:
            self.last_command_time = current_time
            self.last_command = command
            
            # Print command with timestamp
            print(f"[{current_time.strftime('%H:%M:%S')}] {command}")
            
            # Start a new thread for speaking to avoid blocking
            if not self.currently_speaking:
                threading.Thread(target=self.speak, args=(command,), daemon=True).start()
    
    def estimate_distance(self, y_bottom, height=None):
        """
        Estimate real-world distance based on position in the image
        Higher values = further away
        """
        # Basic distance estimation based on position in the frame
        basic_distance = self.frame_height - y_bottom
        
        # If we have object height, use it for more accurate estimation
        if height and height > 10:  # Avoid division by zero or tiny values
            # Using the formula: real_distance = (object_real_height * focal_length) / object_pixel_height
            # This is a simplified version that needs proper calibration
            distance_from_height = (self.avg_human_height * self.focal_length) / height
            
            # Weighted combination
            return 0.7 * distance_from_height + 0.3 * basic_distance
        
        return basic_distance
    
    def classify_obstacle(self, frame, x, y, w, h):
        """Try to classify the type of obstacle"""
        # Extract region of interest
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return "unknown"
        
        # Convert to grayscale for classification
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Try to detect people
        if self.person_cascade:
            faces = self.person_cascade.detectMultiScale(roi_gray, 1.1, 4)
            if len(faces) > 0:
                return "person"
        
        if self.full_body_cascade:
            bodies = self.full_body_cascade.detectMultiScale(roi_gray, 1.1, 4)
            if len(bodies) > 0:
                return "person"
        
        # Basic texture analysis for static objects
        texture = np.std(roi_gray)
        
        if texture > 40:  # High texture variation
            return "complex_object"
        elif texture > 20:  # Medium texture variation
            return "simple_object"
        else:
            return "flat_surface"
    
    def update_motion_tracking(self, gray_frame):
        """Track motion between frames"""
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            # Initialize points for tracking - grid across the image
            y, x = np.mgrid[50:self.frame_height:30, 50:self.frame_width:30].reshape(2, -1)
            self.track_points = np.vstack((x, y)).T.astype(np.float32).reshape(-1, 1, 2)
            return None
        
        # Calculate optical flow using Lucas-Kanade method
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.track_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Select good points
        good_new = new_points[status == 1]
        good_old = self.track_points[status == 1]
        
        # Calculate motion vectors
        if len(good_new) > 0 and len(good_old) > 0:
            motion_vectors = good_new - good_old
            avg_motion = np.mean(motion_vectors, axis=0).reshape(-1)
            
            if not np.isnan(avg_motion).any():
                self.trajectory_history.append(avg_motion)
        
        # Update for next iteration
        self.prev_gray = gray_frame.copy()
        self.track_points = good_new.reshape(-1, 1, 2)
        
        # Reinitialize tracking points if too few remain
        if len(self.track_points) < 10:
            y, x = np.mgrid[50:self.frame_height:30, 50:self.frame_width:30].reshape(2, -1)
            self.track_points = np.vstack((x, y)).T.astype(np.float32).reshape(-1, 1, 2)
    
    def draw_motion_flow(self, frame):
        """Draw motion flow vectors on the frame"""
        if len(self.trajectory_history) > 0:
            # Calculate average motion vector over recent history
            avg_motion = np.mean(list(self.trajectory_history), axis=0)
            
            # Draw arrow from center of frame
            start_point = (self.frame_center_x, self.frame_height // 2)
            # Scale the motion vector for visibility
            end_point = (
                int(start_point[0] + avg_motion[0] * 10),
                int(start_point[1] + avg_motion[1] * 10)
            )
            
            # Draw the arrow
            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 255), 2, tipLength=0.3)
            
            # Add motion text
            motion_mag = np.linalg.norm(avg_motion)
            cv2.putText(frame, f"Motion: {motion_mag:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    def detect_obstacles(self, frame):
     """Advanced obstacle detection with tracking and classification with improved wall detection"""
     # Convert to grayscale
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
     # Update motion tracking
     self.update_motion_tracking(gray)
    
     # Apply adaptive thresholding for better edge detection in varying lighting
     _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
     # Apply more sophisticated edge detection
     # First apply bilateral filter to reduce noise while preserving edges
     blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
     # Then use Canny edge detector with automatic threshold calculation
     median = np.median(blurred)
     sigma = 0.33
     lower = int(max(0, (1.0 - sigma) * median))
     upper = int(min(255, (1.0 + sigma) * median))
     edges = cv2.Canny(blurred, lower, upper)
    
    # Dilate edges to connect nearby edges
     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
     dilated = cv2.dilate(edges, kernel, iterations=1)
    
     # Find contours
     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
     # Filter out small contours (noise)
     significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 800]
    
     # Current frame obstacles
     current_obstacles = []
    
     # ======================
     # Add specific wall detection
     # ======================
    
     # Line detection using Hough transform for identifying walls
     lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100, 
                          minLineLength=100, maxLineGap=20)
    
    # Process detected lines if any
     if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate length of the line
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Skip short lines
            if line_length < 100:
                continue
                
            # Calculate slope to determine if horizontal or vertical
            # Use very small value to avoid division by zero
            if abs(x2 - x1) < 0.001:
                slope = float('inf')  # Vertical line
            else:
                slope = abs((y2 - y1) / (x2 - x1))
            
            # Determine if the line is more horizontal or vertical
            is_vertical = slope > 1.0
            
            # Create pseudo-contour for wall-like lines
            # Make width of wall-like object proportional to detected line
            wall_width = 30  # Default width in pixels
            wall_box_x = min(x1, x2) - wall_width//2
            wall_box_y = min(y1, y2)
            wall_box_w = max(x1, x2) - min(x1, x2) + wall_width
            wall_box_h = max(y1, y2) - min(y1, y2) + 10
            
            # Ensure wall boxes have reasonable dimensions
            if is_vertical:
                # For vertical walls, make sure width is reasonable
                wall_box_w = max(wall_box_w, 30)
            else:
                # For horizontal walls, make sure height is reasonable
                wall_box_h = max(wall_box_h, 30)
            
            # Estimate distance based on position in frame
            # For walls, use bottom of the wall
            y_bottom = max(y1, y2)
            distance = self.estimate_distance(y_bottom)
            
            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Add to obstacles
            obstacle = {
                'id': None,
                'distance': distance,
                'center_x': center_x,
                'center_y': center_y,
                'box': (wall_box_x, wall_box_y, wall_box_w, wall_box_h),
                'type': 'wall',
                'last_seen': 0,
                'velocity': [0, 0]
            }
            
            current_obstacles.append(obstacle)
            
            # Visualize the detected wall line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
      # ======================
      # Process regular obstacles (from contours)
      # ======================
     for contour in significant_contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip if the object is too small
        if w < 20 or h < 20:
            continue
            
        # Check for wall-like characteristics
        is_wall_like = False
        aspect_ratio = w / max(h, 1)  # Avoid division by zero
        
        # Wall-like if very wide and not very tall, or very tall and not very wide
        if (aspect_ratio > 5 and h > 30) or (aspect_ratio < 0.2 and w > 30):
            obstacle_type = "wall"
            is_wall_like = True
        else:
            # Use regular classification
            obstacle_type = self.classify_obstacle(frame, x, y, w, h)
        
        # Estimate real-world distance
        y_bottom = y + h  # Bottom of the object
        distance = self.estimate_distance(y_bottom, h)
        
        # Calculate the center of the contour
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Create obstacle info
        obstacle = {
            'id': None,  # Will be assigned/matched later
            'distance': distance,
            'center_x': center_x,
            'center_y': center_y,
            'box': (x, y, w, h),
            'type': obstacle_type,
            'last_seen': 0,  # Frame counter
            'velocity': [0, 0]  # Initial velocity
        }
        
        current_obstacles.append(obstacle)
        
        # Determine color based on distance and obstacle type
        if is_wall_like:
            # Special color for walls - purple
            color = (255, 0, 255)
        elif distance <= self.extremely_close_threshold:
            # Red for extremely close
            color = (0, 0, 255)
        elif distance <= self.close_threshold:
            # Orange for close objects
            color = (0, 165, 255)
        else:
            # Green for farther obstacles
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add distance and type information
        cv2.putText(frame, f"D:{int(distance)}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, obstacle_type, (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
      # Track obstacles across frames
     self.track_obstacles(current_obstacles)
    
     return frame, current_obstacles

    def classify_obstacle(self, frame, x, y, w, h):
     """Try to classify the type of obstacle with improved wall detection"""
     # Extract region of interest
     roi = frame[y:y+h, x:x+w]
     if roi.size == 0:
        return "unknown"
    
     # Convert to grayscale for classification
     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Check for wall-like characteristics
     aspect_ratio = w / max(h, 1)  # Avoid division by zero
    
    # Wall detection heuristics
     if (w > 150 and h < 50) or (h > 150 and w < 50):
        # Long thin objects are likely walls or barriers
        return "wall"
    
     # Check for flatness using edge detection
     edges = cv2.Canny(roi_gray, 50, 150)
     edge_density = np.sum(edges) / (w * h)
    
    # Walls often have strong edges but only in specific directions
     if edge_density > 0.05 and edge_density < 0.3:
        # Detect if edges are primarily horizontal or vertical
        # This is a simplistic approach - could be improved with Hough lines
        horiz_sum = np.sum(edges, axis=1)
        vert_sum = np.sum(edges, axis=0)
        
        horiz_variance = np.var(horiz_sum)
        vert_variance = np.var(vert_sum)
        
        if (horiz_variance < 0.2 * vert_variance) or (vert_variance < 0.2 * horiz_variance):
            return "wall"
    
    # Try to detect people
     if self.person_cascade:
        faces = self.person_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(faces) > 0:
            return "person"
    
     if self.full_body_cascade:
        bodies = self.full_body_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(bodies) > 0:
            return "person"
    
     # Basic texture analysis for static objects
     texture = np.std(roi_gray)
    
     if texture > 40:  # High texture variation
        return "complex_object"
     elif texture > 20:  # Medium texture variation
        return "simple_object"
     else:
        # Low texture could also indicate a wall or flat surface
        return "flat_surface"
    
    def track_obstacles(self, current_obstacles):
        """Track obstacles across frames to calculate their velocity and predict movement"""
        # First, try to match current obstacles with tracked ones
        for obstacle in current_obstacles:
            best_match = None
            min_distance = float('inf')
            
            for tracked_id, tracked_obstacle in list(self.obstacle_history.items()):
                # Skip if the tracked obstacle is too old
                if tracked_obstacle['last_seen'] > self.obstacle_persistence:
                    continue
                    
                # Calculate Euclidean distance between centers
                dx = obstacle['center_x'] - tracked_obstacle['center_x']
                dy = obstacle['center_y'] - tracked_obstacle['center_y']
                distance = math.sqrt(dx*dx + dy*dy)
                
                # If close enough, consider it a match
                if distance < 50 and distance < min_distance:
                    min_distance = distance
                    best_match = tracked_id
            
            if best_match is not None:
                # Update existing obstacle
                old_x = self.obstacle_history[best_match]['center_x']
                old_y = self.obstacle_history[best_match]['center_y']
                
                # Calculate velocity (pixels/frame)
                vx = obstacle['center_x'] - old_x
                vy = obstacle['center_y'] - old_y
                
                # Update with exponential smoothing for stability
                old_vx, old_vy = self.obstacle_history[best_match]['velocity']
                smooth_vx = 0.7 * vx + 0.3 * old_vx
                smooth_vy = 0.7 * vy + 0.3 * old_vy
                
                # Update tracked obstacle
                self.obstacle_history[best_match].update({
                    'center_x': obstacle['center_x'],
                    'center_y': obstacle['center_y'],
                    'box': obstacle['box'],
                    'distance': obstacle['distance'],
                    'type': obstacle['type'],
                    'last_seen': 0,
                    'velocity': [smooth_vx, smooth_vy]
                })
                
                # Set the ID in the current detection
                obstacle['id'] = best_match
            else:
                # New obstacle detected
                new_id = self.next_obstacle_id
                self.next_obstacle_id += 1
                
                # Add to tracking
                self.obstacle_history[new_id] = {
                    'center_x': obstacle['center_x'],
                    'center_y': obstacle['center_y'],
                    'box': obstacle['box'],
                    'distance': obstacle['distance'],
                    'type': obstacle['type'],
                    'last_seen': 0,
                    'velocity': [0, 0]  # Initial velocity
                }
                
                # Set the ID in the current detection
                obstacle['id'] = new_id
        
        # Increment last_seen for all tracked obstacles
        for tracked_id in list(self.obstacle_history.keys()):
            self.obstacle_history[tracked_id]['last_seen'] += 1
            
            # Remove obstacles that haven't been seen for too long
            if self.obstacle_history[tracked_id]['last_seen'] > self.obstacle_persistence:
                del self.obstacle_history[tracked_id]
    
    def analyze_path_safety(self, obstacles):
        """Analyze potential navigation paths for safety"""
        # Create a horizontal safety map (represents the "danger" at each horizontal position)
        safety_map = np.zeros(self.frame_width)
        
        # For each obstacle, increase danger level based on proximity
        for obstacle in obstacles:
            x, y, w, h = obstacle['box']
            distance = obstacle['distance']
            
            # Calculate danger influence range (wider for closer obstacles)
            influence_width = w + int((self.close_threshold / max(distance, 1)) * 50)
            
            # Calculate danger intensity (higher for closer obstacles)
            danger = 1.0 - min(1.0, distance / self.close_threshold)
            
            # Apply danger to the safety map
            left_bound = max(0, x - influence_width // 4)
            right_bound = min(self.frame_width - 1, x + w + influence_width // 4)
            
            for i in range(left_bound, right_bound + 1):
                # Closer to the center of the obstacle = more dangerous
                center_distance = abs(i - (x + w/2))
                influence_factor = 1.0 - min(1.0, center_distance / (influence_width/2))
                safety_map[i] += danger * influence_factor
        
        # Normalize the safety map
        if np.max(safety_map) > 0:
            safety_map = safety_map / np.max(safety_map)
        
        # Find the safest path
        min_danger = np.min(safety_map)
        safest_points = np.where(safety_map == min_danger)[0]
        
        # Analyze left vs right half of the frame
        left_half_danger = np.mean(safety_map[:self.frame_center_x])
        right_half_danger = np.mean(safety_map[self.frame_center_x:])
        safer_side = "left" if left_half_danger < right_half_danger else "right"
        
        if len(safest_points) > 0:
            # Use the middle of the safest region
            safest_x = (safest_points[0] + safest_points[-1]) // 2
            safest_width = safest_points[-1] - safest_points[0]
            
            return {
                'safest_x': safest_x,
                'danger_level': min_danger,
                'path_width': safest_width,
                'safety_map': safety_map,
                'left_danger': left_half_danger,
                'safer_side': safer_side
            }
        else:
              return {
            'safest_x': self.frame_center_x,  # Default to center
            'danger_level': 0.5,  # Moderate danger when uncertain
            'path_width': 0,
            'safety_map': safety_map,
            'left_danger': left_half_danger,
            'right_danger': right_half_danger,
            'safer_side': safer_side
        }

    def generate_navigation_command(self, safety_analysis, obstacles):
     """Generate voice navigation commands based on safety analysis with improved wall detection"""
     safest_x = safety_analysis['safest_x']
     danger_level = safety_analysis['danger_level']
     path_width = safety_analysis['path_width']
     safer_side = safety_analysis['safer_side']
    
     # Calculate deviation from center
     deviation = safest_x - self.frame_center_x
    
     # Find the closest obstacle
     closest_distance = float('inf')
     closest_type = "unknown"
     closest_position = "unknown"  # Track if obstacle is left, right, or center
    
     # Flag for wall detection
     wall_ahead = False
     wall_position = "unknown"
    
     for obstacle in obstacles:
        # Check specifically for walls
        if obstacle['type'] == 'wall' or obstacle['type'] == 'flat_surface':
            # Position of the wall
            wall_center_x = obstacle['center_x']
            
            # Check if wall is in the center of field of view
            if abs(wall_center_x - self.frame_center_x) < 100:
                wall_ahead = True
                wall_position = "ahead"
            else:
                wall_position = "left" if wall_center_x < self.frame_center_x else "right"
            
            # If the wall is close, it's a priority
            if obstacle['distance'] < self.close_threshold:
                closest_distance = min(closest_distance, obstacle['distance'])
                closest_type = "wall"
                closest_position = wall_position
        
        # Handle other obstacles
        elif obstacle['distance'] < closest_distance:
            closest_distance = obstacle['distance']
            closest_type = obstacle['type']
            
            # Determine position relative to center
            obstacle_center_x = obstacle['center_x']
            if abs(obstacle_center_x - self.frame_center_x) < 50:
                closest_position = "center"
            else:
                closest_position = "left" if obstacle_center_x < self.frame_center_x else "right"
    
     # Emergency stop condition - prioritize walls
     if wall_ahead and closest_distance <= self.close_threshold:
        # Wall directly ahead - find the best direction
        self.give_command(f"Stop! Wall directly ahead, move {safer_side}!", priority=True)
        return safer_side
    
    # Wall to one side but very close
     elif closest_type == "wall" and closest_distance <= self.extremely_close_threshold:
        # Recommend opposite direction of wall
        recommended_direction = "right" if closest_position == "left" else "left"
        self.give_command(f"Caution! Wall very close on {closest_position}, move {recommended_direction}!", priority=True)
        return recommended_direction
    
    # Emergency stop for other obstacles
     elif closest_distance <= self.emergency_stop_distance:
        if closest_position == "center":
            # If obstacle is directly ahead and very close, recommend the safer side
            self.give_command(f"Stop! {closest_type} directly ahead, move {safer_side}!", priority=True)
            return safer_side
        else:
            # If obstacle is to one side, recommend the opposite direction
            recommended_direction = "right" if closest_position == "left" else "left"
            self.give_command(f"Stop! {closest_type} very close on {closest_position}, move {recommended_direction}!", priority=True)
            return recommended_direction
    
    # Object very close - careful navigation needed
     elif closest_distance <= self.extremely_close_threshold:
        if closest_position == "center":
            # Recommend the safer side based on analysis
            self.give_command(f"Caution! {closest_type} ahead, move {safer_side}!", priority=True)
            return safer_side
        else:
            # Recommend opposite direction of obstacle
            recommended_direction = "right" if closest_position == "left" else "left"
            self.give_command(f"Caution! {closest_type} on {closest_position}, move {recommended_direction}!", priority=True)
            return recommended_direction
    
    # Narrow path ahead
     elif path_width < self.path_clear_threshold:
        if abs(deviation) > 50:
            direction = "right" if deviation < 0 else "left"
            self.give_command(f"Narrow path, move {direction}")
            return direction
        else:
            self.give_command("Narrow path ahead, proceed slowly")
            return "slow"
    
    # Clear path but need to adjust direction
     elif abs(deviation) > 100:
        direction = "right" if deviation < 0 else "left"
        self.give_command(f"Clear path to the {direction}")
        return direction
    
    # Moderate danger, be cautious
     elif danger_level > 0.3:
        self.give_command("Path clear, proceed with caution")
        return "proceed_caution"
    
    # All clear
     else:
        if datetime.now().second % 10 == 0:  # Only occasionally give "all clear"
            self.give_command("Path clear")
        return "proceed"

    def visualize_safety_map(self, frame, safety_analysis):
     """Visualize the safety map and suggested path"""
     safety_map = safety_analysis['safety_map']
     safest_x = safety_analysis['safest_x']
     path_width = safety_analysis['path_width']
     safer_side = safety_analysis['safer_side']
    
     # Draw safety heatmap at the bottom of the frame
     heatmap_height = 30
     heatmap_y = self.frame_height - heatmap_height
    
     for x in range(self.frame_width):
        # Red intensity based on danger level
        danger = safety_map[x]
        color_intensity = int(255 * danger)
        color = (0, 255 - color_intensity, color_intensity)
        
        # Draw vertical line representing danger at this x-coordinate
        cv2.line(frame, (x, heatmap_y), (x, heatmap_y + heatmap_height), color, 1)
    
      # Draw the safest path
     cv2.line(frame, (safest_x, heatmap_y), (safest_x, self.frame_height), (0, 255, 0), 2)
    
     # Draw path width indicator if significant
     if path_width > self.path_clear_threshold:
        left_x = safest_x - path_width // 2
        right_x = safest_x + path_width // 2
        
        cv2.line(frame, (left_x, heatmap_y + heatmap_height // 2), 
                (right_x, heatmap_y + heatmap_height // 2), (0, 255, 0), 2)
    
     # Add text for danger level
     danger_text = f"Danger: {safety_analysis['danger_level']:.2f}"
     cv2.putText(frame, danger_text, (10, heatmap_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
     # Add safer side indicator
     safer_text = f"Safer side: {safer_side}"
     cv2.putText(frame, safer_text, (self.frame_width - 200, heatmap_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def run(self):
        """Main processing loop"""
        try:
            self.speak("Starting navigation assistance")
            
            frame_count = 0
            start_time = time.time()
            
            while self.is_running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading from camera")
                    time.sleep(0.1)
                    continue
                
                # Process every other frame for performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                    
                # Flip the frame horizontally for a more intuitive view
                frame = cv2.flip(frame, 1)
                
                # Detect obstacles
                processed_frame, obstacles = self.detect_obstacles(frame)
                
                # Analyze path safety
                safety_analysis = self.analyze_path_safety(obstacles)
                
                # Visualize safety map
                self.visualize_safety_map(processed_frame, safety_analysis)
                
                # Draw motion flow
                self.draw_motion_flow(processed_frame)
                
                # Generate navigation command
                navigation_command = self.generate_navigation_command(safety_analysis, obstacles)
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Show the processed frame
                cv2.imshow('Advanced Navigation System', processed_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.speak("Navigation system shutting down")
                    break
                elif key == ord('c'):
                    self.calibrate_system()
                elif key == ord('v'):
                    self.voice_enabled = not self.voice_enabled
                    status = "enabled" if self.voice_enabled else "disabled"
                    print(f"Voice commands {status}")
                    
        except Exception as e:
            print(f"Error in navigation system: {e}")
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            print("Navigation system stopped")
    
    def stop(self):
        """Stop the navigation system"""
        self.is_running = False


if __name__ == "__main__":
    try:
        # Create and run the navigation system
        nav_system = AdvancedNavigationSystem()
        nav_system.run()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")