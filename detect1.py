import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import subprocess
import platform
from datetime import datetime
from collections import deque

class AdvancedNavigationSystem:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Frame parameters
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2

        # Navigation parameters
        self.close_threshold = 150  # Pixels (approx 1 meter)
        self.extremely_close_threshold = 80  # Pixels (approx 0.5 meter)
        self.focal_length = 500  # For distance estimation
        self.avg_object_height = 170  # cm, for scaling

        # System variables
        self.is_running = True
        self.voice_enabled = True
        self.speak_lock = threading.Lock()
        self.system = platform.system()
        self.command_queue = deque(maxlen=5)

        # Load SSD MobileNet model
        self.model = self.load_ssd_mobilenet()
        self.class_names = self.load_coco_classes()

        # Start command processing thread
        threading.Thread(target=self.process_commands, daemon=True).start()
        self.speak("Advanced navigation system initialized")

    def load_ssd_mobilenet(self):
        # Load pre-trained SSD MobileNet model from TensorFlow Hub
        model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        # Note: Download and extract this model manually, then load it
        # For simplicity, assuming it's downloaded and placed in './ssd_mobilenet_v2'
        model_path = './ssd_mobilenet_v2/saved_model'
        return tf.saved_model.load(model_path)

    def load_coco_classes(self):
        # Load COCO class names (91 classes)
        with open('coco.names', 'r') as f:  # Download coco.names from https://github.com/pjreddie/darknet/blob/master/data/coco.names
            return [line.strip() for line in f.readlines()]

    def speak(self, text):
        if not self.voice_enabled or not text:
            return
        with self.speak_lock:
            try:
                if self.system == "Windows":
                    subprocess.Popen(
                        ["powershell", "-Command", 
                         f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}');"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                elif self.system == "Darwin":  # macOS
                    subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif self.system == "Linux":
                    subprocess.Popen(["espeak", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass

    def process_commands(self):
        while self.is_running:
            if self.command_queue:
                cmd = self.command_queue.popleft()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {cmd}")
                self.speak(cmd)
            time.sleep(0.5)

    def detect_lines(self, frame):
        # Convert to HSV for line detection (blue path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 120, 0])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Edge detection
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)

        if lines is not None:
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:  # Ignore near-horizontal lines
                    continue
                if slope > 0:
                    right_lines.append((x1, y1, x2, y2))
                else:
                    left_lines.append((x1, y1, x2, y2))

            # Calculate steering angle
            if left_lines and right_lines:
                left_x = np.mean([line[0] + line[2] for line in left_lines]) // 2
                right_x = np.mean([line[0] + line[2] for line in right_lines]) // 2
                offset = ((left_x + right_x) - self.frame_width) / self.frame_height
                angle = np.arctan(offset) * 180 / np.pi
                return angle, left_lines, right_lines
        return 0, [], []  # Default: no turn

    def detect_objects(self, frame):
        # Prepare frame for SSD MobileNet
        input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
        detections = self.model(input_tensor)

        # Extract detection results
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)

        obstacles = []
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * self.frame_width)
                y = int(ymin * self.frame_height)
                w = int((xmax - xmin) * self.frame_width)
                h = int((ymax - ymin) * self.frame_height)
                
                # Estimate distance
                distance = self.estimate_distance(y + h, h)
                center_x = x + w // 2
                
                obstacles.append({
                    'box': (x, y, w, h),
                    'distance': distance,
                    'center_x': center_x,
                    'type': self.class_names[classes[i] - 1]  # -1 because COCO classes start at 1
                })
        return obstacles

    def estimate_distance(self, y_bottom, height):
        # Simplified distance estimation
        if height > 10:  # Avoid division by small numbers
            return (self.avg_object_height * self.focal_length) / height
        return self.frame_height - y_bottom

    def generate_navigation_command(self, angle, obstacles):
        # Line-based navigation
        if abs(angle) > 10:
            direction = "right" if angle < 0 else "left"
            cmd = f"Turn {direction} {abs(int(angle))} degrees"
            self.command_queue.append(cmd)
            return direction

        # Obstacle-based navigation
        if obstacles:
            closest = min(obstacles, key=lambda x: x['distance'])
            distance = closest['distance']
            obj_type = closest['type']
            pos = "left" if closest['center_x'] < self.frame_center_x else "right"

            if distance <= self.extremely_close_threshold:
                direction = "right" if pos == "left" else "left"
                cmd = f"Stop! {obj_type} very close on {pos}, move {direction}!"
                self.command_queue.append(cmd)
                return direction
            elif distance <= self.close_threshold:
                direction = "right" if pos == "left" else "left"
                cmd = f"Caution! {obj_type} at {int(distance/50)} meters on {pos}, move {direction}"
                self.command_queue.append(cmd)
                return direction

        if not self.command_queue and datetime.now().second % 5 == 0:
            self.command_queue.append("Path clear")
        return "proceed"

    def visualize(self, frame, angle, left_lines, right_lines, obstacles):
        # Draw lines
        for line in left_lines + right_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw obstacles
        for obs in obstacles:
            x, y, w, h = obs['box']
            color = (0, 0, 255) if obs['distance'] <= self.close_threshold else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{obs['type']} D:{int(obs['distance']/50)}m", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw steering direction
        cv2.putText(frame, f"Angle: {int(angle)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return frame

    def run(self):
        self.speak("Starting navigation")
        fps_time = time.time()
        frame_count = 0

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)

                # Detect lines and objects
                angle, left_lines, right_lines = self.detect_lines(frame)
                obstacles = self.detect_objects(frame)

                # Generate command and visualize
                self.generate_navigation_command(angle, obstacles)
                processed_frame = self.visualize(frame, angle, left_lines, right_lines, obstacles)

                # FPS calculation
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                cv2.imshow('Advanced Navigation', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.speak("Navigation system shutting down")
                    break

        finally:
            self.is_running = False
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        nav = AdvancedNavigationSystem()
        nav.run()
    except Exception as e:
        print(f"Error: {e}")