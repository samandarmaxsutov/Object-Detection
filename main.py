import cv2
import numpy as np
from ultralytics import YOLO

# Constants
MODEL_PATH = "best.pt"
VIDEO_SOURCE = 0
LINE_POSITION = 300
MAX_DISTANCE = 50
CONFIDENCE_THRESHOLD = 0.5

IMG_SIZE = 640
FRAME_WINDOW_NAME = "Object Detection and Counting"

# Initialize model
model = YOLO(MODEL_PATH)

# Initialize video capture
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Check if the video capture is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()


# Set the resolution to maximum
def set_max_resolution(cap):
    """Set the camera resolution to the maximum supported by the device."""
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print current resolution
    print(f"Current Resolution: {max_width}x{max_height}")

    # Try setting to higher resolution (if supported by the camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Example: 4K Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Example: 4K Height

    # Verify and print the new resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"New Resolution: {width}x{height}")


# Apply the max resolution settings
set_max_resolution(video_capture)

# Object tracking variables
object_id = 0
object_tracker = {}
object_count_yellow = 0
object_count_blue = 0
crossed_objects = set()

# Button coordinates
BUTTON_X1, BUTTON_Y1 = 10, 10
BUTTON_X2, BUTTON_Y2 = 130, 50


def reset_counters():
    """Reset all counters to zero."""
    global object_count_yellow, object_count_blue, crossed_objects, object_tracker, object_id
    object_count_yellow = 0
    object_count_blue = 0
    crossed_objects.clear()
    object_tracker.clear()
    object_id = 0
    print("Counters reset.")


def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_text(frame, text, position, color, font_scale=1, thickness=2):
    """Draw text on the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_bounding_box(frame, box, color, label):
    """Draw a bounding box with a label on the frame."""
    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_text(frame, label, (x1, y1 - 10), color)


def process_detections(detections, frame_width, frame):
    global object_id, object_count_blue, object_count_yellow
    current_objects = {}
    for result in detections:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls] if cls < len(model.names) else "Unknown"
            color = (0, 255, 0) if class_name == "yellow_wheel" else (255, 0, 0)

            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            matched_id = None
            for obj_id, prev_pos in object_tracker.items():
                if calculate_distance(prev_pos, center) < MAX_DISTANCE:
                    matched_id = obj_id
                    break
            if matched_id is None:
                object_id += 1
                matched_id = object_id

            object_tracker[matched_id] = center
            current_objects[matched_id] = center

            label = f"ID: {matched_id}, {class_name}: {conf:.2f}"
            draw_bounding_box(frame, box, color, label)

            if center[1] > LINE_POSITION and matched_id not in crossed_objects:
                crossed_objects.add(matched_id)
                if class_name == "yellow_wheel":
                    object_count_yellow += 1
                elif class_name == "blue_wheel":
                    object_count_blue += 1
    return current_objects


def draw_button(frame):
    """Draw a reset button on the frame."""
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (0, 255, 0), -1)
    draw_text(frame, "RESTART", (20, 35), (0, 0, 0), font_scale=0.8, thickness=2)


def button_clicked(x, y):
    """Check if the reset button was clicked."""
    return BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events."""
    if event == cv2.EVENT_LBUTTONDOWN and button_clicked(x, y):
        reset_counters()


# Set the mouse callback function for the window
cv2.namedWindow(FRAME_WINDOW_NAME)
cv2.setMouseCallback(FRAME_WINDOW_NAME, mouse_callback)

while True:
    ret, video_frame = video_capture.read()
    if not ret:
        print("Error: Couldn't read a frame or video ended.")
        break

    frame_height, frame_width = video_frame.shape[:2]
    results = model.predict(video_frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMG_SIZE)

    # Draw the line
    cv2.line(video_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (0, 0, 255), 2)

    # Process detections and update tracker
    object_tracker = process_detections(results, frame_width, video_frame)

    # Display counters
    total_count = object_count_blue + object_count_yellow
    draw_text(video_frame, f"Total: {total_count}", (10, 80), (255, 0, 255))
    draw_text(video_frame, f"Blue: {object_count_blue}", (10, 120), (255, 0, 0))
    draw_text(video_frame, f"Yellow: {object_count_yellow}", (10, 160), (0, 255, 255))

    # Draw the reset button
    draw_button(video_frame)

    # Show the frame
    cv2.imshow(FRAME_WINDOW_NAME, video_frame)

    # Check for quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
