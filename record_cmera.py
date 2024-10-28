import cv2

# Initialize video capture from the default camera
camera = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set camera to maximum resolution (modify if needed)
def set_max_resolution(cap):
    """Set the camera to the highest supported resolution."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Example 4K width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Example 4K height

    # Print current resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Recording at: {width}x{height}")

# Apply maximum resolution
set_max_resolution(camera)

# Get resolution and frame rate (for saving video correctly)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(camera.get(cv2.CAP_PROP_FPS)) or 30  # Use 30 FPS if unable to detect

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_file = "camera_recording2.mp4"  # Output filename

video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

print("Recording... Press 'q' to stop.")

# Start recording loop
while True:
    ret, frame = camera.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read the frame.")
        break

    # Write the frame to the output file
    video_writer.write(frame)

    # Display the frame in a window
    cv2.imshow('Recording...', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped.")
        break

# Release resources
camera.release()
video_writer.release()
cv2.destroyAllWindows()
