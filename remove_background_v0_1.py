import cv2
import numpy as np
from halo import Halo

def remove_green_background(frame):
    # Define the lower and upper bounds of the green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert the mask so that the background is black
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to extract the foreground
    result = cv2.bitwise_and(frame, frame, mask=mask_inv)

    return result

@Halo(text='Processing video...', text_color='cyan', spinner='dots', color='cyan')
def process_video(input_video, output_video, background_video):
    # Open the input video file
    cap = cv2.VideoCapture(input_video)
    
    # Open the background video file
    bg_cap = cv2.VideoCapture(background_video)

    # Get the video's frames per second (fps) and frame size
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Read the first frame from the background video
    ret_bg, bg_frame = bg_cap.read()
    if not ret_bg:
        raise ValueError("Background video is empty")

    # Resize the background video to match the dimensions of the input video
    bg_frame = cv2.resize(bg_frame, (width, height))

    # Process each frame of the input video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = remove_green_background(frame)

        # Combine the processed frame with the resized background frame
        result_frame = cv2.add(processed_frame, bg_frame)

        out.write(result_frame)

    # Release video capture and writer objects
    cap.release()
    bg_cap.release()
    out.release()
    
if __name__ == "__main__":
    input_video_path = "input_video.mp4"
    output_video_path = "output_video.mp4"
    background_video_path = "background_video.mp4"

    process_video(input_video_path, output_video_path, background_video_path)
