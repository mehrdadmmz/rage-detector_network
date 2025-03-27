# please modify codes below comments #1, 2, 3, 4 accordingly

import cv2
import os

# Modify these 3 buttons accordingly 
MAX_FRAME_COUNT = 90
EDITOR = "Arthur"   # Arthur, Daniel, Mehrdad
VIDEO_TYPE = "rage" # rage or non_rage

def videoToFrames(video_path, output_folder, start_frame_number):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    current_frame_label = start_frame_number
    while cap.isOpened() and frame_count < MAX_FRAME_COUNT:
        success, frame = cap.read()
        if success:
            frame_count += 1
            current_frame_label += 1

            # 1
            if VIDEO_TYPE == "rage":
                frame_name = os.path.join(output_folder, f"non_rage_{current_frame_label:04d}.jpg")
            else if VIDEO_TYPE == "non_rage":
                frame_name = os.path.join(output_folder, f"rage_{current_frame_label:04d}.jpg")
            cv2.imwrite(frame_name, frame)
        else:
            break

    cap.release()
    print(f"Saved {frame_count} frames from {os.path.basename(video_path)} starting at {start_frame_number+1}.")

if __name__ == "__main__":
    
    # 2
    # please modify the paths based on the location of the vidoes saved on your machine
    if VIDEO_TYPE == "rage": 
        video_files = [f"./MMZ_data/rage/rage_{i}.mov" for i in range(1, 21)]
    else if VIDEO_TYPE == "non_rage": 
        video_files = [f"./MMZ_data/non_rage/non_rage_{i}.mov" for i in range(1, 21)]

    # 3
    if VIDEO_TYPE == "rage": 
        output_folder = "NON_RAGE"
    else if VIDEO_TYPE == "non_rage": 
        output_folder = "RAGE"
    
    # 4
    if EDITOR == "Arthur": 
        current_start_label = 0    # Arthur 
    else if EDITOR == "Daniel": 
        current_start_label = 1800 # Daniel
    else if EDITOR == "Mehrdad": 
        current_start_label = 3600 # Mehrdad
    

    for video in video_files:
        # Pass the current start label to videoToFrames
        video_name = os.path.basename(video)
        print(f"Processing {video_name}, saving frames to folder: {output_folder}")
        videoToFrames(video, output_folder, current_start_label)
        # Advance the label for the next video
        current_start_label += MAX_FRAME_COUNT
