# please modify codes below comments #1, 2, 3, 4 accordingly

import cv2
import os

MAX_FRAME_COUNT = 90

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
            # please comment out the one you are not working on right now
            frame_name = os.path.join(output_folder, f"non_rage_{current_frame_label:04d}.jpg")
            frame_name = os.path.join(output_folder, f"rage_{current_frame_label:04d}.jpg")
            cv2.imwrite(frame_name, frame)
        else:
            break

    cap.release()
    print(f"Saved {frame_count} frames from {os.path.basename(video_path)} starting at {start_frame_number+1}.")

if __name__ == "__main__":
    
    # 2
    # please comment out the one you are not working on right now
    video_files = [f"./MMZ_data/rage/rage_{i}.mov" for i in range(1, 21)]
    video_files = [f"./MMZ_data/non_rage/non_rage_{i}.mov" for i in range(1, 21)]

    # 3
    # please comment out the one you are not working on right now
    output_folder = "NON_RAGE"
    output_folder = "RAGE"
    # This will track the last frame label used, so we know where to pick up
    
    # 4
    # please comment out the ones which is not your name
    current_start_label = 0    # Arthur 
    current_start_label = 1800 # Daniel
    current_start_label = 3600 # Mehrdad

    for video in video_files:
        video_name = os.path.basename(video)
        print(f"Processing {video_name}, saving frames to folder: {output_folder}")
        videoToFrames(video, output_folder, current_start_label)
        current_start_label += MAX_FRAME_COUNT
