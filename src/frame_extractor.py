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
            frame_name = os.path.join(
                output_folder, f"frame_{current_frame_label:04d}.jpg"
            )
            cv2.imwrite(frame_name, frame)
        else:
            break

    cap.release()
    print(f"Saved {frame_count} frames from {os.path.basename(video_path)} starting at {start_frame_number+1}.")


if __name__ == "__main__":

    # modify the file path accordingly
    video_files = [f"./DUMMY/rage/rage_{i}.mov" for i in range(1, 21)]

    output_folder = "DUMMY/frames"
    current_start_label = 0

    for video in video_files:
        video_name = os.path.basename(video)
        print(
            f"Processing {video_name}, saving frames to folder: {output_folder}")
        videoToFrames(video, output_folder, current_start_label)
        current_start_label += MAX_FRAME_COUNT
