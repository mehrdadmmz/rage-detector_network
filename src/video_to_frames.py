import cv2
import os 

def videoToFrames(video_path, output_folder): 
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("Error opening video file")
        exit()
        
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read() 
        if success: 
            frame_name = os.path.join(output_folder, f"frame_{frame_count + 1:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            frame_count += 1
        else: 
            break
            
    cap.release()
    print(f"Total frames saved: {frame_count}")

if __name__ == "__main__": 
  # path to the vides
  video_files = ["./video1.mp4", 
                 "./video2.mp4", 
                 ...,
                ]

for idx, video in enumerate(video_files, start=1):
    video_name = os.path.basename(video)
    output_folder = f"data{idx}"
    print(f"Processing {video_name}, saving frames to folder: {output_folder}")
    videoToFrames(video, output_folder)
