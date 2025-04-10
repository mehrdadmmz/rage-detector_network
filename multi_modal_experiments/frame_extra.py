import os
import re
import cv2
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# === CONFIGURATION ===
input_folders = ["Mehrdad_data","data_croped", "data_daniel"]  # Add more folders here
output_folder = "processed"
os.makedirs(output_folder, exist_ok=True)
NUM_FRAMES = 90
TARGET_SR = 16000

def extract_frames(video_path, output_subfolder, video_name):
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    selected_frames = []
    if len(all_frames) >= NUM_FRAMES:
        step = len(all_frames) // NUM_FRAMES
        indices = [i * step for i in range(NUM_FRAMES)]
        selected_frames = [all_frames[i] for i in indices]
    else:
        while len(selected_frames) < NUM_FRAMES:
            for frame in all_frames:
                selected_frames.append(frame)
                if len(selected_frames) == NUM_FRAMES:
                    break

    for i, frame in enumerate(selected_frames):
        frame_filename = os.path.join(output_subfolder, f"{video_name}_{i+1}.jpg")
        cv2.imwrite(frame_filename, frame)

def extract_audio(video_path, output_subfolder, video_name, target_sr=16000):
    temp_audio_path = os.path.join(output_subfolder, f"{video_name}_temp.wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

    audio = AudioSegment.from_file(temp_audio_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    clipped_audio = audio[:3000]  # first 3 seconds
    final_audio_path = os.path.join(output_subfolder, f"{video_name}_audio.wav")
    clipped_audio.export(final_audio_path, format="wav")
    os.remove(temp_audio_path)

def get_next_index(prefix):
    max_index = 0
    for name in os.listdir(output_folder):
        if name.startswith(prefix):
            match = re.match(rf"{prefix}_(\d+)", name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1

# Initialize counters
rage_index = get_next_index("rage")
non_rage_index = get_next_index("non_rage")

# === MAIN PROCESSING LOOP ===
for folder in input_folders:
    print(f"\n📁 Processing folder: {folder}")
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".mov") and not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(folder, filename)
        base_name = filename.lower()

        if "rage" in base_name and "non_rage" not in base_name:
            video_name = f"rage_{rage_index}"
            rage_index += 1
        elif "non_rage" in base_name:
            video_name = f"non_rage_{non_rage_index}"
            non_rage_index += 1
        else:
            print(f"⚠️ Skipping unclassified file: {filename}")
            continue

        output_subfolder = os.path.join(output_folder, video_name)
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"🎬 Processing {video_name} from {filename}...")
        extract_frames(video_path, output_subfolder, video_name)
        extract_audio(video_path, output_subfolder, video_name, TARGET_SR)

print("\n✅ All videos processed and saved in 'processed/'")
