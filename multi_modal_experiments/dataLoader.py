import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchaudio

import torch.nn.functional as F

class RageDataset(Dataset):
    def __init__(self, root_dir, transform=None, audio_transform=None, max_audio_len=48000):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.max_audio_len = max_audio_len
        self.video_dirs = sorted(os.listdir(root_dir))
        self.label_map = {
            'non_rage': 0,
            'rage': 1
        }

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_folder = self.video_dirs[idx]
        full_path = os.path.join(self.root_dir, video_folder)

        if video_folder.startswith("non_rage"):
            label = self.label_map["non_rage"]
        elif video_folder.startswith("rage"):
            label = self.label_map["rage"]
        else:
            raise ValueError(f"Unknown label for folder: {video_folder}")

        # Load image frames
        frames = []
        for i in range(1, 91):
            frame_path = os.path.join(full_path, f"{video_folder}_{i}.jpg")
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames_tensor = torch.stack(frames)  # [90, 3, H, W]

        # Load and pad audio
        audio_path = os.path.join(full_path, f"{video_folder}_audio.wav")
        waveform, _ = torchaudio.load(audio_path)  # [1, N]
        if waveform.shape[1] > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        else:
            pad_len = self.max_audio_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))  # pad last dim

        return {
            'frames': frames_tensor,      # [90, 3, H, W]
            'audio': waveform,            # [1, max_audio_len]
            'label': torch.tensor(label)
        }

