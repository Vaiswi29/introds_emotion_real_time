import os
import numpy as np
import cv2
import kagglehub
from utils import get_face_landmarks
from collections import Counter

# Download dataset
path = kagglehub.dataset_download("gauravsharma99/fer13-cleaned-dataset")
print("Dataset path from KaggleHub:", path)
print("Root contents:", os.listdir(path))

output = []

# Map class folders to labels (lowercase to handle casing)
emotion_to_label = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Loop over folders in dataset root
for emotion_label in os.listdir(path):
    folder_path = os.path.join(path, emotion_label)
    if not os.path.isdir(folder_path):
        continue

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Resize and normalize for MediaPipe
        image = cv2.resize(image, (192, 192))
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] != 3:
            print(f"Skipping image with unexpected shape: {image.shape}")
            continue

        face_landmarks = get_face_landmarks(image)
        if face_landmarks and len(face_landmarks) == 1404:
            label_index = emotion_to_label.get(emotion_label.lower())
            if label_index is not None:
                face_landmarks.append(label_index)
                output.append(face_landmarks)

    print(f"✔ Processed emotion folder: {emotion_label}")

# Report and save
label_counts = Counter([sample[-1] for sample in output])
print("\nSamples per emotion label:")
for label, count in sorted(label_counts.items()):
    name = [k for k, v in emotion_to_label.items() if v == label][0]
    print(f"{label} ({name}): {count}")

np.savetxt("data.txt", np.asarray(output))
print(f"Saved {len(output)} samples to data.txt")
