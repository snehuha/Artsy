import os
import pickle

BASE_FOLDER = "data/images/dataset_updated/training_set"

image_paths = []

for root, dirs, files in os.walk(BASE_FOLDER):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            full_path = os.path.join(root, file)

            # convert to Streamlit-safe relative path
            relative_path = os.path.relpath(full_path)

            image_paths.append(relative_path)

print("Total images:", len(image_paths))

with open("model/image_paths.pkl", "wb") as f:

    pickle.dump(image_paths, f)

print("image_paths.pkl FIXED")