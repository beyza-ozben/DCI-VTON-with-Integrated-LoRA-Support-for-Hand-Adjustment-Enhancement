import os
import json
import numpy as np

def is_hand_visible(keypoints_2d, threshold=0.3):
    if not keypoints_2d or len(keypoints_2d) % 3 != 0:
        return False
    scores = [keypoints_2d[i+2] for i in range(0, len(keypoints_2d), 3)]
    scores = [s for s in scores if s > 0]
    return np.mean(scores) > threshold if scores else False

def find_visible_hand_samples(json_folder, threshold=0.3):
    visible_samples = []
    
    for file_name in os.listdir(json_folder):
        if file_name.endswith("_keypoints.json"):
            path = os.path.join(json_folder, file_name)
            with open(path, "r") as f:
                data = json.load(f)

            if not data["people"]:
                continue

            person = data["people"][0]
            left = is_hand_visible(person.get("hand_left_keypoints_2d", []), threshold)
            right = is_hand_visible(person.get("hand_right_keypoints_2d", []), threshold)

            if left or right:
                visible_samples.append(file_name.replace("_keypoints.json", "")) 

    return visible_samples

if __name__ == "__main__":
    folder_path = "./openpose_json"  # check your path
    visible_ids = find_visible_hand_samples(folder_path, threshold=0.3)
    with open("hand_visible_samples.txt", "w") as f:
        for vid in visible_ids:
            f.write(vid + "\n")
