import os
import numpy as np
from PIL import Image


os.makedirs("data/hand_masks", exist_ok=True)


for img_name in os.listdir("data/little-VITON-HD/image"):
    if not img_name.endswith(".jpg"):
        continue


    parse_path = f"data/little-VITON-HD/image-parse-v3/{img_name.replace('.jpg', '.png')}"
    parse = np.array(Image.open(parse_path)) 


    # For grayscale masks (Hand pixel values in VITON-HD =14 and 15)
    hand_mask = ((parse == 14) | (parse == 15)) * 255
    Image.fromarray(hand_mask.astype('uint8')).save(f"data/hand_masks/{img_name}")
