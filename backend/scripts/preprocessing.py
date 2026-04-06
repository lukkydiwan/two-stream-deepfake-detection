import cv2
import numpy as np
import os
from tqdm import tqdm

def apply_srm_filter(img):
    # Standard SRM kernel for noise extraction
    srm_kernel = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ]) / 4.0
    return cv2.filter2D(img, -1, srm_kernel)

def preprocess_dataset(source_base, target_base):
    categories = ['train', 'test']
    labels = ['real', 'fake']

    for cat in categories:
        # Find the actual folder name (handling that timestamp suffix)
        folder_name = [f for f in os.listdir(source_base) if f.startswith(cat)][0]
        source_path = os.path.join(source_base, folder_name)
        
        for label in labels:
            src_dir = os.path.join(source_path, label)
            dst_dir = os.path.join(target_base, cat, label)
            os.makedirs(dst_dir, exist_ok=True)

            print(f"Processing {cat}/{label}...")
            for img_name in tqdm(os.listdir(src_dir)):
                img_path = os.path.join(src_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 1. Resize for EfficientNet
                    img = cv2.resize(img, (224, 224))
                    # 2. Apply SRM Filter
                    srm_img = apply_srm_filter(img)
                    # 3. Save to new directory
                    cv2.imwrite(os.path.join(dst_dir, img_name), srm_img)

# RUN IT: Update 'archive' to the path where your folders are
preprocess_dataset('../archive', '../srm_dataset')