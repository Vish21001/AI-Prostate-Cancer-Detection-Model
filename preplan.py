import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data/prostate/"
IMG_SIZE = 128
ANNOTATIONS_FILE = "annotations.csv"  # columns: filename,label

def load_images():
    df = pd.read_csv(ANNOTATIONS_FILE)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # MRI can be grayscale
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(row['label'])
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
    labels = to_categorical(np.array(labels), num_classes=2)
    return images, labels
