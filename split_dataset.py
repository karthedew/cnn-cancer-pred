from Dataset import Dataset
# Core Libraries
import os
import shutil
import random
import numpy as np
import pandas as pd
import polars as pl

# Progress and Visualization Libraries
from tqdm import tqdm

# Image Processing Libraries
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import shannon_entropy

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Image Similarity Libraries
import imagehash

# Deep Learning Libraries (Use PyTorth instead of TensorFlow or Keras)
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Dimensionality Reduction Libraries
from sklearn.manifold import TSNE

from concurrent.futures import ProcessPoolExecutor, as_completed

def split_dataset():
    train_labels = pd.read_csv('data/train_labels.csv')
    # Calculate hashes for all images
    image_dir = "data/train"
    image_ids = train_labels['id']
    image_hashes = calculate_image_hashes(image_ids, image_dir)

    # Check for duplicates
    hash_to_ids = {}
    for img_id, img_hash in image_hashes.items():
        if img_hash not in hash_to_ids:
            hash_to_ids[img_hash] = []
        hash_to_ids[img_hash].append(img_id)

    # Identify duplicates
    duplicates = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}

    # Check if duplicates exist across labels
    cross_label_duplicates = []
    for img_ids in duplicates.values():
        labels = train_labels[train_labels['id'].isin(img_ids)]['label'].unique()
        if len(labels) > 1:
            cross_label_duplicates.append(img_ids)

    dataset = Dataset(train_labels, image_dir="data/train", labels_to_remove=cross_label_duplicates)

    base_train_dir = "data/train"
    smoke_test_train = "data/smoke_train"
    fast_tune_train = "data/fast_tune_train"
    final_tune_train = "data/final_tune_train"

    smoke_tune_df = dataset.get_reduced_df(size=0.01)
    fast_tune_df  = dataset.get_reduced_df(size=0.05)
    final_tune_df = dataset.get_reduced_df(size=0.1)

    os.makedirs(smoke_test_train, exist_ok=True)
    os.makedirs(fast_tune_train, exist_ok=True)
    os.makedirs(final_tune_train, exist_ok=True)

    # Function to copy images to a new directory
    def copy_images(image_ids, source_dir, target_dir):
        for img_id in tqdm(image_ids, desc=f"Copying images to {target_dir}"):
            src_path = os.path.join(source_dir, f"{img_id}.tif")
            dst_path = os.path.join(target_dir, f"{img_id}.tif")
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    # Copy images for smoke test
    smoke_test_ids = smoke_tune_df['id'].tolist()
    copy_images(smoke_test_ids, base_train_dir, smoke_test_train)
    # Copy images for fast tuning
    fast_tune_ids = fast_tune_df['id'].tolist()
    copy_images(fast_tune_ids, base_train_dir, fast_tune_train)
    # Copy images for final tuning
    final_tune_ids = final_tune_df['id'].tolist()
    copy_images(final_tune_ids, base_train_dir, final_tune_train)
    # Verify the copied images
    print(f"Smoke test images copied to {smoke_test_train}: {len(os.listdir(smoke_test_train))} images")
    print(f"Fast tune images copied to {fast_tune_train}: {len(os.listdir(fast_tune_train))} images")
    print(f"Final tune images copied to {final_tune_train}: {len(os.listdir(final_tune_train))} images")
    # Save the sampled DataFrames to CSV files
    smoke_tune_df.to_csv(os.path.join(smoke_test_train, "sampled_smoke_tune.csv"), index=False)
    fast_tune_df.to_csv(os.path.join(fast_tune_train, "sampled_fast_tune.csv"), index=False)
    final_tune_df.to_csv(os.path.join(final_tune_train, "sampled_final_tune.csv"), index=False)


# Function to calculate image hashes
def calculate_image_hashes(image_ids, image_dir):
    hashes = {}
    for img_id in tqdm(image_ids, desc="Calculating image hashes"):
        img_path = f"{image_dir}/{img_id}.tif"
        img = Image.open(img_path)
        img_hash = imagehash.average_hash(img)
        hashes[img_id] = img_hash
    return hashes

if __name__ == '__main__':
    split_dataset()
