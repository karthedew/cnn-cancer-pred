class Dataset:
    """
    A dataset class that:
      - Loads metadata into a Polars DataFrame
      - Computes perceptual hashes to identify duplicates
      - Filters out all duplicate images
      - Exposes the cleaned Polars DataFrame for further processing
    """
    def __init__(self, metadata, image_dir, id_col="id", label_col="label", labels_to_remove=None):
        """
        :param metadata: Path to CSV, pandas DataFrame, or Polars DataFrame
                         Must contain at least `id_col` and `label_col`.
        :param image_dir: Directory where images are stored as "<id>.tif"
        :param id_col:    Name of the column containing the image IDs
        :param label_col: Name of the column containing the labels
        """
        self.image_dir = image_dir
        self.id_col = id_col
        self.label_col = label_col
        self.df = metadata
        self.labels_to_remove = labels_to_remove if labels_to_remove is not None else []

        print(self.df.shape[0])
        self.remove_duplicates()
        print(self.df.shape[0])
        
    def get_reduced_df(self, size=None):
        if not size:
            size = 0.2
        # Ensure the class balance: 50% of samples from class 0, 50% from class 1
        total_samples = int(len(self.df) * size)
        half_samples = total_samples // 2
        
        df_0 = self.df[self.df['label'] == 0]
        df_1 = self.df[self.df['label'] == 1]
        
        num_samples_0 = int(len(df_0) * size)
        num_samples_1 = int(len(df_1) * size)
        
        
        sample_0 = df_0.sample(n=num_samples_0, random_state=42)
        sample_1 = df_1.sample(n=num_samples_1, random_state=42)

        # Combine and shuffle
        balanced_df = pd.concat([sample_0, sample_1]).sample(frac=size, random_state=42).reset_index(drop=True)

        return balanced_df

        
    def remove_duplicates(self):
        bad_ids = {img_id for group in self.labels_to_remove for img_id in group}
        df_clean = self.df.loc[~self.df['id'].isin(bad_ids)].reset_index(drop=True)
        self.df = df_clean

        print(f"Dropped {len(bad_ids)} conflicting‐label images; {len(df_clean)} remain.")

    def __len__(self):
        return self.df.shape[0]

    def ids(self):
        """List of image IDs kept."""
        return self.df[self.id_col].to_list()

    def labels(self):
        """List of labels corresponding to kept IDs."""
        return self.df[self.label_col].to_list()


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
