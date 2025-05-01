import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
base_dir = "/Users/mananmathur/Documents/Academics/MIT/subject matter/YEAR 4/SEM 8/PROJECT/project/dataset split/OCTDL"
output_dir = "/Users/mananmathur/Documents/Academics/MIT/subject matter/YEAR 4/SEM 8/PROJECT/project/dataset split/OCTDL_Splits"
labels_file = "/Users/mananmathur/Documents/Academics/MIT/subject matter/YEAR 4/SEM 8/PROJECT/project/OCTDL_labels (2).csv"

# Load labels
labels_df = pd.read_csv(labels_file)

# Get unique classes
classes = labels_df["disease"].unique()

# Create 5 sets with no data overlap
total_images = labels_df.shape[0]
indices = np.arange(total_images)
np.random.shuffle(indices)
splits = np.array_split(indices, 5)

for i, split in enumerate(splits, start=1):
    set_dir = os.path.join(output_dir, f"OCTDL_Set_{i}")
    train_dir = os.path.join(set_dir, "train")
    test_dir = os.path.join(set_dir, "test")
    val_dir = os.path.join(set_dir, "validation")
    
    for dir_path in [train_dir, test_dir, val_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(dir_path, cls), exist_ok=True)
    
    # Split images
    split_images = labels_df.iloc[split]
    train, test, val = np.split(split_images.sample(frac=1, random_state=42), [int(.5*len(split_images)), int(.8*len(split_images))])
    
    for dataset, folder in zip([train, test, val], [train_dir, test_dir, val_dir]):
        for _, row in dataset.iterrows():
            src = os.path.join(base_dir, row["file_name"])
            dst = os.path.join(folder, row["disease"], row["file_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)

print("✅ OCTDL dataset successfully split into 5 sets with no data overlap!")
