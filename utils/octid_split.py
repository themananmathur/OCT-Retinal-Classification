import os
import shutil

# Define the dataset split directory
dataset_split_path = "/Users/mananmathur/Documents/Academics/MIT/subject matter/YEAR 4/SEM 8/PROJECT/project/dataset split/OCTID_Splits"

# Class names mapping based on filename prefixes
categories = {
    "NORMAL": "Normal",
    "MH": "Macular_Hole",
    "AMRD": "Age_Macular",
    "CSR": "Central_Serous",
    "DR": "Diabetic"
}

# Process each split set
for set_num in range(1, 6):  # OCTID_Set_1 to OCTID_Set_5
    set_path = os.path.join(dataset_split_path, f"OCTID_Set_{set_num}")
    
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(set_path, split)

        if not os.path.exists(split_path):
            continue  # Skip if the folder doesn't exist

        # Ensure class subfolders exist
        for class_name in categories.values():
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)

        # Move images into respective class folders based on filename prefixes
        for filename in os.listdir(split_path):
            file_path = os.path.join(split_path, filename)
            
            if not os.path.isfile(file_path):
                continue  # Skip directories
            
            # Check filename prefix and move the file
            for prefix, class_name in categories.items():
                if filename.startswith(prefix):
                    shutil.move(file_path, os.path.join(split_path, class_name, filename))
                    break  # Stop after moving the file

print("✅ Dataset successfully organized into class folders!")