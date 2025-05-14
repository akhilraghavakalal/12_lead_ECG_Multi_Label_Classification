import os
import shutil
from pathlib import Path

def copy_and_flatten(main_source_dir, main_dest_dir):
    # Get all subdirectories in the main source directory
    source_folders = [f.path for f in os.scandir(main_source_dir) if f.is_dir()]

    for source_folder in source_folders:
        # Create corresponding destination folder
        folder_name = os.path.basename(source_folder)
        dest_folder = os.path.join(main_dest_dir, folder_name)
        Path(dest_folder).mkdir(parents=True, exist_ok=True)

        # Walk through all subdirectories in the source folder
        for root, _, files in os.walk(source_folder):
            for file in files:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)

                # If file already exists in destination, rename it
                if os.path.exists(dest_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_name = f"{base}_{counter}{extension}"
                        dest_path = os.path.join(dest_folder, new_name)
                        counter += 1

                # Copy the file
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {source_path} -> {dest_path}")

    print("All files have been copied successfully.")

if __name__ == "__main__":
    main_source_directory = r"D:\Thesis_Project\12_lead_ECG_Classification\Data"
    main_destination_directory = r"D:\Thesis_Project\12_lead_ECG_Classification\Code\data\original_data"
    
    copy_and_flatten(main_source_directory, main_destination_directory)