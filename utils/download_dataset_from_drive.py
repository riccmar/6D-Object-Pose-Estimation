import os
import shutil
import zipfile
import argparse
import subprocess

def download_dataset_from_drive(drive_zip_path, output_dir="data"):
    """
    Copies a zip file from a mounted Google Drive to the local Colab environment
    and extracts it.

    Args:
        drive_zip_path (str): The full path to the zip file on Google Drive 
                              (e.g., '/content/drive/MyDrive/Datasets/Linemod_preprocessed.zip').
        output_dir (str): The local folder in Colab where data should be extracted.
    """
        
    # Check if source file exists
    if not os.path.exists(drive_zip_path):
        print(f"Error: Source file not found at {drive_zip_path}")
        print("Please check the path and try again.")
        return

    # Setup Local Directories
    # Structure: output_dir/linemod
    target_dir = os.path.join(output_dir, "linemod")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")

    # Define where to copy the zip locally (renaming it slightly to ensure no conflicts)
    local_zip_name = "Linemod_preprocessed.zip"
    local_zip_path = os.path.join(target_dir, local_zip_name)

    # Copy the file (Drive -> Colab Local)
    print(f"Copying from Drive: {drive_zip_path}...")
    
    # Try using rsync for progress bar (efficient & verbose), fall back to shutil
    if shutil.which("rsync"):
        try:
            subprocess.run(["rsync", "-ahP", drive_zip_path, local_zip_path], check=True)
            print("Copy complete.")
        except subprocess.CalledProcessError:
            print("rsync failed, falling back to shutil...")
            shutil.copy(drive_zip_path, local_zip_path)
    else:
        shutil.copy(drive_zip_path, local_zip_path)
        print("Copy complete.")

    # Extract the zip file
    print(f"Extracting {local_zip_name}...")
    
    # Try using unzip for speed and verbosity
    if shutil.which("unzip"):
        try:
            # -o: overwrite, -d: destination
            subprocess.run(["unzip", "-o", local_zip_path, "-d", target_dir], check=True)
            print("Extraction complete.")
        except subprocess.CalledProcessError:
            print("unzip failed, falling back to zipfile...")
            try:
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print("Extraction complete.")
            except Exception as e:
                print(f"An error occurred during extraction: {e}")
    else:
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("Error: The copied file is not a valid zip file.")
        except Exception as e:
            print(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_zip_path', type=str, required=True,
                        help='Full path to the zip file on Google Drive (e.g., /content/drive/MyDrive/Datasets/Linemod_preprocessed.zip)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Local directory to extract the dataset into (default: data)')
    args = parser.parse_args()
    
    download_dataset_from_drive(args.drive_zip_path, args.output_dir)