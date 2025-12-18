import os
import subprocess
import sys

def setup_linemod_from_drive(drive_zip_path, output_dir="data"):
    """
    Copies the LineMOD dataset from Google Drive and extracts it using 7zip.
    
    Args:
        drive_zip_path (str): The full path to the zip file on Google Drive.
        output_dir (str): The local folder where data should be saved.
    """

    # Setup Directories
    # Structure: output_dir/linemod
    target_dir = os.path.join(output_dir, "linemod")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}.")

    # Define the local destination path
    local_zip_filename = 'Linemod_preprocessed.zip'
    local_zip_path = os.path.join(target_dir, local_zip_filename)

    # Check source file
    if not os.path.exists(drive_zip_path):
        print(f"Error: Source file not found at {drive_zip_path}")
        print("Make sure Google Drive is mounted and the path is correct.")
        return

    # Install 7zip if not present
    try:
        subprocess.run(["7z"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Installing 7zip...")
        subprocess.run(["sudo", "apt-get", "install", "-y", "p7zip-full"], check=True)

    # Copy the dataset
    print(f"Copying from Drive: {drive_zip_path}...")
    try:
        # We use subprocess to run the 'cp' command efficiently
        subprocess.run(["cp", drive_zip_path, local_zip_path], check=True)
        print("Copy complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        return

    # Extract using 7zip
    if os.path.exists(local_zip_path):
        print(f"Extracting {local_zip_filename} using 7zip...")
        try:
            # Command: 7z x "source" -o"destination"
            # Note: No space after -o
            subprocess.run(["7z", "x", local_zip_path, f"-o{target_dir}"], check=True)
            
            # Optional: Remove the zip file after extraction to save space
            #os.remove(local_zip_path) 
            #print("Cleanup: Removed local zip file.")
            
            print("Extraction complete.")     
        except subprocess.CalledProcessError:
            print("Error: 7zip extraction failed.")
    else:
        print("Error: Local zip file not found after copy.")


if __name__ == "__main__": 
    import argparse

# ... (keep your setup_linemod_from_drive function above) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and extract dataset from Google Drive to Colab.")
    parser.add_argument("--drive_path", type=str, required=True, help="Full path to the zip file on Google Drive")
    parser.add_argument("--output_dir", type=str, default="data", help="Local destination folder (default: 'data')")

    args = parser.parse_args()

    # Call the function with the parsed arguments
    setup_linemod_from_drive(args.drive_path, args.output_dir)