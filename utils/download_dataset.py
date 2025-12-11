import os
import gdown
import zipfile

def download_linemod_preprocessed_dataset(output_dir="data"):
    """
    Downloads and extracts the LineMOD dataset from a Google Drive URL.
    
    Args:
        output_dir (str): The local folder where data should be saved.
    """

    # Create the directory structure if it doesn't exist
    target_dir = os.path.join(output_dir, "linemod")

    if not target_dir.exists():
        os.makedir(target_dir)
        print(f"Created directory: {target_dir}.")

    # Define the file path and URL
    zip_filename = 'Linemod_preprocessed.zip'
    output_path = os.path.join(target_dir, zip_filename)
    
    url = 'https://drive.google.com/file/d/1mHrrxVMQJFZRjqDt184wP2hxjicpc68C/view'
    #url = 'https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing'

    # Check if the folder already exists to avoid re-downloading (optional check)
    extracted_folder_path = os.path.join(target_dir, 'Linemod_preprocessed')
    if os.path.exists(extracted_folder_path):
        print(f"Dataset already extracted at {extracted_folder_path}.")
        return

    # Download using gdown
    print(f"Downloading {zip_filename} from Google Drive...")
    gdown.download(url, output_path, quiet=False, fuzzy=True)

    # Extract the zip file
    if os.path.exists(output_path):
        print(f"Extracting {zip_filename}...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Optional: Remove the zip file after extraction to save space
            # os.remove(output_path) 
            
            print("Extraction complete.")     
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
    else:
        print("Error: Download failed, zip file not found.")


if __name__ == "__main__": 
    download_linemod_preprocessed_dataset()