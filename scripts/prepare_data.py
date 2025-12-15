"""
Script to download and prepare RSNA 2023 Abdominal Trauma Detection dataset.
"""
import os
import argparse
import glob
import zipfile

def download_rsna_data(output_dir):
    """Download RSNA 2023 Abdominal Trauma Detection dataset"""
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed. Please install it with: pip install kaggle")
        return

    print(f"Downloading data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        kaggle.api.competition_download_files(
            'rsna-2023-abdominal-trauma-detection',
            path=output_dir,
            quiet=False
        )
        print("Download complete.")
        
        # Unzip
        zip_files = glob.glob(os.path.join(output_dir, '*.zip'))
        for zip_path in zip_files:
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove zip file to save space
            # os.remove(zip_path) # Optional
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please ensure you have set up your Kaggle API credentials (kaggle.json).")

def verify_data_structure(data_path):
    """Verify data structure"""
    required_files = [
        'train_series_meta.csv',
        'train_images' 
        # Note: 'segmentations' might not be in the root download, often separate dataset
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(data_path, f)):
            missing.append(f)
            
    if missing:
        print(f"Warning: Missing files/directories: {missing}")
        return False
        
    print(f"Data structure at {data_path} looks correct.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare RSNA Dataset")
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    args = parser.parse_args()
    
    download_rsna_data(args.output_dir)
    verify_data_structure(args.output_dir)
