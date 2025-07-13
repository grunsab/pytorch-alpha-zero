#!/usr/bin/env python3
"""
Script to download and reformat the CCRL dataset for training.
"""

import os
import subprocess
import sys
import urllib.request
import gzip
import shutil
from pathlib import Path

def download_file(url, filename):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownload of {filename} completed!")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def extract_gzip(gz_file, extract_to):
    """Extract a gzip file."""
    print(f"Extracting {gz_file}...")
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extraction completed: {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {gz_file}: {e}")
        return False

def extract_7z(archive_file, extract_to_dir):
    """Extract a 7z file using py7zr or system 7z command."""
    print(f"Extracting {archive_file}...")
    
    # Try using py7zr first
    try:
        import py7zr
        extract_dir = Path(extract_to_dir).parent
        with py7zr.SevenZipFile(archive_file, mode='r') as archive:
            archive.extractall(extract_dir)
        print(f"Extraction completed using py7zr")
        return True
    except ImportError:
        print("py7zr not available, trying system 7z command...")
    except Exception as e:
        print(f"Error with py7zr: {e}, trying system 7z command...")
    
    # Try using system 7z command
    try:
        extract_dir = Path(extract_to_dir).parent
        result = subprocess.run(['7z', 'x', str(archive_file), f'-o{extract_dir}', '-y'], 
                              check=True, capture_output=True, text=True)
        print(f"Extraction completed using system 7z")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error with system 7z: {e}")
    
    # Try using unzip (in case it's a zip file)
    try:
        import zipfile
        extract_dir = Path(extract_to_dir).parent
        with zipfile.ZipFile(archive_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extraction completed using zipfile")
        return True
    except Exception as e:
        print(f"Error with zipfile: {e}")
    
    print("Failed to extract archive. Please install py7zr (pip install py7zr) or 7z command line tool")
    return False

def main():
    # Create directories
    home_dir = Path.home()
    ccrl_dir = home_dir / "ccrl"
    raw_dir = ccrl_dir / "raw"
    reformated_dir = ccrl_dir / "reformated"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(reformated_dir, exist_ok=True)
    
    print(f"CCRL dataset will be downloaded to: {ccrl_dir}")
    
    # CCRL dataset URL (without comments, more suitable for training)
    ccrl_url = "https://www.computerchess.org.uk/ccrl/4040/CCRL-4040.[2131733].pgn.7z"
    archive_file = raw_dir / "CCRL-4040.pgn.7z"
    pgn_file = raw_dir / "CCRL-4040.[2131733].pgn"
    
    # Download the dataset if not already present
    if not archive_file.exists():
        print("Downloading CCRL dataset...")
        if not download_file(ccrl_url, str(archive_file)):
            print("Failed to download CCRL dataset")
            return False
    else:
        print(f"CCRL dataset already exists: {archive_file}")
    
    # Extract the dataset if not already extracted
    if not pgn_file.exists():
        print("Extracting CCRL dataset...")
        if not extract_7z(str(archive_file), str(pgn_file)):
            print("Failed to extract CCRL dataset")
            return False
    else:
        print(f"CCRL dataset already extracted: {pgn_file}")
    
    # Run the reformat script
    print("Reformatting CCRL dataset...")
    reformat_script = Path(__file__).parent / "reformat.py"
    
    if not reformat_script.exists():
        print(f"Error: reformat.py not found at {reformat_script}")
        return False
    
    try:
        # Run reformat.py with the input and output paths
        cmd = [sys.executable, str(reformat_script), str(pgn_file), str(reformated_dir)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Reformatting completed successfully!")
        print(f"Reformatted data is in: {reformated_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running reformat.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    if main():
        print("\nCCRL dataset download and formatting completed successfully!")
        print("You can now run training with:")
        print("python train.py")
    else:
        print("\nFailed to download and format CCRL dataset")
        sys.exit(1)