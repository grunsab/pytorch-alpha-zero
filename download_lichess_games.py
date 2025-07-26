#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and filter grandmaster-level chess games from Lichess database.
Filters games where both players have ratings >= 2600 (grandmaster level).
"""

import requests
import os
import sys
import gzip
import shutil
from datetime import datetime
import re
import argparse
from tqdm import tqdm
import time
from extract_zst import extract_zst
import chess.pgn 
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count
import time
from functools import partial
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

MIN_RATING = 2600

import uuid
import zstandard as zstd



def count_games_in_pgn_fast(input_file):
    """
    Count games by scanning for Result tags, which is much faster than parsing.
    Each game ends with a Result tag.
    """
    count = 0
    with zstd.open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('[Result '):
                count += 1
    return count

    
def count_games_parallel(args):
    """
    Count games in a single file for parallel processing.
    """
    pgn_file, input_dir = args
    input_path = os.path.join(input_dir, pgn_file)
    game_count = count_games_in_pgn_fast(input_path)
    return (pgn_file, input_path, game_count)


# Fix Windows console encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Lichess database URLs
LICHESS_DB_URL = "https://database.lichess.org/"

def get_available_databases():
    """Fetch list of available Lichess databases."""
    response = requests.get(LICHESS_DB_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch database list: {response.status_code}")
    
    # Parse HTML to find .pgn.zst files
    pattern = r'href="(standard/lichess_db_standard_rated_\d{4}-\d{2}\.pgn\.zst)"'
    matches = re.findall(pattern, response.text)
    
    return sorted(matches, reverse=True)  # Most recent first

def download_file(url_filename_pairs, chunk_size=8192):
    """Download a file with progress bar."""
    url = url_filename_pairs[0]
    filepath = url_filename_pairs[1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for data in response.iter_content(chunk_size):
                pbar.update(len(data))
                f.write(data)

def parallel_download_mp(url_filename_pairs):
    """Downloads multiple files in parallel using multiprocessing.Pool."""
    num_processes = cpu_count() - 1 if cpu_count() > 1 else 1 # Use one less than available CPUs
    num_processes = max(num_processes, len(url_filename_pairs))
    with Pool(processes=num_processes) as pool:
        pool.map(download_file, url_filename_pairs)


def extract_and_verify_rating(pgn_headers):
    """Extract white and black ratings from PGN headers."""
    white_rating = None
    black_rating = None
    
    for line in pgn_headers.split('\n'):
        if line.startswith('[WhiteElo'):
            match = re.search(r'"(\d+)"', line)
            if match:
                white_rating = int(match.group(1))
        elif line.startswith('[BlackElo'):
            match = re.search(r'"(\d+)"', line)
            if match:
                black_rating = int(match.group(1))
    
    return (white_rating >= MIN_RATING and black_rating >= MIN_RATING)

def extract_and_verify_time_controls(pgn_headers, min_seconds=180):
    """Extract white and black ratings from PGN headers."""

    time_control = None

    time_control_match = re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', pgn_headers)

    if time_control_match:
        initial_time, increment = time_control_match.groups()
    else:
        return False

    return int(initial_time) >= 180


def filter_games_by_rating_and_time_control(input_file, output_file, min_rating=2600, min_seconds=180):
    """Filter games where both players have rating >= min_rating."""
    
    games_processed = 0
    games_kept = 0
    
    # Open compressed input file
    with open(input_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = reader.read(16384)  # Read in chunks
            buffer = ""
            current_game = ""
            in_game = False
            
            print(f"Filtering games with both players rated >= {min_rating}...")
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                while True:
                    if not text_stream:
                        # Try to read more data
                        chunk = reader.read(16384)
                        if not chunk:
                            break
                        text_stream = chunk
                    
                    # Process text stream
                    try:
                        decoded = text_stream.decode('utf-8', errors='ignore')
                        buffer += decoded
                        text_stream = b""
                    except:
                        text_stream = text_stream[1:]  # Skip problematic byte
                        continue
                    
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line
                    
                    for line in lines[:-1]:
                        if line.startswith('[Event'):
                            # Start of new game
                            if current_game and in_game:
                                # Process previous game
                                rating_check = extract_and_verify_rating(current_game)
                                time_control_check = extract_and_verify_time_controls(current_game)
                                if rating_check and time_control_check:
                                    out_f.write(current_game + '\n\n')
                                    games_kept += 1
                                games_processed += 1
                                
                                if games_processed % 10000 == 0:
                                    print(f"Processed: {games_processed:,} games, Kept: {games_kept:,} games ({games_kept/games_processed*100:.1f}%)")
                            
                            current_game = line + '\n'
                            in_game = True
                        elif in_game:
                            current_game += line + '\n'
                
                # Process last game
                if current_game and in_game:
                    rating_check = extract_and_verify_rating(current_game)
                    time_control_check = extract_and_verify_time_controls(current_game)
                    if rating_check and time_control_check:
                        out_f.write(current_game + '\n\n')
                        games_kept += 1
                    games_processed += 1
    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {games_kept:,}")
    print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    
    return games_kept, games_processed



# def process_pgn_chunk(args):
#     """
#     Process a chunk of games from a PGN file.
    
#     Args:
#         args: Tuple of (input_file, output_dir, start_idx, end_idx, offset, counter, lock)
#     """
#     input_file, output_dir, start_idx, end_idx, offset, counter, lock = args
    
#     with zstd.open(input_file, 'r') as pgn_fh:
#         # Skip to the starting position
#         for _ in range(start_idx):
#             game = chess.pgn.skip_game(pgn_fh)
#             if not game:
#                 return 0
        
#         # Process games in this chunk
#         games_written = 0
#         for i in range(start_idx, end_idx):
#             game = chess.pgn.read_game(pgn_fh)
#             if not game:
#                 break
            
#             if extract_and_verify_rating(game) and verify_time_controls(game) and verify_game_termination(game):
#                 unique_filename = str(uuid.uuid4())
#                 output_file = os.path.join(output_dir, f'{unique_filename}.pgn')
#                 with open(output_file, 'w') as game_fh:
#                     print(game, file=game_fh, end='\n\n')
#                 games_written += 1
            
#             # Update shared counter
#             if counter is not None and lock is not None:
#                 with lock:
#                     counter.value += 1
#                     if counter.value % 1000 == 0 and not TQDM_AVAILABLE:
#                         print(f'Processed {counter.value} games')
    
#     return games_written


# def process_single_huge_pgn(input_file, output_dir, num_processes=100):
#     """
#     Reformat a single large PGN file into individual game files using multiprocessing.
    
#     Args:
#         input_file: Path to the input PGN file
#         output_dir: Directory to write individual game files
#         num_processes: Number of processes to use (default: CPU count)
#         approx_games: Approximate number of games (to skip counting)
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     if num_processes is None:
#         num_processes = max(mp.cpu_count(), 100)
    
#     print(f"Reformatting {input_file} to {output_dir} using {num_processes} processes")
    
#     # Count games
#     print("Counting games in file...")
#     start_count = time.time()
#     total_games = count_games_in_pgn_fast(input_file)
#     count_time = time.time() - start_count
#     print(f"Found {total_games} games to process (counting took {count_time:.2f}s)")
    
#     if total_games == 0:
#         print("No games found in file")
#         return
        
#     # Setup shared counter for progress tracking
#     manager = Manager()
#     counter = manager.Value('i', 0)
#     lock = manager.Lock()
    
#     # Divide work among processes
#     games_per_process = total_games // num_processes
#     chunks = []
    
#     for i in range(num_processes):
#         start_idx = i * games_per_process
#         if i == num_processes - 1:
#             end_idx = total_games
#         else:
#             end_idx = (i + 1) * games_per_process
        
#         chunks.append((input_file, output_dir, start_idx, end_idx, 0, counter, lock))
    
#     # Process chunks in parallel
#     start_time = time.time()
    
#     if TQDM_AVAILABLE:
#         with Pool(num_processes) as pool:
#             with tqdm(total=total_games, desc="Processing games") as pbar:
#                 results = []
#                 for chunk in chunks:
#                     result = pool.apply_async(process_pgn_chunk, (chunk,))
#                     results.append(result)
                
#                 # Monitor progress
#                 last_count = 0
#                 while any(not r.ready() for r in results):
#                     current_count = counter.value
#                     pbar.update(current_count - last_count)
#                     last_count = current_count
#                     time.sleep(0.1)
                
#                 # Final update
#                 pbar.update(counter.value - last_count)
                
#                 # Get results
#                 results = [r.get() for r in results]
#     else:
#         with Pool(num_processes) as pool:
#             results = pool.map(process_pgn_chunk, chunks)
    
#     total_written = sum(results)
#     elapsed_time = time.time() - start_time
    
#     print(f'Completed processing and reformatting: {total_written} total games in {elapsed_time:.2f} seconds')
#     print(f'Processing speed: {total_written/elapsed_time:.2f} games/second')

#     return total_written, total_games



def main():
    parser = argparse.ArgumentParser(description="Download and filter grandmaster-level games from Lichess")
    parser.add_argument('--months', type=int, default=1, help='Number of months to download (default: 1)')
    parser.add_argument('--min-rating', type=int, default=2600, help='Minimum rating for both players (default: 2600)')
    parser.add_argument('--output-dir', default='games_training_data/unprocessed', help='Output directory (default: games_training_data/reformatted)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only filter existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/', help='Output directory to store LiChess Databases (default: games_training_data)')

    args = parser.parse_args()
    
    # Create output directory downloads 
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    
    if not args.skip_download:
        # Get available databases
        print("Fetching available Lichess databases...")
        databases = get_available_databases()
        
        if not databases:
            print("No databases found!")
            return
        
        # Download requested number of months
        databases_to_download = databases[:args.months]
        
        print(f"\nFound {len(databases)} databases. Will download {len(databases_to_download)}:")
        for db in databases_to_download:
            print(f"  - {db}")
        
        # Download files

        url_paths_and_fnames = []

        for db_path in databases_to_download:
            filename = os.path.basename(db_path)
            filepath = os.path.join(args.output_dir, filename)
            url = LICHESS_DB_URL + db_path
            url_paths_and_fnames.append((url, filepath))

        parallel_download_mp(url_paths_and_fnames)
    
    # Filter downloaded files
    print("\n" + "="*50)
    print("Starting filtering process...")
    print("="*50)
    
    
    total_kept = 0
    total_processed = 0
    
    MIN_RATING = args.min_rating or 2750

    # Process all .pgn.zst files in the output directory

    fName_count = 0
    for filename in sorted(os.listdir(args.output_dir_downloads)):
        if filename.endswith('.pgn.zst'):
            input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
            input_path_for_parsing = input_path_for_extraction
            output_directory = os.path.join(args.output_dir)
            output_path = os.path.join(args.output_dir, f"filtered_{args.min_rating}+_{fName_count}.pgn")
            print(f"\nProcessing {filename}...")
            kept, processed = filter_games_by_rating_and_time_control(input_path_for_parsing, output_path)
            
            total_kept += kept
            total_processed += processed
            fName_count += 1
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Total games processed: {total_processed:,}")
    print(f"Total games kept: {total_kept:,}")
    if total_processed > 0:
        print(f"Overall percentage: {total_kept/total_processed*100:.1f}%")
    print(f"\nFiltered games saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()