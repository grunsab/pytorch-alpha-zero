
import chess.pgn
import os
import sys
import argparse
from pathlib import Path

def reformat_single_pgn(input_file, output_dir):
    """
    Reformat a single large PGN file into individual game files.
    
    Args:
        input_file: Path to the input PGN file
        output_dir: Directory to write individual game files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reformatting {input_file} to {output_dir}")
    
    with open(input_file, 'r') as pgn_fh:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(pgn_fh)
            if not game:
                break
            
            # Write each game to a separate file
            output_file = os.path.join(output_dir, f'{game_idx}.pgn')
            with open(output_file, 'w') as game_fh:
                print(game, file=game_fh, end='\n\n')
            
            game_idx += 1
            if game_idx % 1000 == 0:
                print(f'Wrote {game_idx} games')
    
    print(f'Completed reformatting: {game_idx} total games')

def reformat_directory(input_dir, output_dir):
    """
    Reformat all PGN files in a directory.
    
    Args:
        input_dir: Directory containing PGN files
        output_dir: Directory to write reformatted files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith('.pgn')]
    print(f"Found {len(pgn_files)} PGN files in {input_dir}")
    
    game_idx = 0
    for pgn_file in pgn_files:
        input_path = os.path.join(input_dir, pgn_file)
        print(f"Processing {input_path}")
        
        with open(input_path, 'r') as pgn_fh:
            while True:
                game = chess.pgn.read_game(pgn_fh)
                if not game:
                    break
                
                output_file = os.path.join(output_dir, f'{game_idx}.pgn')
                with open(output_file, 'w') as game_fh:
                    print(game, file=game_fh, end='\n\n')
                
                game_idx += 1
                if game_idx % 1000 == 0:
                    print(f'Wrote {game_idx} games')
    
    print(f'Completed reformatting: {game_idx} total games')

def main():
    parser = argparse.ArgumentParser(description='Reformat PGN files for AlphaZero training')
    parser.add_argument('input', help='Input PGN file or directory')
    parser.add_argument('output', help='Output directory for reformatted files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    if input_path.is_file():
        reformat_single_pgn(str(input_path), str(output_path))
    elif input_path.is_dir():
        reformat_directory(str(input_path), str(output_path))
    else:
        print(f"Error: {input_path} is neither a file nor directory")
        sys.exit(1)

if __name__ == "__main__":
    main()

