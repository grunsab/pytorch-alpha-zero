import os
import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
import numpy as np
from typing import List, Tuple
from nnue_encoder import NNUEEncoder

class NNUEDataset(Dataset):
    """
    Dataset for training NNUE networks from PGN files.
    Optimized for sparse feature representation.
    """
    
    def __init__(self, pgn_dir: str, max_positions_per_game: int = 50,
                 skip_early_moves: int = 10):
        """
        Args:
            pgn_dir: Directory containing PGN files
            max_positions_per_game: Maximum positions to extract per game
            skip_early_moves: Number of opening moves to skip
        """
        self.pgn_dir = pgn_dir
        self.max_positions_per_game = max_positions_per_game
        self.skip_early_moves = skip_early_moves
        self.positions = []
        
        print(f"Loading PGN files from {pgn_dir}")
        self._load_positions()
        print(f"Loaded {len(self.positions)} positions")
        
    def _load_positions(self):
        """Load all positions from PGN files."""
        pgn_files = [f for f in os.listdir(self.pgn_dir) if f.endswith('.pgn')]
        
        for i, pgn_file in enumerate(pgn_files):
            if i % 1000 == 0:
                print(f"Processing file {i}/{len(pgn_files)}")
                
            filepath = os.path.join(self.pgn_dir, pgn_file)
            try:
                with open(filepath, 'r') as f:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        continue
                        
                    # Get game result
                    result = game.headers.get('Result', '*')
                    if result == '1-0':
                        outcome = 1.0
                    elif result == '0-1':
                        outcome = -1.0
                    elif result == '1/2-1/2':
                        outcome = 0.0
                    else:
                        continue  # Skip unfinished games
                        
                    # Extract positions
                    board = game.board()
                    moves = list(game.mainline_moves())
                    
                    # Skip if game is too short
                    if len(moves) < self.skip_early_moves + 5:
                        continue
                        
                    # Sample positions from the game
                    positions_to_sample = min(len(moves) - self.skip_early_moves, 
                                            self.max_positions_per_game)
                    
                    if positions_to_sample > 0:
                        # Sample evenly throughout the game
                        indices = np.linspace(self.skip_early_moves, 
                                            len(moves) - 1, 
                                            positions_to_sample, 
                                            dtype=int)
                        
                        # Play through to each sampled position
                        board = game.board()
                        move_idx = 0
                        
                        for target_idx in indices:
                            # Play moves up to target position
                            while move_idx <= target_idx and move_idx < len(moves):
                                board.push(moves[move_idx])
                                move_idx += 1
                                
                            # Store position data
                            position_data = {
                                'fen': board.fen(),
                                'outcome': outcome,
                                'ply': move_idx
                            }
                            self.positions.append(position_data)
                            
            except Exception as e:
                print(f"Error processing {pgn_file}: {e}")
                continue
                
    def __len__(self):
        return len(self.positions)
        
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - indices: Active feature indices
                - value: Position evaluation (from game outcome)
                - ply: Move number
        """
        position_data = self.positions[idx]
        
        # Create board from FEN
        board = chess.Board(position_data['fen'])
        
        # Get NNUE features from side to move perspective
        if board.turn == chess.WHITE:
            indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=True)
            value = position_data['outcome']
        else:
            indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=False)
            value = -position_data['outcome']  # Flip for black's perspective
            
        # Convert to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        value_tensor = torch.tensor(value, dtype=torch.float32)
        
        # Apply tapering based on game phase (optional)
        ply = position_data['ply']
        taper = min(1.0, ply / 40.0)  # Full weight after move 40
        value_tensor *= taper
        
        return {
            'indices': indices_tensor,
            'value': value_tensor,
            'num_features': len(indices)
        }
        
def collate_nnue_batch(batch: List[dict]) -> dict:
    """
    Custom collate function for NNUE batches.
    Handles variable-length sparse features.
    """
    batch_size = len(batch)
    max_features = max(sample['num_features'] for sample in batch)
    
    # Initialize tensors
    indices = torch.full((batch_size, max_features), -1, dtype=torch.long)
    values = torch.zeros(batch_size, dtype=torch.float32)
    
    # Fill tensors
    for i, sample in enumerate(batch):
        num_features = sample['num_features']
        indices[i, :num_features] = sample['indices']
        values[i] = sample['value']
        
    return {
        'indices': indices,
        'values': values
    }