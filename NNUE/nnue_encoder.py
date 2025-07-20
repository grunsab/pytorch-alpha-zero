import chess
import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_optimal_device

# Get the optimal device once at module level
DEVICE, DEVICE_STR = get_optimal_device()

class NNUEEncoder:
    """
    Encoder for NNUE features, specifically HalfKP (Half-King-Piece).
    Optimized for incremental updates and multi-threaded evaluation.
    """
    
    # Constants for HalfKP encoding
    NUM_SQUARES = 64
    NUM_PIECE_TYPES = 10  # Pawn, Knight, Bishop, Rook, Queen (x2 for colors), no kings
    NUM_FEATURES = NUM_SQUARES * (NUM_SQUARES * NUM_PIECE_TYPES + 1)  # 41024 features
    
    # Piece to feature offset mapping (excluding kings)
    PIECE_TO_OFFSET = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }
    
    @staticmethod
    def square_to_index(square: int, flip: bool = False) -> int:
        """Convert chess square to array index, optionally flipping perspective."""
        if flip:
            # Flip rank (vertical flip)
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            return rank * 8 + file
        return square
    
    @staticmethod
    def get_halfkp_indices(board: chess.Board, perspective_white: bool = True) -> List[int]:
        """
        Extract HalfKP feature indices from a position.
        
        Args:
            board: Chess board position
            perspective_white: Whether to encode from white's perspective
            
        Returns:
            List of active feature indices
        """
        active_indices = []
        
        # Determine perspective
        king_color = chess.WHITE if perspective_white else chess.BLACK
        opponent_color = chess.BLACK if perspective_white else chess.WHITE
        flip = not perspective_white
        
        # Get king position
        king_square = board.king(king_color)
        if king_square is None:
            return active_indices
            
        king_idx = NNUEEncoder.square_to_index(king_square, flip)
        
        # Process all pieces except kings
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
                
            # Get piece square index from perspective
            piece_idx = NNUEEncoder.square_to_index(square, flip)
            
            # Calculate feature index
            piece_offset = NNUEEncoder.PIECE_TO_OFFSET[piece.piece_type]
            if piece.color == opponent_color:
                piece_offset += 5  # Opponent pieces offset
                
            # HalfKP index formula
            feature_idx = king_idx * (NUM_SQUARES * 10 + 1) + piece_idx * 10 + piece_offset
            active_indices.append(feature_idx)
            
        return active_indices
    
    @staticmethod
    def encode_for_nnue(board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode position for NNUE evaluation.
        
        Args:
            board: Chess board position
            
        Returns:
            Tuple of (white_features, black_features) as sparse indices
        """
        # Get features from both perspectives
        white_indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=True)
        black_indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=False)
        
        # Convert to tensors
        white_features = torch.tensor(white_indices, dtype=torch.long, device=DEVICE)
        black_features = torch.tensor(black_indices, dtype=torch.long, device=DEVICE)
        
        return white_features, black_features
    
    @staticmethod
    def encode_batch_for_nnue(boards: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple positions for batch NNUE evaluation.
        
        Args:
            boards: List of chess board positions
            
        Returns:
            Tuple of (white_features, black_features) as padded batch tensors
        """
        batch_size = len(boards)
        max_features = 32  # Maximum active features per position (conservative estimate)
        
        # Initialize batch tensors with -1 (padding value)
        white_batch = torch.full((batch_size, max_features), -1, dtype=torch.long, device=DEVICE)
        black_batch = torch.full((batch_size, max_features), -1, dtype=torch.long, device=DEVICE)
        
        for i, board in enumerate(boards):
            white_indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=True)
            black_indices = NNUEEncoder.get_halfkp_indices(board, perspective_white=False)
            
            # Fill in active features
            if white_indices:
                white_batch[i, :len(white_indices)] = torch.tensor(white_indices, dtype=torch.long)
            if black_indices:
                black_batch[i, :len(black_indices)] = torch.tensor(black_indices, dtype=torch.long)
                
        return white_batch, black_batch
    
    @staticmethod
    def get_feature_updates(old_board: chess.Board, new_board: chess.Board, 
                           move: chess.Move, perspective_white: bool = True) -> Tuple[List[int], List[int]]:
        """
        Get feature updates for incremental NNUE evaluation.
        
        Args:
            old_board: Board before the move
            new_board: Board after the move
            move: The move that was made
            perspective_white: Perspective for feature extraction
            
        Returns:
            Tuple of (added_features, removed_features)
        """
        # Get features before and after
        old_features = set(NNUEEncoder.get_halfkp_indices(old_board, perspective_white))
        new_features = set(NNUEEncoder.get_halfkp_indices(new_board, perspective_white))
        
        # Calculate differences
        added = list(new_features - old_features)
        removed = list(old_features - new_features)
        
        return added, removed
    
    @staticmethod
    def encode_for_training(board: chess.Board, move: chess.Move, winner: int) -> dict:
        """
        Encode a training position for NNUE.
        
        Args:
            board: Chess board position
            move: Move to make from this position
            winner: Game outcome (-1 for black, 0 for draw, 1 for white)
            
        Returns:
            Dictionary with training data
        """
        # Get features from the side to move's perspective
        if board.turn == chess.WHITE:
            features = NNUEEncoder.get_halfkp_indices(board, perspective_white=True)
            value = float(winner)
        else:
            features = NNUEEncoder.get_halfkp_indices(board, perspective_white=False)
            value = float(-winner)  # Flip value for black's perspective
            
        # Convert to sparse representation
        indices = torch.tensor(features, dtype=torch.long)
        
        # For NNUE, we typically don't need move probabilities, just the evaluation
        # But we'll include the move for compatibility
        return {
            'indices': indices,
            'value': value,
            'move': move,
            'fen': board.fen()
        }
    
    @staticmethod
    def create_move_mapping() -> dict:
        """
        Create a mapping from moves to indices for policy head (if needed).
        NNUE typically doesn't use policy output, but included for compatibility.
        """
        move_to_idx = {}
        idx = 0
        
        # All possible from-to square combinations
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    move_to_idx[(from_sq, to_sq)] = idx
                    idx += 1
                    
        # Promotions
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        move_to_idx[(from_sq, to_sq, promo)] = idx
                        idx += 1
                        
        return move_to_idx

def callNeuralNetworkNNUE(board: chess.Board, nnue_net, use_incremental: bool = False,
                         cached_accumulator: Optional[torch.Tensor] = None,
                         last_move: Optional[chess.Move] = None) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Call NNUE network on a position.
    
    Args:
        board: Chess board position
        nnue_net: NNUE network model
        use_incremental: Whether to use incremental updates
        cached_accumulator: Cached accumulator from previous position
        last_move: Last move made (for incremental updates)
        
    Returns:
        Tuple of (evaluation, new_accumulator)
    """
    # Get current player perspective
    perspective_white = board.turn == chess.WHITE
    
    if use_incremental and cached_accumulator is not None and last_move is not None:
        # Get feature updates
        old_board = board.copy()
        old_board.pop()  # Undo last move
        added, removed = NNUEEncoder.get_feature_updates(old_board, board, last_move, perspective_white)
        
        # Incremental forward
        added_tensor = torch.tensor(added, dtype=torch.long, device=DEVICE)
        removed_tensor = torch.tensor(removed, dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            value, new_accumulator = nnue_net.incremental_forward(added_tensor, removed_tensor, cached_accumulator)
            
    else:
        # Full forward pass
        indices = NNUEEncoder.get_halfkp_indices(board, perspective_white)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            value = nnue_net(indices_tensor, batch_mode=False)
            
            if use_incremental:
                new_accumulator = nnue_net.compute_initial_accumulator(indices_tensor)
            else:
                new_accumulator = None
                
    # Convert to float and flip if black's turn
    eval_score = float(value.item())
    if not perspective_white:
        eval_score = -eval_score
        
    return eval_score, new_accumulator

def callNeuralNetworkNNUEBatched(boards: List[chess.Board], nnue_net) -> np.ndarray:
    """
    Evaluate multiple positions with NNUE.
    
    Args:
        boards: List of chess positions
        nnue_net: NNUE network model
        
    Returns:
        Array of evaluations
    """
    batch_size = len(boards)
    
    # Separate positions by side to move
    white_positions = []
    black_positions = []
    white_indices = []
    black_indices = []
    
    for i, board in enumerate(boards):
        if board.turn == chess.WHITE:
            white_positions.append(board)
            white_indices.append(i)
        else:
            black_positions.append(board)
            black_indices.append(i)
    
    evaluations = np.zeros(batch_size, dtype=np.float32)
    
    # Evaluate white positions
    if white_positions:
        white_batch, _ = NNUEEncoder.encode_batch_for_nnue(white_positions)
        with torch.no_grad():
            white_evals = nnue_net(white_batch, batch_mode=True).cpu().numpy()
        for i, idx in enumerate(white_indices):
            evaluations[idx] = white_evals[i]
    
    # Evaluate black positions (flip evaluation)
    if black_positions:
        _, black_batch = NNUEEncoder.encode_batch_for_nnue(black_positions)
        with torch.no_grad():
            black_evals = nnue_net(black_batch, batch_mode=True).cpu().numpy()
        for i, idx in enumerate(black_indices):
            evaluations[idx] = -black_evals[i]
            
    return evaluations