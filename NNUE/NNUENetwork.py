import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F

class ClippedReLU(nn.Module):
    """
    Clipped ReLU activation function, clamps values between 0 and 1.
    Common in NNUE architectures for chess engines.
    """
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)

class NNUENet(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network) for chess evaluation.
    Uses HalfKP feature set and is optimized for CPU inference with SIMD operations.
    """
    
    def __init__(self, feature_size=41024, hidden1_size=256, hidden2_size=32):
        """
        Args:
            feature_size (int): Size of input features (HalfKP = 41024)
            hidden1_size (int): Size of first hidden layer (typically 256-512)
            hidden2_size (int): Size of second hidden layer (typically 32-64)
        """
        super().__init__()
        
        self.feature_size = feature_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        # Layers
        self.fc1 = nn.Linear(feature_size, hidden1_size)
        self.clipped_relu1 = ClippedReLU()
        
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.clipped_relu2 = ClippedReLU()
        
        self.fc3 = nn.Linear(hidden2_size, 1)
        
        # Initialize weights for better performance
        self._initialize_weights()
        
        # For incremental updates (caching intermediate computations)
        self.register_buffer('feature_accumulator', torch.zeros(hidden1_size))
        self.register_buffer('active_features', torch.zeros(feature_size, dtype=torch.bool))
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, active_indices: torch.Tensor, 
                active_values: Optional[torch.Tensor] = None,
                batch_mode: bool = False) -> torch.Tensor:
        """
        Forward pass optimized for sparse input features.
        
        Args:
            active_indices: Indices of active features. Shape:
                            - Single position: (num_active_features,)
                            - Batch: (batch_size, max_active_features)
            active_values: Optional values for features (default: 1.0)
            batch_mode: Whether processing a batch of positions
            
        Returns:
            Evaluation score(s)
        """
        if batch_mode:
            return self._forward_batch(active_indices, active_values)
        else:
            return self._forward_single(active_indices, active_values)
    
    def _forward_single(self, active_indices: torch.Tensor, 
                       active_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for a single position (optimized for inference)."""
        # Create sparse input representation
        x = torch.zeros(self.feature_size, device=active_indices.device)
        
        if active_values is not None:
            x[active_indices] = active_values
        else:
            x[active_indices] = 1.0
        
        # Standard forward pass
        x = self.fc1(x)
        x = self.clipped_relu1(x)
        
        x = self.fc2(x)
        x = self.clipped_relu2(x)
        
        x = self.fc3(x)
        return torch.tanh(x)  # Output in [-1, 1] range
    
    def _forward_batch(self, active_indices: torch.Tensor,
                      active_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for a batch of positions."""
        batch_size = active_indices.shape[0]
        device = active_indices.device
        
        # Create dense representation from sparse indices
        x = torch.zeros(batch_size, self.feature_size, device=device)
        
        for i in range(batch_size):
            # Get valid indices for this sample (excluding padding)
            valid_mask = active_indices[i] >= 0
            valid_indices = active_indices[i][valid_mask]
            
            if len(valid_indices) > 0:
                if active_values is not None:
                    x[i, valid_indices] = active_values[i][valid_mask]
                else:
                    x[i, valid_indices] = 1.0
        
        # Standard forward pass
        x = self.fc1(x)
        x = self.clipped_relu1(x)
        
        x = self.fc2(x)
        x = self.clipped_relu2(x)
        
        x = self.fc3(x)
        return torch.tanh(x).squeeze(-1)
    
    def incremental_forward(self, added_features: torch.Tensor,
                           removed_features: torch.Tensor,
                           cached_accumulator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform incremental forward pass by updating only changed features.
        This is the key optimization for NNUE - we only update the parts that changed.
        
        Args:
            added_features: Indices of newly active features
            removed_features: Indices of newly inactive features
            cached_accumulator: Cached first layer accumulator from previous position
            
        Returns:
            Tuple of (evaluation, updated_accumulator)
        """
        # Update accumulator incrementally
        accumulator = cached_accumulator.clone()
        
        # Add contributions from new features
        if len(added_features) > 0:
            accumulator += self.fc1.weight[added_features].sum(dim=0) + self.fc1.bias / len(added_features)
        
        # Remove contributions from old features
        if len(removed_features) > 0:
            accumulator -= self.fc1.weight[removed_features].sum(dim=0) + self.fc1.bias / len(removed_features)
        
        # Continue with rest of network
        x = self.clipped_relu1(accumulator)
        x = self.fc2(x)
        x = self.clipped_relu2(x)
        x = self.fc3(x)
        
        return torch.tanh(x), accumulator
    
    def compute_initial_accumulator(self, active_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the initial accumulator for a position.
        Used to start incremental updates.
        
        Args:
            active_indices: Indices of active features
            
        Returns:
            First layer accumulator
        """
        accumulator = self.fc1.bias.clone()
        if len(active_indices) > 0:
            accumulator += self.fc1.weight[active_indices].sum(dim=0)
        return accumulator
    
    def prepare_for_simd(self):
        """
        Prepare the network for SIMD operations by ensuring memory alignment
        and contiguous tensors.
        """
        # Ensure all weight matrices are contiguous for better SIMD performance
        self.fc1.weight.data = self.fc1.weight.data.contiguous()
        self.fc2.weight.data = self.fc2.weight.data.contiguous()
        self.fc3.weight.data = self.fc3.weight.data.contiguous()
        
        # Ensure proper alignment for SIMD operations
        if self.fc1.weight.data.numel() % 8 != 0:
            # Pad to multiple of 8 for AVX operations
            padding_needed = 8 - (self.fc1.weight.data.numel() % 8)
            self.fc1.weight.data = F.pad(self.fc1.weight.data.view(-1), (0, padding_needed)).view(self.fc1.weight.shape[0], -1)[:, :self.fc1.weight.shape[1]]
    
    def quantize_weights(self, bits=8):
        """
        Quantize weights for faster inference (especially on CPU).
        
        Args:
            bits: Number of bits for quantization (default: 8)
        """
        def quantize_tensor(tensor, bits):
            qmin = -(2**(bits-1))
            qmax = 2**(bits-1) - 1
            scale = (tensor.max() - tensor.min()) / (qmax - qmin)
            zero_point = qmin - tensor.min() / scale
            quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
            return quantized, scale, zero_point
        
        # Store quantization parameters
        self.quantized = True
        self.quant_params = {}
        
        # Quantize each layer's weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                quantized, scale, zero_point = quantize_tensor(param.data, bits)
                self.quant_params[name] = {'quantized': quantized.to(torch.int8),
                                          'scale': scale,
                                          'zero_point': zero_point}
    
    def get_num_parameters(self):
        """Get total number of parameters in the network."""
        return sum(p.numel() for p in self.parameters())

class NNUEFeatureTransformer:
    """
    Transforms chess positions into NNUE features (HalfKP).
    HalfKP: Features based on king position and piece positions.
    """
    
    # Piece type mapping
    PIECE_TYPES = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    @staticmethod
    def get_halfkp_indices(board, perspective_white=True):
        """
        Extract HalfKP feature indices from a chess board.
        
        Args:
            board: chess.Board object
            perspective_white: Whether to get features from white's perspective
            
        Returns:
            List of active feature indices
        """
        import chess
        
        active_indices = []
        
        # Get king square
        king_color = chess.WHITE if perspective_white else chess.BLACK
        king_square = board.king(king_color)
        if king_square is None:
            return active_indices
        
        # Adjust king square based on perspective
        if not perspective_white:
            king_square = chess.square_mirror(king_square)
        
        # Iterate through all pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and square != board.king(king_color):
                # Get piece type and color
                piece_type = NNUEFeatureTransformer.PIECE_TYPES[piece.symbol()]
                
                # Adjust square based on perspective
                feature_square = square if perspective_white else chess.square_mirror(square)
                
                # Calculate HalfKP index
                # Index = king_square * 641 + piece_square * 10 + piece_type
                # 641 = 64 squares * 10 piece types (excluding kings) + 1
                index = king_square * 641 + feature_square * 10 + (piece_type % 6)
                if piece.color != king_color:
                    index += 5  # Offset for opponent pieces
                    
                active_indices.append(index)
        
        return active_indices