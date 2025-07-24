# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Play chess against the AI (strong play)
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --verbose --rollouts 1000 --threads 10 --mode h

# Run web interface
python3 server.py

# Profile performance
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --mode p --rollouts 100 --threads 1
```

### Training
```bash
# Preprocess CCRL dataset (required first)
python reformat.py input.pgn ccrl/reformated

# Train the model
python train.py
```

### Testing
No formal test suite exists. To verify functionality:
- Run `python3 playchess.py` with a model to test chess playing
- Check training with a small dataset subset
- Verify server starts with `python3 server.py`

## Architecture

### Core Neural Network Design
The AlphaZero network (AlphaZeroNetwork.py) follows the original paper's architecture:
1. **Input encoding**: 16 planes (12 for pieces + 4 for castling) via encoder.py
2. **Residual tower**: ConvBlock → N residual blocks (configurable depth)
3. **Dual heads**: Value head (win probability) + Policy head (move probabilities)
4. **Current models**: 10×128 and 20×256 (blocks × filters)

### MCTS Implementation
MCTS.py implements parallel Monte Carlo Tree Search:
- UCT formula with C=1.5 exploration constant
- Virtual loss for thread-safe parallel rollouts
- Tree structure: Node class contains edges, Edge class tracks statistics
- Supports configurable rollouts and thread count

### Training Pipeline
Unlike original AlphaZero, this uses supervised learning:
- CCRLDataset.py loads individual PGN files from reformatted CCRL data
- train.py uses combined loss: MSE (value) + CrossEntropy (policy)
- Automatic device selection and memory-aware batch sizing via device_utils.py

### Device Optimization
device_utils.py provides cross-platform GPU/CPU support:
- Automatically detects CUDA (including B200), Apple MPS, or CPU
- Configures device-specific optimizations (TF32, mixed precision)
- Handles proper tensor/model migration between devices

### Web Interface
server.py serves a Flask API with React frontend:
- POST /ai_move endpoint for move requests
- Static files served from /static
- Frontend communicates via fetch API with board state

## Key Implementation Details

1. **Position Encoding**: All positions are mirrored when it's black's turn (encoder.py)
2. **Move Representation**: 72 planes for moves (8×8 source × 72 move types)
3. **Legal Move Masking**: Policy outputs are masked to legal moves only
4. **Temperature Sampling**: Move selection uses temperature-based sampling from MCTS visits

## Development Notes

- The codebase prioritizes clarity over performance in some areas
- GPU memory usage scales with batch size and model size
- The supervised learning approach trades perfect play for computational efficiency
- Models are saved as .pt files in the weights/ directory