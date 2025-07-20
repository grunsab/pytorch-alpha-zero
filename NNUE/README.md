# NNUE Chess Engine Implementation

This folder contains a complete NNUE (Efficiently Updatable Neural Networks) implementation for chess, optimized for CPU performance with SIMD operations and multi-core support.

## Architecture

The NNUE implementation uses the HalfKP feature set:
- **Input**: 41,024 sparse binary features (64 king squares × 641 piece configurations)
- **Hidden Layer 1**: 256 neurons with ClippedReLU activation
- **Hidden Layer 2**: 32 neurons with ClippedReLU activation  
- **Output**: Single evaluation score (tanh activation, range [-1, 1])

Key optimizations:
- Incremental updates for position changes
- SIMD-friendly tensor operations
- Multi-threaded MCTS search
- Batch evaluation support
- Optional weight quantization for faster inference

## Components

### Core Files
- `NNUENetwork.py`: Neural network architecture with SIMD optimizations
- `nnue_encoder.py`: Feature extraction and encoding (HalfKP features)
- `NNUE_MCTS.py`: Monte Carlo Tree Search optimized for NNUE
- `NNUEDataset.py`: Dataset loader for training from PGN files
- `train_nnue.py`: Training script with automatic device optimization
- `playchess_nnue.py`: Interactive chess playing interface
- `server_nnue.py`: Web server for browser-based play

## Usage

### Training

1. Prepare your PGN data (using the same reformatted CCRL data):
```bash
cd ..
python reformat.py input.pgn ccrl/reformated
```

2. Train the NNUE network:
```bash
cd NNEU
python train_nnue.py
```

The training script will:
- Automatically detect and use the best available device (GPU/CPU)
- Save checkpoints every 10 epochs
- Save the best model based on validation loss
- Optionally quantize weights for CPU inference

### Playing Chess

1. Play against the AI:
```bash
python playchess_nnue.py --model nnue_best_256x32.pt --mode h --rollouts 1000 --threads 8
```

2. Watch AI vs AI:
```bash
python playchess_nnue.py --model nnue_best_256x32.pt --mode a --rollouts 400 --threads 4 --verbose
```

3. Profile performance:
```bash
python playchess_nnue.py --model nnue_best_256x32.pt --mode p --rollouts 100 --threads 1 --verbose
```

### Web Interface

Run the web server:
```bash
python server_nnue.py --model nnue_best_256x32.pt --port 5000 --rollouts 400 --threads 4
```

Then open http://localhost:5000 in your browser.

## Command Line Options

### playchess_nnue.py
- `--model`: Path to NNUE model checkpoint (required)
- `--mode`: Game mode - 'h' (human vs AI), 'a' (AI vs AI), 'p' (profile)
- `--rollouts`: Number of MCTS rollouts per move (default: 400)
- `--threads`: Number of threads for parallel search (default: 4)
- `--batch-size`: Batch size for evaluation (default: 8)
- `--verbose`: Show detailed information
- `--fast`: No delays in AI vs AI mode

### train_nnue.py
Configuration is done by editing the script variables:
- `num_epochs`: Training epochs (default: 100)
- `learning_rate`: Initial learning rate (default: 0.001)
- `batch_size`: Base batch size (auto-scaled by device)
- `hidden1_size`: First hidden layer size (default: 256)
- `hidden2_size`: Second hidden layer size (default: 32)

## Performance

The NNUE implementation is optimized for CPU inference:
- Single position evaluation: ~50,000-100,000 positions/second
- Incremental updates: 2-3x faster than full evaluation
- Multi-threaded MCTS: Near-linear scaling with thread count
- SIMD operations: Automatically utilized via PyTorch

## Model Files

Trained models are saved as PyTorch checkpoints:
- `nnue_best_256x32.pt`: Best model during training
- `nnue_epoch_N_256x32.pt`: Checkpoint at epoch N
- `nnue_final_256x32.pt`: Final model after all epochs
- `nnue_quantized_256x32.pt`: Quantized model for faster CPU inference

## Technical Details

### Incremental Updates
The NNUE implementation supports incremental updates of the first layer accumulator when pieces move. This significantly speeds up evaluation in tree search.

### Multi-Core Support
The MCTS implementation uses thread-safe data structures and virtual loss to enable parallel rollouts across multiple CPU cores.

### SIMD Optimization
Weight matrices are aligned and made contiguous for optimal SIMD performance. The implementation automatically uses AVX/AVX2/AVX-512 instructions when available.

### Feature Encoding
Uses HalfKP (Half-King-Piece) features:
- Features depend on king position and piece positions
- Separate feature sets for each king position
- Efficiently handles sparse feature representation