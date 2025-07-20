import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from NNUEDataset import NNUEDataset, collate_nnue_batch
from NNUENetwork import NNUENet
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device

# Training parameters
num_epochs = 100
learning_rate = 0.001
weight_decay = 1e-4
gradient_clip = 1.0

# NNUE architecture parameters
feature_size = 41024  # HalfKP features
hidden1_size = 256
hidden2_size = 32

# Data parameters
ccrl_dir = os.path.abspath('../ccrl/reformated')
max_positions_per_game = 30
skip_early_moves = 10

def train_nnue():
    # Get optimal device and configure for training
    device, device_str = get_optimal_device()
    print(f'Using device: {device_str}')
    
    # Optimize batch size and num_workers for the device
    base_batch_size = 1024  # NNUE typically uses larger batches
    batch_size = get_batch_size_for_device(base_batch_size)
    num_workers = get_num_workers_for_device()
    print(f'Batch size: {batch_size}, Workers: {num_workers}')
    
    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = NNUEDataset(ccrl_dir, max_positions_per_game, skip_early_moves)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_nnue_batch,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create and optimize model for the device
    print("Creating NNUE network...")
    nnue_net = NNUENet(feature_size, hidden1_size, hidden2_size)
    nnue_net = optimize_for_device(nnue_net, device)
    
    # Prepare for SIMD operations
    nnue_net.prepare_for_simd()
    
    print(f"Network parameters: {nnue_net.get_num_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nnue_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print('Starting training...')
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        nnue_net.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            indices = batch['indices'].to(device)
            values = batch['values'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (batch mode)
            predictions = nnue_net(indices, batch_mode=True)
            
            # Calculate loss
            loss = criterion(predictions, values)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(nnue_net.parameters(), gradient_clip)
            
            # Update weights
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch:03d} | Batch {batch_idx:05d}/{len(train_loader):05d} | '
                      f'Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f}')
        
        # Epoch statistics
        epoch_loss = total_loss / num_batches
        print(f'Epoch {epoch:03d} completed | Average Loss: {epoch_loss:.6f}')
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_name = f'nnue_best_{hidden1_size}x{hidden2_size}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': nnue_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'architecture': {
                    'feature_size': feature_size,
                    'hidden1_size': hidden1_size,
                    'hidden2_size': hidden2_size
                }
            }, checkpoint_name)
            print(f'Saved best model to {checkpoint_name}')
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_name = f'nnue_epoch_{epoch+1}_{hidden1_size}x{hidden2_size}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': nnue_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'architecture': {
                    'feature_size': feature_size,
                    'hidden1_size': hidden1_size,
                    'hidden2_size': hidden2_size
                }
            }, checkpoint_name)
            print(f'Saved checkpoint to {checkpoint_name}')
    
    # Final model
    final_name = f'nnue_final_{hidden1_size}x{hidden2_size}.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': nnue_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'architecture': {
            'feature_size': feature_size,
            'hidden1_size': hidden1_size,
            'hidden2_size': hidden2_size
        }
    }, final_name)
    print(f'Training completed! Final model saved to {final_name}')
    
    # Prepare for inference
    print("\nPreparing model for inference...")
    nnue_net.eval()
    
    # Optional: Quantize for faster CPU inference
    if device.type == 'cpu':
        print("Quantizing weights for CPU inference...")
        nnue_net.quantize_weights(bits=8)
        quantized_name = f'nnue_quantized_{hidden1_size}x{hidden2_size}.pt'
        torch.save({
            'model_state_dict': nnue_net.state_dict(),
            'quantization': nnue_net.quant_params if hasattr(nnue_net, 'quant_params') else None,
            'architecture': {
                'feature_size': feature_size,
                'hidden1_size': hidden1_size,
                'hidden2_size': hidden2_size
            }
        }, quantized_name)
        print(f'Quantized model saved to {quantized_name}')

def load_nnue_model(checkpoint_path: str, device=None):
    """
    Load a trained NNUE model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (None for auto-detect)
        
    Returns:
        Loaded NNUE model
    """
    if device is None:
        device, _ = get_optimal_device()
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract architecture
    arch = checkpoint['architecture']
    
    # Create model
    model = NNUENet(
        feature_size=arch['feature_size'],
        hidden1_size=arch['hidden1_size'],
        hidden2_size=arch['hidden2_size']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimize for device
    model = optimize_for_device(model, device)
    model.prepare_for_simd()
    model.eval()
    
    return model

if __name__ == '__main__':
    train_nnue()