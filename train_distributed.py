import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet
from device_utils import optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse

# Training params
num_epochs = 500
num_blocks = 20
num_filters = 256
ccrl_dir = os.path.expanduser('~/ccrl/reformated')
logmode = True

def setup(rank, world_size, backend='nccl', init_method='env://'):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPUs, 'gloo' for CPUs)
        init_method: Method to initialize the process group
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args):
    """
    Distributed training function that runs on each process.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup distributed training
    setup(rank, world_size, backend=args.backend)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        device_str = f"CUDA GPU {rank}: {torch.cuda.get_device_name(rank)}"
    else:
        device = torch.device('cpu')
        device_str = f"CPU (Process {rank})"
    
    if rank == 0:
        print(f'Process {rank} using device: {device_str}')
        total_memory = torch.cuda.get_device_properties(rank).total_memory / 1024**3 if torch.cuda.is_available() else 0
        if total_memory > 0:
            print(f"GPU Memory: {total_memory:.1f}GB")
    
    # Adjust batch size for distributed training
    batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
    # Scale down batch size per GPU when using multiple GPUs
    batch_size = batch_size // world_size
    num_workers = args.num_workers if args.num_workers else get_num_workers_for_device()
    
    if rank == 0:
        print(f'Batch size per GPU: {batch_size}, Total batch size: {batch_size * world_size}, Workers: {num_workers}')
    
    # Create dataset and distributed sampler
    train_ds = CCRLDataset(ccrl_dir)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, 
                             num_workers=num_workers, pin_memory=True)
    
    # Create model and move to device
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters)
    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    # Wrap model with DDP
    if torch.cuda.is_available():
        alphaZeroNet = DDP(alphaZeroNet, device_ids=[rank], output_device=rank, 
                          find_unused_parameters=False)
    else:
        alphaZeroNet = DDP(alphaZeroNet)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.learning_rate)
    
    # Optional: Use mixed precision training for better performance
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    if rank == 0:
        print('Starting distributed training')
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
        alphaZeroNet.train()
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move data to device
            position = data['position'].to(device)
            valueTarget = data['value'].to(device)
            policyTarget = data['policy'].to(device)
            
            # Forward pass with optional mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget,
                                                         policyTarget=policyTarget)
                    loss = valueLoss + policyLoss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward and backward pass
                valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget,
                                                     policyTarget=policyTarget)
                loss = valueLoss + policyLoss
                loss.backward()
                optimizer.step()
            
            epoch_value_loss += valueLoss.item()
            epoch_policy_loss += policyLoss.item()
            
            # Only rank 0 prints progress
            if rank == 0:
                message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                         epoch, iter_num, len(train_loader), float(valueLoss), float(policyLoss))
                
                if iter_num != 0 and not logmode:
                    print(('\b' * len(message)), end='')
                print(message, end='', flush=True)
                if logmode:
                    print('')
        
        if rank == 0:
            print('')
        
        # Gather losses from all processes
        avg_value_loss = epoch_value_loss / len(train_loader)
        avg_policy_loss = epoch_policy_loss / len(train_loader)
        
        # Reduce losses across all processes
        if world_size > 1:
            avg_value_tensor = torch.tensor([avg_value_loss], device=device)
            avg_policy_tensor = torch.tensor([avg_policy_loss], device=device)
            dist.all_reduce(avg_value_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_policy_tensor, op=dist.ReduceOp.SUM)
            avg_value_loss = avg_value_tensor.item() / world_size
            avg_policy_loss = avg_policy_tensor.item() / world_size
        
        # Save model checkpoint (only rank 0)
        if rank == 0:
            print(f'Epoch {epoch} - Avg Value Loss: {avg_value_loss:.5f}, Avg Policy Loss: {avg_policy_loss:.5f}')
            
            networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}_distributed.pt'
            
            # Save the model state dict (unwrap DDP if necessary)
            if isinstance(alphaZeroNet, DDP):
                torch.save(alphaZeroNet.module.state_dict(), networkFileName)
            else:
                torch.save(alphaZeroNet.state_dict(), networkFileName)
            
            print(f'Saved model to {networkFileName}')
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed training for AlphaZero')
    
    # Distributed training arguments
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total number of processes (GPUs) to use')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of the current process')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend to use')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Total batch size across all GPUs')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--learning-rate', type=float, default=0.00033,
                        help='Learning rate for optimizer')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (requires GPU)')
    
    # Multi-node training arguments
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='Master node address for multi-node training')
    parser.add_argument('--master-port', type=str, default='12355',
                        help='Master node port for multi-node training')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--gpus-per-node', type=int, default=None,
                        help='Number of GPUs per node')
    
    args = parser.parse_args()
    
    # Set environment variables for multi-node training
    if args.master_addr:
        os.environ['MASTER_ADDR'] = args.master_addr
    if args.master_port:
        os.environ['MASTER_PORT'] = args.master_port
    
    # Determine world size and number of GPUs
    if args.gpus_per_node is None:
        args.gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # For multi-node training
    if args.nodes > 1:
        args.world_size = args.nodes * args.gpus_per_node
        # When using torchrun or torch.distributed.launch, these will be set
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = args.gpus_per_node
    
    if args.world_size > 1:
        # Use multiprocessing spawn for single-node multi-GPU
        if args.nodes == 1:
            mp.spawn(train_distributed,
                    args=(args.world_size, args),
                    nprocs=args.world_size,
                    join=True)
        else:
            # For multi-node, assume launched with torchrun
            train_distributed(args.local_rank, args.world_size, args)
    else:
        # Single GPU/CPU training
        train_distributed(0, 1, args)

if __name__ == '__main__':
    main()