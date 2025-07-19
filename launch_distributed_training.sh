#!/bin/bash

# Launch script for distributed training of AlphaZero
# Supports both single-node multi-GPU and multi-node multi-GPU training

# Default values
GPUS_PER_NODE=8
NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT="12355"
BATCH_SIZE=1024
MIXED_PRECISION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="--mixed-precision"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus NUM           Number of GPUs per node (default: 8)"
            echo "  --nodes NUM          Number of nodes (default: 1)"
            echo "  --node-rank RANK     Rank of this node (default: 0)"
            echo "  --master-addr ADDR   Master node address (default: localhost)"
            echo "  --master-port PORT   Master node port (default: 12355)"
            echo "  --batch-size SIZE    Total batch size across all GPUs (default: 1024)"
            echo "  --mixed-precision    Enable mixed precision training"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Single node with 8 GPUs:"
            echo "  $0 --gpus 8"
            echo ""
            echo "  # Multi-node training (run on each node):"
            echo "  # On master node (node 0):"
            echo "  $0 --gpus 8 --nodes 2 --node-rank 0 --master-addr 192.168.1.1"
            echo ""
            echo "  # On worker node (node 1):"
            echo "  $0 --gpus 8 --nodes 2 --node-rank 1 --master-addr 192.168.1.1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Calculate total world size
WORLD_SIZE=$((NODES * GPUS_PER_NODE))

echo "Starting distributed training..."
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank: $NODE_RANK"
echo "Batch size: $BATCH_SIZE"

# Check if torchrun is available (PyTorch >= 1.10)
if command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training"
    
    # Launch with torchrun
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_distributed.py \
        --batch-size $BATCH_SIZE \
        $MIXED_PRECISION
        
elif command -v python3 -m torch.distributed.launch &> /dev/null; then
    echo "Using torch.distributed.launch for distributed training"
    
    # Launch with torch.distributed.launch (deprecated but still works)
    python3 -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_distributed.py \
        --batch-size $BATCH_SIZE \
        $MIXED_PRECISION
        
else
    echo "Neither torchrun nor torch.distributed.launch found"
    echo "Falling back to direct execution"
    
    # Direct execution for single node
    if [ $NODES -eq 1 ]; then
        python3 train_distributed.py \
            --world-size $GPUS_PER_NODE \
            --batch-size $BATCH_SIZE \
            $MIXED_PRECISION
    else
        echo "Multi-node training requires torchrun or torch.distributed.launch"
        exit 1
    fi
fi