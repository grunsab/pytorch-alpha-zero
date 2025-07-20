
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device

#Training params
num_epochs = 500
num_blocks = 20
num_filters = 256
ccrl_dir = os.path.abspath('ccrl/reformated')  # Use expanduser for cross-platform compatibility
logmode=True

def train():
    # Get optimal device and configure for training
    device, device_str = get_optimal_device()
    print(f'Using device: {device_str}')
    
    # Optimize batch size and num_workers for the device
    batch_size = get_batch_size_for_device()
    num_workers = get_num_workers_for_device()
    print(f'Batch size: {batch_size}, Workers: {num_workers}')
    
    train_ds = CCRLDataset( ccrl_dir )
    train_loader = DataLoader( train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers )

    # Create and optimize model for the device
    alphaZeroNet = AlphaZeroNet( num_blocks, num_filters )
    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    optimizer = optim.Adam( alphaZeroNet.parameters() )
    mseLoss = nn.MSELoss()

    print( 'Starting training' )

    for epoch in range( num_epochs ):
        
        alphaZeroNet.train()
        for iter_num, data in enumerate( train_loader ):

            optimizer.zero_grad()

            # Move data to device
            position = data[ 'position' ].to(device)
            valueTarget = data[ 'value' ].to(device)
            policyTarget = data[ 'policy' ].to(device)

            # You can manually examine some the training data here

            valueLoss, policyLoss = alphaZeroNet( position, valueTarget=valueTarget,
                                 policyTarget=policyTarget )

            loss = valueLoss + policyLoss

            loss.backward()

            optimizer.step()

            message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                     epoch, iter_num, len( train_loader ), float( valueLoss ), float( policyLoss ) )
            
            if iter_num != 0 and not logmode:
                print( ('\b' * len(message) ), end='' )
            print( message, end='', flush=True )
            if logmode:
                print('')
        
        print( '' )
        
        networkFileName = 'AlphaZeroNet_{}x{}.pt'.format( num_blocks, num_filters ) 

        torch.save( alphaZeroNet.state_dict(), networkFileName )

        print( 'Saved model to {}'.format( networkFileName ) )

if __name__ == '__main__':

    train()
