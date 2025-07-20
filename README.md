
# A chess engine based on the AlphaZero algorithm

This is a pytorch implementation of Google Deep Mind's AlphaZero algorithm for chess. It's a fork of another GitHub project that sets the framework for this, but it adds performance improvements.

## Live

A LiChess Bot is on the roadmap. The bot will allow anyone to play with this chess engine.

## Dependencies

Standard python libraries. See requirements.txt.

## Running the chess engine

The entry point to the chess engine is the python file playchess.py. Good parameters for strong, long-thinking moves would be:
```
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --verbose --rollouts 1000 --threads 10 --mode h
```
The current position is displayed with an ascii chess board. Enter your moves in long algebraic notation. Note that running the engine requires a weights file.  

## Training script

Download the [CCRL Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/), reformat it using `reformat.py`and run `train.py`.

For tips on how to use reformat.py, reference the bottom of the reformat.py, which has all the arguments that reformat offers. Reformat automatically uses the number of threads that your system supports. It should take no more than 60 minutes to download the CCRL Dataset, and reformat all th files using reformat on a modern system.

For train.py, if you have access to an NVIDIA GPU, than you can leave the settings as is. You should have a fully trained chess engine that plays at 2700-2900 ELO in about seven days on a consumer GPU like a 3090 or 4080. If you don't have access to a high end GPU, you might want to decrease the size of the model that's being trained for, by decreasing the number of blocks and filters. You might want to decrease the number of epochs to 50-100 instead of 500 as well, since you'll notice diminishing returns as number of epochs increases.

## About the algorithm

The algorithm is based on [this paper](https://arxiv.org/pdf/1712.01815.pdf). One very important difference between the algorithm used here and the one described in that paper is that this implementation used supervised learning instead of reinforcement learning. Doing reienforcement learning is very computationally intensive. As said in that paper, it took thousands of TPUs to generate the self play games. This program, on the other hand, trains on the [CCRL Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/), which contains 2.5 million top notch chess games. Because each game has around 80 unique positions in it, this yields about 200 million data points for training on. 

## Strength

It plays at a level between 2700-2900. It can easily win against 2500, and usually draws up to 2900.
