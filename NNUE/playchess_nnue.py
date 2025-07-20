#!/usr/bin/env python3

import chess
import chess.svg
import torch
import argparse
import time
import cProfile
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNUENetwork import NNUENet
from NNUE_MCTS import NNUE_MCTS
from nnue_encoder import callNeuralNetworkNNUE
from train_nnue import load_nnue_model
from device_utils import get_optimal_device

def print_board(board):
    """Print the chess board to console."""
    print()
    print(board)
    print()

def get_human_move(board):
    """Get a move from human player."""
    while True:
        try:
            move_str = input("Enter your move (e.g., e2e4): ").strip()
            
            # Allow special commands
            if move_str.lower() in ['quit', 'exit']:
                return None
            elif move_str.lower() == 'undo':
                return 'undo'
                
            # Try to parse the move
            try:
                move = chess.Move.from_uci(move_str)
            except:
                # Try SAN notation
                try:
                    move = board.parse_san(move_str)
                except:
                    print("Invalid move format. Use UCI (e2e4) or SAN (Nf3) notation.")
                    continue
                    
            # Check if move is legal
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move!")
                
        except KeyboardInterrupt:
            return None

def evaluate_position(board, nnue_net):
    """Evaluate current position with NNUE."""
    eval_score, _ = callNeuralNetworkNNUE(board, nnue_net, use_incremental=False)
    return eval_score

def play_human_vs_ai(nnue_net, args):
    """Play a game between human and AI."""
    board = chess.Board()
    mcts = NNUE_MCTS(nnue_net, num_threads=args.threads, batch_size=args.batch_size)
    
    # Determine who plays white
    human_white = input("Do you want to play as white? (y/n): ").lower().startswith('y')
    
    print("\nStarting game!")
    print("Commands: 'quit' to exit, 'undo' to take back move")
    
    move_history = []
    
    while not board.is_game_over():
        print_board(board)
        
        # Show evaluation
        if args.verbose:
            eval_score = evaluate_position(board, nnue_net)
            print(f"NNUE Evaluation: {eval_score:+.3f}")
        
        # Determine whose turn
        if (board.turn == chess.WHITE and human_white) or \
           (board.turn == chess.BLACK and not human_white):
            # Human move
            print("Your move:")
            move = get_human_move(board)
            
            if move is None:
                print("Game aborted.")
                return
            elif move == 'undo':
                if len(move_history) >= 2:
                    board.pop()
                    board.pop()
                    move_history = move_history[:-2]
                    mcts.clear_cache()
                    print("Move undone.")
                else:
                    print("No moves to undo.")
                continue
            else:
                board.push(move)
                move_history.append(move)
                mcts.update_root(move)
        else:
            # AI move
            print("AI thinking...")
            start_time = time.time()
            
            # Run MCTS
            ai_move = mcts.search(board, args.rollouts, temperature=0.0)
            
            elapsed = time.time() - start_time
            
            # Show search info
            if args.verbose:
                move_probs = mcts.get_move_probabilities()
                print(f"\nSearch completed in {elapsed:.2f}s")
                print(f"Rollouts: {args.rollouts}")
                print(f"Rollouts/second: {args.rollouts/elapsed:.0f}")
                
                # Show top moves
                sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                print("\nTop moves:")
                for move, prob in sorted_moves:
                    print(f"  {board.san(move)}: {prob:.1%}")
            
            print(f"\nAI plays: {board.san(ai_move)}")
            board.push(ai_move)
            move_history.append(ai_move)
            mcts.update_root(ai_move)
    
    # Game over
    print_board(board)
    print(f"Game over! Result: {board.result()}")
    
    # Print game PGN
    if args.verbose:
        print("\nGame PGN:")
        pgn = chess.pgn.Game()
        node = pgn
        
        board_copy = chess.Board()
        for move in move_history:
            node = node.add_variation(move)
            board_copy.push(move)
            
        pgn.headers["Result"] = board.result()
        print(pgn)

def ai_vs_ai(nnue_net, args):
    """Watch two AI players play against each other."""
    board = chess.Board()
    mcts1 = NNUE_MCTS(nnue_net, num_threads=args.threads, batch_size=args.batch_size)
    mcts2 = NNUE_MCTS(nnue_net, num_threads=args.threads, batch_size=args.batch_size)
    
    move_count = 0
    
    print("AI vs AI game starting...")
    
    while not board.is_game_over() and move_count < 200:  # Limit moves to prevent infinite games
        print_board(board)
        
        # Choose MCTS instance based on turn
        current_mcts = mcts1 if board.turn == chess.WHITE else mcts2
        
        # AI move
        start_time = time.time()
        ai_move = current_mcts.search(board, args.rollouts, temperature=0.1 if move_count < 20 else 0.0)
        elapsed = time.time() - start_time
        
        if args.verbose:
            eval_score = evaluate_position(board, nnue_net)
            print(f"Move {move_count//2 + 1}. Eval: {eval_score:+.3f}, Time: {elapsed:.2f}s")
        
        print(f"{chess.COLOR_NAMES[board.turn].capitalize()} plays: {board.san(ai_move)}")
        
        board.push(ai_move)
        mcts1.update_root(ai_move)
        mcts2.update_root(ai_move)
        
        move_count += 1
        
        # Add small delay for visibility
        if not args.fast:
            time.sleep(0.5)
    
    print_board(board)
    print(f"Game over! Result: {board.result()}")

def profile_performance(nnue_net, args):
    """Profile NNUE performance."""
    print("Profiling NNUE performance...")
    
    # Test position
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    
    print("\n1. Single position evaluation:")
    start = time.time()
    for _ in range(1000):
        eval_score, _ = callNeuralNetworkNNUE(board, nnue_net, use_incremental=False)
    elapsed = time.time() - start
    print(f"   1000 evaluations in {elapsed:.3f}s = {1000/elapsed:.0f} evals/sec")
    
    print("\n2. MCTS performance:")
    mcts = NNUE_MCTS(nnue_net, num_threads=args.threads)
    
    def run_search():
        mcts.search(board, args.rollouts, temperature=0.0)
    
    if args.verbose:
        print("   Running with profiler...")
        profiler = cProfile.Profile()
        profiler.enable()
        
    start = time.time()
    run_search()
    elapsed = time.time() - start
    
    if args.verbose:
        profiler.disable()
        profiler.print_stats(sort='cumulative', lines=20)
    
    print(f"\n   {args.rollouts} rollouts in {elapsed:.3f}s = {args.rollouts/elapsed:.0f} rollouts/sec")
    print(f"   Using {args.threads} threads")
    
    # Test incremental updates
    print("\n3. Incremental update performance:")
    moves = list(board.legal_moves)[:5]
    
    start = time.time()
    for _ in range(100):
        eval_score, accumulator = callNeuralNetworkNNUE(board, nnue_net, use_incremental=True)
        for move in moves:
            board.push(move)
            eval_score, accumulator = callNeuralNetworkNNUE(
                board, nnue_net, use_incremental=True,
                cached_accumulator=accumulator, last_move=move
            )
            board.pop()
    elapsed = time.time() - start
    
    total_evals = 100 * (1 + len(moves))
    print(f"   {total_evals} incremental evaluations in {elapsed:.3f}s = {total_evals/elapsed:.0f} evals/sec")

def main():
    parser = argparse.ArgumentParser(description='Play chess against NNUE AI')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to NNUE model checkpoint')
    parser.add_argument('--mode', type=str, choices=['h', 'a', 'p'], default='h',
                        help='Game mode: h=human vs AI, a=AI vs AI, p=profile')
    parser.add_argument('--rollouts', type=int, default=400,
                        help='Number of MCTS rollouts per move')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads for parallel MCTS')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for NNUE evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information')
    parser.add_argument('--fast', action='store_true',
                        help='No delays in AI vs AI mode')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading NNUE model from {args.model}...")
    try:
        nnue_net = load_nnue_model(args.model)
        device, device_str = get_optimal_device()
        print(f"Model loaded successfully on {device_str}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run selected mode
    if args.mode == 'h':
        play_human_vs_ai(nnue_net, args)
    elif args.mode == 'a':
        ai_vs_ai(nnue_net, args)
    elif args.mode == 'p':
        profile_performance(nnue_net, args)

if __name__ == '__main__':
    main()