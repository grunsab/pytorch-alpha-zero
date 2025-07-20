#!/usr/bin/env python3

from flask import Flask, request, jsonify, send_from_directory
import chess
import chess.svg
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNUENetwork import NNUENet
from NNUE_MCTS import NNUE_MCTS
from train_nnue import load_nnue_model
import argparse

app = Flask(__name__)

# Global variables for the model and MCTS
nnue_net = None
mcts = None
args = None

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('../static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('../static', path)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Handle AI move requests."""
    try:
        data = request.get_json()
        fen = data.get('fen', chess.STARTING_FEN)
        
        # Create board from FEN
        board = chess.Board(fen)
        
        if board.is_game_over():
            return jsonify({
                'error': 'Game is over',
                'result': board.result()
            })
        
        # Get AI move using MCTS
        global mcts
        if mcts is None:
            mcts = NNUE_MCTS(nnue_net, num_threads=args.threads)
        
        # Search for best move
        best_move = mcts.search(board, args.rollouts, temperature=0.0)
        
        # Update MCTS root for next search
        mcts.update_root(best_move)
        
        # Apply move
        board.push(best_move)
        
        # Get move in different formats
        response = {
            'move': {
                'from': chess.square_name(best_move.from_square),
                'to': chess.square_name(best_move.to_square),
                'promotion': best_move.promotion if best_move.promotion else None
            },
            'san': chess.Board(fen).san(best_move),
            'fen': board.fen(),
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        }
        
        # Add evaluation if verbose
        if args.verbose:
            from nnue_encoder import callNeuralNetworkNNUE
            eval_score, _ = callNeuralNetworkNNUE(board, nnue_net, use_incremental=False)
            response['evaluation'] = eval_score
            
            # Add top moves
            move_probs = mcts.get_move_probabilities()
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            response['top_moves'] = [
                {
                    'move': chess.Board(fen).san(move),
                    'probability': prob
                }
                for move, prob in sorted_moves
            ]
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate a position."""
    try:
        data = request.get_json()
        fen = data.get('fen', chess.STARTING_FEN)
        
        board = chess.Board(fen)
        
        from nnue_encoder import callNeuralNetworkNNUE
        eval_score, _ = callNeuralNetworkNNUE(board, nnue_net, use_incremental=False)
        
        return jsonify({
            'evaluation': eval_score,
            'fen': fen
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_mcts', methods=['POST'])
def reset_mcts():
    """Reset MCTS tree (useful when starting new game)."""
    global mcts
    if mcts:
        mcts.clear_cache()
        mcts.root = None
    return jsonify({'status': 'ok'})

def main():
    global nnue_net, args
    
    parser = argparse.ArgumentParser(description='NNUE Chess Web Server')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to NNUE model checkpoint')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    parser.add_argument('--rollouts', type=int, default=400,
                        help='Number of MCTS rollouts per move')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads for parallel MCTS')
    parser.add_argument('--verbose', action='store_true',
                        help='Return detailed information')
    parser.add_argument('--debug', action='store_true',
                        help='Run Flask in debug mode')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading NNUE model from {args.model}...")
    try:
        nnue_net = load_nnue_model(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Start server
    print(f"Starting server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()