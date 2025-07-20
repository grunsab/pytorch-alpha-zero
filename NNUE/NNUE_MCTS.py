import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import chess
import torch
from typing import Optional, List, Tuple
from nnue_encoder import callNeuralNetworkNNUE, callNeuralNetworkNNUEBatched, NNUEEncoder

class NNUEEdge:
    """
    Edge in the MCTS tree, optimized for NNUE evaluation.
    """
    def __init__(self, move: chess.Move, prior: float = 0.0):
        self.move = move
        self.N = 0.0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Average value
        self.P = prior  # Prior probability
        self._lock = Lock()
        
    def update(self, value: float):
        """Thread-safe update of edge statistics."""
        with self._lock:
            self.N += 1.0
            self.W += value
            self.Q = self.W / self.N
            
    def get_ucb(self, parent_N: float, c_puct: float = 1.5) -> float:
        """Calculate UCB value for this edge."""
        with self._lock:
            if self.N == 0:
                return self.P * c_puct * math.sqrt(parent_N)
            return self.Q + self.P * c_puct * math.sqrt(parent_N) / (1.0 + self.N)

class NNUENode:
    """
    Node in the MCTS tree, optimized for NNUE with caching.
    """
    def __init__(self, board: chess.Board, eval_score: float, 
                 accumulator: Optional[torch.Tensor] = None):
        self.board = board.copy()
        self.eval_score = eval_score  # NNUE evaluation
        self.accumulator = accumulator  # Cached NNUE accumulator
        self.N = 1.0
        self.edges = {}  # move -> Edge
        self.children = {}  # move -> Node
        self._lock = Lock()
        self.is_terminal = board.is_game_over()
        
        # Initialize edges for all legal moves
        if not self.is_terminal:
            # Simple prior based on move type
            for move in board.legal_moves:
                prior = self._get_move_prior(move)
                self.edges[move] = NNUEEdge(move, prior)
                
    def _get_move_prior(self, move: chess.Move) -> float:
        """Simple move prior based on move type."""
        # Prioritize captures and checks
        if self.board.is_capture(move):
            return 0.3
        elif self.board.gives_check(move):
            return 0.2
        else:
            return 0.1
            
    def select_best_edge(self) -> Optional[NNUEEdge]:
        """Select edge with highest UCB value."""
        if self.is_terminal:
            return None
            
        with self._lock:
            best_edge = None
            best_ucb = -float('inf')
            
            for edge in self.edges.values():
                ucb = edge.get_ucb(self.N)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_edge = edge
                    
        return best_edge
    
    def update(self, value: float):
        """Update node visit count."""
        with self._lock:
            self.N += 1.0

class NNUE_MCTS:
    """
    MCTS implementation optimized for NNUE evaluation.
    Features:
    - Multi-core parallel search
    - Incremental NNUE updates
    - Batch evaluation
    - Virtual loss for thread safety
    """
    
    def __init__(self, nnue_net, num_threads: int = 1, batch_size: int = 8):
        self.nnue_net = nnue_net
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.root = None
        self._node_cache = {}  # FEN -> Node cache
        self._cache_lock = Lock()
        
    def search(self, board: chess.Board, num_rollouts: int, 
               temperature: float = 1.0) -> chess.Move:
        """
        Perform MCTS search and return best move.
        
        Args:
            board: Current position
            num_rollouts: Number of rollouts to perform
            temperature: Temperature for move selection
            
        Returns:
            Selected move
        """
        # Initialize root if needed
        if self.root is None or self.root.board.fen() != board.fen():
            eval_score, accumulator = callNeuralNetworkNNUE(board, self.nnue_net, 
                                                           use_incremental=True)
            self.root = NNUENode(board, eval_score, accumulator)
            
        # Perform rollouts
        if self.num_threads > 1:
            self._parallel_search(num_rollouts)
        else:
            for _ in range(num_rollouts):
                self._rollout(self.root)
                
        # Select move based on visit counts
        return self._select_move(temperature)
        
    def _parallel_search(self, num_rollouts: int):
        """Perform parallel MCTS rollouts."""
        rollouts_per_thread = num_rollouts // self.num_threads
        remainder = num_rollouts % self.num_threads
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            # Submit rollout tasks
            for i in range(self.num_threads):
                n_rollouts = rollouts_per_thread + (1 if i < remainder else 0)
                if n_rollouts > 0:
                    if self.batch_size > 1 and n_rollouts >= self.batch_size:
                        # Use batch evaluation
                        future = executor.submit(self._batch_rollouts, n_rollouts)
                    else:
                        # Single rollouts
                        future = executor.submit(self._thread_rollouts, n_rollouts)
                    futures.append(future)
                    
            # Wait for completion
            for future in as_completed(futures):
                future.result()
                
    def _thread_rollouts(self, num_rollouts: int):
        """Perform rollouts for a single thread."""
        for _ in range(num_rollouts):
            self._rollout(self.root)
            
    def _batch_rollouts(self, num_rollouts: int):
        """Perform rollouts with batch evaluation."""
        num_batches = num_rollouts // self.batch_size
        remainder = num_rollouts % self.batch_size
        
        for _ in range(num_batches):
            # Collect positions for batch evaluation
            paths = []
            leaves = []
            
            for _ in range(self.batch_size):
                path, leaf = self._select_leaf(self.root)
                paths.append(path)
                leaves.append(leaf)
                
            # Batch evaluate leaves
            if leaves:
                boards = [leaf.board for leaf in leaves]
                values = callNeuralNetworkNNUEBatched(boards, self.nnue_net)
                
                # Backpropagate values
                for i, (path, value) in enumerate(zip(paths, values)):
                    self._backpropagate(path, float(value))
                    
        # Handle remainder
        for _ in range(remainder):
            self._rollout(self.root)
            
    def _rollout(self, node: NNUENode) -> float:
        """Perform a single MCTS rollout."""
        path, leaf = self._select_leaf(node)
        value = self._evaluate_leaf(leaf)
        self._backpropagate(path, value)
        return value
        
    def _select_leaf(self, node: NNUENode) -> Tuple[List[Tuple[NNUENode, NNUEEdge]], NNUENode]:
        """Select a leaf node for expansion."""
        path = []
        current = node
        
        while not current.is_terminal:
            # Select best edge
            edge = current.select_best_edge()
            if edge is None:
                break
                
            # Apply virtual loss
            edge.update(-1.0)
            
            # Check if child exists
            if edge.move in current.children:
                path.append((current, edge))
                current = current.children[edge.move]
            else:
                # Expand node
                new_board = current.board.copy()
                new_board.push(edge.move)
                
                # Check cache
                fen = new_board.fen()
                with self._cache_lock:
                    if fen in self._node_cache:
                        child = self._node_cache[fen]
                    else:
                        # Evaluate with NNUE (incremental if possible)
                        eval_score, accumulator = callNeuralNetworkNNUE(
                            new_board, self.nnue_net,
                            use_incremental=current.accumulator is not None,
                            cached_accumulator=current.accumulator,
                            last_move=edge.move
                        )
                        child = NNUENode(new_board, eval_score, accumulator)
                        self._node_cache[fen] = child
                        
                current.children[edge.move] = child
                path.append((current, edge))
                current = child
                break
                
        return path, current
        
    def _evaluate_leaf(self, leaf: NNUENode) -> float:
        """Evaluate a leaf node."""
        if leaf.is_terminal:
            # Terminal node evaluation
            result = leaf.board.result()
            if result == "1-0":
                return 1.0 if leaf.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                return -1.0 if leaf.board.turn == chess.BLACK else 1.0
            else:
                return 0.0
        else:
            # Use NNUE evaluation (already computed during expansion)
            return leaf.eval_score
            
    def _backpropagate(self, path: List[Tuple[NNUENode, NNUEEdge]], value: float):
        """Backpropagate value through the path."""
        # Remove virtual loss and update with actual value
        for node, edge in reversed(path):
            # Undo virtual loss
            edge.update(1.0)
            # Update with actual value
            edge.update(value)
            node.update(value)
            # Flip value for opponent
            value = -value
            
    def _select_move(self, temperature: float) -> chess.Move:
        """Select move based on visit counts and temperature."""
        if temperature == 0.0:
            # Select most visited move
            best_move = None
            best_visits = -1
            
            for move, edge in self.root.edges.items():
                if edge.N > best_visits:
                    best_visits = edge.N
                    best_move = move
                    
            return best_move
        else:
            # Sample based on visit counts
            moves = []
            probs = []
            
            for move, edge in self.root.edges.items():
                moves.append(move)
                probs.append(edge.N ** (1.0 / temperature))
                
            probs = np.array(probs)
            probs /= probs.sum()
            
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
            
    def get_move_probabilities(self) -> dict:
        """Get visit count distribution over moves."""
        total_visits = sum(edge.N for edge in self.root.edges.values())
        
        probs = {}
        for move, edge in self.root.edges.items():
            probs[move] = edge.N / total_visits if total_visits > 0 else 0.0
            
        return probs
        
    def clear_cache(self):
        """Clear the node cache to free memory."""
        with self._cache_lock:
            self._node_cache.clear()
            
    def update_root(self, move: chess.Move):
        """Update root after a move is played."""
        if self.root and move in self.root.children:
            self.root = self.root.children[move]
        else:
            self.root = None