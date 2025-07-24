#!/usr/bin/env python3
"""
Simple manual test for pondering functionality.
"""

import subprocess
import sys
import time

def test_manual():
    """Manual test to verify pondering works."""
    print("Starting manual pondering test...")
    print("="*60)
    
    # Start the engine
    engine = subprocess.Popen(
        ["python3", "uci_engine.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    def send(cmd):
        print(f">>> {cmd}")
        engine.stdin.write(cmd + '\n')
        engine.stdin.flush()
    
    def read_until(expected=None, timeout=5):
        """Read output until expected string or timeout."""
        import select
        start = time.time()
        lines = []
        
        while time.time() - start < timeout:
            ready, _, _ = select.select([engine.stdout], [], [], 0.1)
            if ready:
                line = engine.stdout.readline().strip()
                if line:
                    print(f"<<< {line}")
                    lines.append(line)
                    if expected and expected in line:
                        return lines
        return lines
    
    try:
        # Initialize
        send("uci")
        read_until("uciok")
        
        send("setoption name Ponder value true")
        send("setoption name Verbose value true")
        send("setoption name Threads value 4")
        
        send("isready")
        read_until("readyok")
        
        # Test 1: Normal move with ponder suggestion
        print("\n--- Test 1: Normal move ---")
        send("position startpos")
        send("go movetime 500")
        output = read_until("bestmove")
        
        # Extract moves
        bestmove_line = [l for l in output if l.startswith("bestmove")][0]
        print(f"\nBestmove line: {bestmove_line}")
        assert "ponder" in bestmove_line, "No ponder move suggested!"
        
        # Test 2: Pondering
        print("\n--- Test 2: Start pondering ---")
        # Extract ponder move from bestmove line
        parts = bestmove_line.split()
        if "ponder" in parts:
            ponder_idx = parts.index("ponder")
            ponder_move = parts[ponder_idx + 1]
            print(f"Pondering on move: {ponder_move}")
            send(f"position startpos moves e2e4 {ponder_move}")
        else:
            send("position startpos moves e2e4 e7e5")  # fallback
        send("go ponder")
        
        # Let it ponder for a bit
        print("\nPondering for 2 seconds...")
        time.sleep(2)
        
        # Stop pondering
        send("stop")
        output = read_until("bestmove", timeout=2)
        
        bestmove_line = [l for l in output if l.startswith("bestmove")]
        assert bestmove_line, "No bestmove after stopping ponder!"
        
        print("\n--- Tests completed successfully! ---")
        
    finally:
        send("quit")
        time.sleep(0.5)
        engine.terminate()
        

if __name__ == "__main__":
    test_manual()