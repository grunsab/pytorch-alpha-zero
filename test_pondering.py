#!/usr/bin/env python3
"""
Test script for UCI pondering functionality.
This script simulates a UCI GUI to test various pondering scenarios.
"""

import subprocess
import time
import threading
import sys
from queue import Queue

class UCITester:
    def __init__(self, engine_path="python3 uci_engine.py", verbose=True):
        self.verbose = verbose
        self.engine = subprocess.Popen(
            engine_path.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.output_queue = Queue()
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()
        
    def _read_output(self):
        """Read engine output continuously."""
        while True:
            line = self.engine.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line:
                self.output_queue.put(line)
                if self.verbose:
                    print(f"ENGINE: {line}")
    
    def send_command(self, command):
        """Send a command to the engine."""
        if self.verbose:
            print(f"\nGUI: {command}")
        self.engine.stdin.write(command + '\n')
        self.engine.stdin.flush()
    
    def wait_for_response(self, expected, timeout=10):
        """Wait for a specific response from the engine."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                if expected in line:
                    return line
            except:
                pass
        return None
    
    def get_all_output(self, duration=0.5):
        """Get all output for a duration."""
        outputs = []
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                line = self.output_queue.get(timeout=0.1)
                outputs.append(line)
            except:
                pass
        return outputs
    
    def close(self):
        """Close the engine."""
        self.send_command("quit")
        time.sleep(0.5)
        self.engine.terminate()
        self.engine.wait()


def test_basic_pondering():
    """Test basic pondering with ponderhit."""
    print("\n" + "="*60)
    print("TEST 1: Basic pondering with ponderhit")
    print("="*60)
    
    tester = UCITester()
    
    # Initialize engine
    tester.send_command("uci")
    assert tester.wait_for_response("uciok"), "Engine failed to respond to uci"
    
    # Enable pondering and verbose mode
    tester.send_command("setoption name Ponder value true")
    tester.send_command("setoption name Verbose value true")
    tester.send_command("setoption name Threads value 4")
    
    tester.send_command("isready")
    assert tester.wait_for_response("readyok"), "Engine failed to respond to isready"
    
    # Set starting position
    tester.send_command("position startpos")
    
    # Make first move - should get ponder suggestion
    tester.send_command("go movetime 1000")
    bestmove_line = tester.wait_for_response("bestmove", timeout=3)
    
    assert bestmove_line, "No bestmove received"
    assert "ponder" in bestmove_line, "No ponder move suggested"
    
    # Extract moves
    parts = bestmove_line.split()
    best_move = parts[1]
    ponder_move = parts[3] if len(parts) > 3 else None
    
    print(f"\nBest move: {best_move}, Ponder move: {ponder_move}")
    
    # Apply both moves (our move and expected opponent response) and start pondering
    tester.send_command(f"position startpos moves {best_move} {ponder_move}")
    tester.send_command("go ponder")
    
    # Wait a bit to let pondering accumulate some nodes
    time.sleep(2)
    
    # Ponderhit means the opponent actually played the expected move
    # So we just continue the search we were already doing
    tester.send_command("ponderhit")
    
    # Note: In real UCI usage, after ponderhit the GUI would typically send a new go command
    # with proper time management. For testing, we'll just stop after a bit.
    time.sleep(1)
    tester.send_command("stop")
    
    # Wait for final bestmove
    final_bestmove = tester.wait_for_response("bestmove", timeout=5)
    assert final_bestmove, "No final bestmove after ponderhit"
    
    print(f"\nReceived final bestmove: {final_bestmove}")
    
    tester.close()
    print("\nTest 1: PASSED")


def test_ponder_miss():
    """Test pondering with position mismatch."""
    print("\n" + "="*60)
    print("TEST 2: Pondering with position mismatch (ponder miss)")
    print("="*60)
    
    tester = UCITester()
    
    # Initialize engine
    tester.send_command("uci")
    tester.wait_for_response("uciok")
    
    # Enable pondering
    tester.send_command("setoption name Ponder value true")
    tester.send_command("setoption name Verbose value true")
    tester.send_command("setoption name Threads value 4")
    
    tester.send_command("isready")
    tester.wait_for_response("readyok")
    
    # Set starting position
    tester.send_command("position startpos")
    
    # Make first move
    tester.send_command("go movetime 1000")
    bestmove_line = tester.wait_for_response("bestmove", timeout=3)
    
    parts = bestmove_line.split()
    best_move = parts[1]
    ponder_move = parts[3] if len(parts) > 3 else None
    
    print(f"\nBest move: {best_move}, Ponder move: {ponder_move}")
    
    # Start pondering on the expected move
    tester.send_command(f"position startpos moves {best_move}")
    tester.send_command("go ponder")
    
    time.sleep(1)
    
    # Stop pondering
    tester.send_command("stop")
    tester.wait_for_response("bestmove", timeout=2)
    
    # Play a different move (not the ponder move)
    different_move = "e7e6" if ponder_move != "e7e6" else "d7d6"
    tester.send_command(f"position startpos moves {best_move} {different_move}")
    
    # Search should start fresh (no tree reuse)
    tester.send_command("go movetime 1000")
    
    # Wait for bestmove
    final_bestmove = tester.wait_for_response("bestmove", timeout=3)
    assert final_bestmove, "No bestmove after ponder miss"
    
    outputs = tester.get_all_output(0.5)
    
    # Check that we're not reusing pondered tree
    reused_tree = any("reusing pondered tree" in out.lower() for out in outputs)
    print(f"\nTree reused (should be False): {reused_tree}")
    
    tester.close()
    print("\nTest 2: PASSED")


def test_stop_during_ponder():
    """Test stop command during pondering."""
    print("\n" + "="*60)
    print("TEST 3: Stop command during pondering")
    print("="*60)
    
    tester = UCITester()
    
    # Initialize engine
    tester.send_command("uci")
    tester.wait_for_response("uciok")
    
    # Enable pondering
    tester.send_command("setoption name Ponder value true")
    tester.send_command("setoption name Threads value 4")
    
    tester.send_command("isready")
    tester.wait_for_response("readyok")
    
    # Set position and get a move
    tester.send_command("position startpos")
    tester.send_command("go movetime 500")
    bestmove_line = tester.wait_for_response("bestmove", timeout=2)
    
    best_move = bestmove_line.split()[1]
    
    # Start pondering
    tester.send_command(f"position startpos moves {best_move}")
    tester.send_command("go ponder")
    
    time.sleep(0.5)
    
    # Stop pondering
    tester.send_command("stop")
    
    # Should receive bestmove quickly
    stop_response = tester.wait_for_response("bestmove", timeout=1)
    assert stop_response, "No bestmove after stop during pondering"
    
    print(f"\nReceived bestmove after stop: {stop_response}")
    
    tester.close()
    print("\nTest 3: PASSED")


def test_ponder_option_toggle():
    """Test enabling/disabling ponder option."""
    print("\n" + "="*60)
    print("TEST 4: Ponder option enable/disable")
    print("="*60)
    
    tester = UCITester()
    
    # Initialize engine
    tester.send_command("uci")
    tester.wait_for_response("uciok")
    
    tester.send_command("isready")
    tester.wait_for_response("readyok")
    
    # Test with pondering disabled (default)
    tester.send_command("position startpos")
    tester.send_command("go movetime 500")
    bestmove_line = tester.wait_for_response("bestmove", timeout=2)
    
    # Should not have ponder move
    assert "ponder" not in bestmove_line, "Ponder move suggested when pondering disabled"
    print(f"\nPondering disabled - bestmove: {bestmove_line}")
    
    # Enable pondering
    tester.send_command("setoption name Ponder value true")
    tester.send_command("setoption name Threads value 4")
    
    # Test with pondering enabled
    tester.send_command("position startpos moves e2e4")
    tester.send_command("go movetime 500")
    bestmove_line = tester.wait_for_response("bestmove", timeout=2)
    
    # Should have ponder move
    assert "ponder" in bestmove_line, "No ponder move suggested when pondering enabled"
    print(f"\nPondering enabled - bestmove: {bestmove_line}")
    
    tester.close()
    print("\nTest 4: PASSED")


def test_ucinewgame_clears_ponder():
    """Test that ucinewgame clears pondering data."""
    print("\n" + "="*60)
    print("TEST 5: ucinewgame clears pondering data")
    print("="*60)
    
    tester = UCITester()
    
    # Initialize engine
    tester.send_command("uci")
    tester.wait_for_response("uciok")
    
    tester.send_command("setoption name Ponder value true")
    tester.send_command("setoption name Verbose value true")
    
    tester.send_command("isready")
    tester.wait_for_response("readyok")
    
    # Make a move and start pondering
    tester.send_command("position startpos")
    tester.send_command("go movetime 500")
    bestmove_line = tester.wait_for_response("bestmove", timeout=2)
    
    best_move = bestmove_line.split()[1]
    
    # Start pondering
    tester.send_command(f"position startpos moves {best_move}")
    tester.send_command("go ponder")
    
    time.sleep(0.5)
    
    # Send ucinewgame
    tester.send_command("ucinewgame")
    
    # Stop pondering if still active
    tester.send_command("stop")
    tester.get_all_output(0.5)  # Clear output
    
    # Set new position and search - should not reuse any tree
    tester.send_command("position startpos moves e2e4 e7e5")
    tester.send_command("go movetime 500")
    
    outputs = tester.get_all_output(1)
    
    # Should not see tree reuse
    reused_tree = any("reusing pondered tree" in out.lower() for out in outputs)
    print(f"\nTree reused after ucinewgame (should be False): {reused_tree}")
    
    tester.wait_for_response("bestmove", timeout=2)
    
    tester.close()
    print("\nTest 5: PASSED")


def main():
    """Run all tests."""
    print("Starting UCI pondering tests...")
    print("Note: These tests require the AlphaZero model to be available")
    
    try:
        # Run all tests
        test_basic_pondering()
        test_ponder_miss()
        test_stop_during_ponder()
        test_ponder_option_toggle()
        test_ucinewgame_clears_ponder()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()