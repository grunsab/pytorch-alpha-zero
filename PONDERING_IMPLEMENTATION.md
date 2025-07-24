# Pondering Implementation Summary

This document describes the pondering feature added to the AlphaZero UCI chess engine.

## Overview

Pondering allows the engine to think during the opponent's turn, improving response time when the opponent plays the expected move. The implementation follows the UCI protocol standard for pondering.

## Key Changes

### 1. MCTS.py
- Modified `Root.__init__()` to accept optional `reuse_subtree` parameter for tree reuse
- Added `getBestMoveChild()` to extract child node for the best move
- Added `getSecondBestMove()` for alternative ponder move suggestions
- Added `getN()` method to Root class for accessing node count

### 2. uci_engine.py
- Added pondering state management:
  - `pondering`: Current pondering state
  - `ponder_enabled`: Whether pondering is enabled (UCI option)
  - `ponder_move`: Suggested move to ponder on
  - `ponder_hit`: Threading event for ponderhit synchronization
  - `saved_root`: Stored MCTS tree for reuse
  - `ponder_board`: Board position being pondered

- Modified `search_position()` to:
  - Accept `ponder` parameter
  - Support tree reuse when position matches pondered position
  - Save tree state when pondering
  - Handle ponderhit during search

- Updated `go()` command to:
  - Parse "ponder" parameter
  - Return immediately when pondering (non-blocking)
  - Output ponder move in bestmove response

- Added `ponderhit()` command handler:
  - Verifies position matches pondered position
  - Signals search thread to continue if match
  - Stops search and outputs bestmove if no match

- Updated other commands:
  - `stop`: Outputs bestmove when stopping ponder
  - `position`: Clears pondering data
  - `ucinewgame`: Clears pondering data
  - `setoption`: Added "Ponder" option (default: false)

## How It Works

1. **Normal Search**: Engine calculates best move and suggests opponent's likely response
   ```
   bestmove e2e4 ponder c7c5
   ```

2. **Pondering**: GUI sets position after both moves and starts pondering
   ```
   position startpos moves e2e4 c7c5
   go ponder
   ```

3. **Ponderhit**: If opponent plays expected move, search continues with saved tree
   ```
   ponderhit
   ```

4. **Ponder Miss**: If opponent plays different move, new search starts fresh

## Benefits

- Faster response when opponent plays expected move
- Better time management by utilizing opponent's thinking time
- Backward compatible - pondering is disabled by default
- Thread-safe implementation with proper synchronization

## Testing

Comprehensive test suite (`test_pondering.py`) covers:
- Basic pondering with ponderhit
- Ponder miss scenarios
- Stop during pondering
- Enable/disable pondering option
- Position changes clearing ponder data

## Usage

To enable pondering in a UCI GUI:
```
setoption name Ponder value true
```

The engine will automatically suggest ponder moves and handle pondering when the GUI supports it.