# Search Agent Improvements

## Original Weaknesses

The original `C4DepthLimitedAlphaBetaAgent` had:
- Only center-out move ordering (no winning-move-first)
- No transposition table
- No iterative deepening
- Fixed depth (no time budget option)
- Heuristic: `-80` for opponent 3-in-a-row (inconsistently weighted vs own 3)

## Improvements in AdvancedAlphaBetaC4Agent

### 1. Winning-Move-First Ordering
At each node, moves are sorted:
1. Moves that immediately win (current player scores 4-in-a-row)
2. Moves that block opponent's immediate win
3. All other moves: center-out ordering

**Why it helps:** Alpha-beta efficiency depends critically on move ordering.
Exploring winning moves first triggers immediate cutoffs. Exploring opponent
threats early also enables deep pruning. Combined with center-out ordering
for neutral moves, this dramatically reduces the effective branching factor.

### 2. Transposition Table
A dictionary `{board.tobytes() → (depth, flag, value, best_move)}` stores
previously computed subtree results.

TT entry flags (standard minimax convention):
- `EXACT`: `α < value < β` — exact node, can return immediately
- `LOWER`: `value ≥ β` — lower bound (cut-node); update α
- `UPPER`: `value ≤ α_orig` — upper bound (all-node); update β

**Why it helps:** Connect4 positions can be reached via many different move
sequences (transpositions). Without a TT, these are recomputed. With TT,
they're looked up in O(1). This is especially valuable in Connect4 because
the same position is reachable via many column orderings.

**Safety:** TT is cleared at the start of each `select_action()` call to
avoid stale entries from previous game positions.

### 3. Iterative Deepening
Search is run at depth 1, then 2, ..., then `max_depth`. After each
completed depth, the best move is stored as a fallback.

**Why it helps:**
- If a time budget is exceeded mid-depth, the best move from the last
  completed depth is returned (always a valid move)
- Move ordering at depth d+1 can use the best move from depth d
- Provides "anytime" behaviour: can use with or without time budget

### 4. Improved Heuristic
Windows of 4 cells are scored symmetrically:
- Own 3-in-a-row with 1 empty:  +10
- Own 2-in-a-row with 2 empty:  +2
- Opp 3-in-a-row with 1 empty:  −50  (was −80 in original, but inconsistent)
- Opp 2-in-a-row with 2 empty:  −3

Center column weighting is extended to the two adjacent columns (±1),
not just the exact center, reflecting the actual strategic value of
central columns in Connect4.

### 5. Time Budget Support
An optional `time_budget` (seconds) causes the search to raise `_Timeout`
when exceeded. Iterative deepening ensures a valid move always exists
from the last completed depth.

## Correctness Guarantee

Move ordering and TT usage do not change the minimax value — only the
efficiency of finding it. The TT is only used when `tt_depth >= depth`,
preventing incorrect cutoffs from shallow entries.

## Performance Expectation

At depth 8, the Advanced Alpha-Beta should consistently beat:
- Random agent: ~100% win rate
- Default agent: ~95%+ win rate (P1 and P2 combined)

It does not guarantee optimal play (Connect4 is solved — first player wins
with perfect play) but is designed to be very strong in practice.
