"""
experiments/run_match.py
────────────────────────────────────────────────────────────────────────────
Single-game runner and agent registry.

Usage:
    python experiments/run_match.py --game ttt --agent1 minimax --agent2 default --games 50 --seed 42

Agent names (--agent1 / --agent2):
    ttt : random, default, minimax, alphabeta, qlearning_ttt, dqn_ttt
    c4  : random, default, c4_alphabeta, qlearning_c4, dqn_c4
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from games.tictactoe_core import TicTacToe
from games.connect4_core import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.ttt_minimax_agent import TTTMinimaxAgent
from agents.ttt_alphabeta_agent import TTTAlphaBetaAgent
from agents.c4_depthlimited_alphabeta_agent import C4DepthLimitedAlphaBetaAgent
from agents.qlearning_ttt_agent import QLearningTTTAgent
from agents.qlearning_c4_reduced_agent import QLearningC4ReducedAgent
from agents.dqn_ttt_agent import DQNTTTAgent
from agents.dqn_c4_agent import DQNC4Agent
from utils.metrics import MetricsCollector
from utils.seed import set_seed


def get_agent(name: str, game_type: str, model_paths: dict = None):
    """Return an agent instance by name. Returns None for unknown names.

    model_paths: optional dict mapping agent key → model file path override.
      Supported keys: 'qlearning_ttt', 'qlearning_c4', 'dqn_ttt', 'dqn_c4'
      If a key is absent or None the agent falls back to its default path.
    """
    mp = model_paths or {}
    # RL agents accept an optional model_path override
    if name == 'qlearning_ttt':
        return QLearningTTTAgent(model_path=mp.get('qlearning_ttt'))
    if name == 'qlearning_c4':
        return QLearningC4ReducedAgent(model_path=mp.get('qlearning_c4'))
    if name == 'dqn_ttt':
        return DQNTTTAgent(model_path=mp.get('dqn_ttt'))
    if name == 'dqn_c4':
        return DQNC4Agent(model_path=mp.get('dqn_c4'))
    registry = {
        'random':        RandomAgent,
        'default':       DefaultAgent,
        'minimax':       TTTMinimaxAgent,
        'alphabeta':     TTTAlphaBetaAgent,
        'ttt_minimax':   TTTMinimaxAgent,
        'ttt_alphabeta': TTTAlphaBetaAgent,
        'c4_alphabeta':  C4DepthLimitedAlphaBetaAgent,
    }
    cls = registry.get(name)
    return cls() if cls else None


def play_game(game, agent1, agent2, game_num: int = 0):
    """
    Play a single game between agent1 and agent2.

    Alternation: even game_num → agent1 is board player 1 (moves first);
                 odd  game_num → agent2 is board player 1 (moves first).

    Returns
    -------
    winner_code : int  1 = agent1 won, 2 = agent2 won, 0 = draw
    game_length : int  number of moves played
    agent1_time : float  total wall-clock seconds spent by agent1
    agent2_time : float  total wall-clock seconds spent by agent2
    agent1_nodes: int   total nodes expanded by agent1 (0 if N/A)
    agent2_nodes: int   total nodes expanded by agent2 (0 if N/A)
    """
    game.reset()
    agent1.reset()
    agent2.reset()

    agents = [agent1, agent2]

    # Determine who moves first.
    # agent1_is_player1=True  → agents[0]=agent1 plays as board player 1
    # agent1_is_player1=False → agents[1]=agent2 plays as board player 1
    agent1_is_player1 = (game_num % 2 == 0)
    current_idx = 0 if agent1_is_player1 else 1

    game_length = 0
    agent1_time = 0.0
    agent2_time = 0.0
    agent1_nodes = 0
    agent2_nodes = 0

    while not game.is_terminal():
        agent = agents[current_idx]
        if hasattr(agent, 'nodes_expanded'):
            agent.nodes_expanded = 0

        t0 = time.perf_counter()
        action = agent.select_action(game)
        legal = game.legal_moves()
        if action not in legal:
            raise ValueError(
                f"{agent.name} produced illegal move {action}. Legal moves: {legal}"
            )

        elapsed = time.perf_counter() - t0

        # Map current_idx to actual agent identity
        is_agent1 = (current_idx == 0)
        if is_agent1:
            agent1_time += elapsed
            if hasattr(agent, 'nodes_expanded'):
                agent1_nodes += agent.nodes_expanded
        else:
            agent2_time += elapsed
            if hasattr(agent, 'nodes_expanded'):
                agent2_nodes += agent.nodes_expanded

        game.apply_move(action)
        current_idx = 1 - current_idx
        game_length += 1

    # Map board winner (1 or 2) to agent winner (1=agent1, 2=agent2).
    # Board player 1 = agent at initial position (agent1 if agent1_is_player1, else agent2).
    board_winner = game.winner()  # 1, 2, or 0
    if board_winner == 0:
        winner_code = 0
    elif board_winner == 1:
        # Board player 1 won.
        winner_code = 1 if agent1_is_player1 else 2
    else:
        # Board player 2 won.
        winner_code = 2 if agent1_is_player1 else 1

    return winner_code, game_length, agent1_time, agent2_time, agent1_nodes, agent2_nodes


def main():
    parser = argparse.ArgumentParser(description="Run a headless match between two agents")
    parser.add_argument('--game',    choices=['ttt', 'c4'], required=True)
    parser.add_argument('--agent1',  required=True,  help='Agent 1 name')
    parser.add_argument('--agent2',  required=True,  help='Agent 2 name')
    parser.add_argument('--games',   type=int, default=100, help='Number of games')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--output_dir', default='experiments/results')
    args = parser.parse_args()

    set_seed(args.seed)

    game = TicTacToe() if args.game == 'ttt' else Connect4()
    agent1 = get_agent(args.agent1, args.game)
    agent2 = get_agent(args.agent2, args.game)

    if agent1 is None:
        print(f"Unknown agent: {args.agent1}")
        return
    if agent2 is None:
        print(f"Unknown agent: {args.agent2}")
        return

    collector = MetricsCollector()
    for i in range(args.games):
        result = play_game(game, agent1, agent2, game_num=i)
        collector.record_game(agent1.name, agent2.name, *result)

    summary = collector.get_summary()
    print(f"\n{args.agent1} vs {args.agent2}  ({args.games} games, seed={args.seed})")
    print(f"  Wins:  {summary['agent1_wins']} / {summary['agent2_wins']} / {summary['draws']} "
          f"(agent1 / agent2 / draw)")
    print(f"  Win rates:  {summary['agent1_win_rate']:.1%} / "
          f"{summary['agent2_win_rate']:.1%} / {summary['draw_rate']:.1%}")
    print(f"  Avg game length: {summary['avg_game_length']:.1f} moves")
    print(f"  Avg time/game:  agent1={summary['avg_agent1_time']*1000:.1f}ms  "
          f"agent2={summary['avg_agent2_time']*1000:.1f}ms")
    if summary['avg_agent1_nodes'] > 0 or summary['avg_agent2_nodes'] > 0:
        print(f"  Avg nodes/game: agent1={summary['avg_agent1_nodes']:.0f}  "
              f"agent2={summary['avg_agent2_nodes']:.0f}")

    # Optionally save
    from utils.serialization import save_csv, get_timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.join(args.output_dir,
        f"match_{args.game}_{args.agent1}_vs_{args.agent2}_{get_timestamp()}.csv")
    fieldnames = list(summary.keys()) + ['agent1_name', 'agent2_name', 'game', 'seed', 'timestamp']
    row = dict(summary)
    row.update({'agent1_name': args.agent1, 'agent2_name': args.agent2,
                'game': args.game, 'seed': args.seed, 'timestamp': get_timestamp()})
    save_csv([row], filename, list(row.keys()))
    print(f"  Results saved → {filename}")


if __name__ == '__main__':
    main()
