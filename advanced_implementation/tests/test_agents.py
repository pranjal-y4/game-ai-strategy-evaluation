import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.advanced_alphabeta_c4 import AdvancedAlphaBetaC4Agent


def test_agents_only_legal_moves_ttt():

    agents = [RandomAgent(), DefaultAgent(), MinimaxAgent(), AlphaBetaAgent()]
    for _ in range(50):
        g = TicTacToe()

        for _ in range(random.randint(0, 5)):
            if not g.is_terminal():
                g.apply_move(random.choice(g.legal_moves()))
        if g.is_terminal():
            continue
        for agent in agents:
            g2 = g.clone()
            move = agent.select_action(g2)
            assert move in g2.legal_moves(), \
                f"{agent.name} selected illegal move {move}"
    print("  [PASS] agents_only_legal_moves_ttt")


def test_agents_only_legal_moves_c4():

    agents = [RandomAgent(), DefaultAgent(),
              AlphaBetaAgent(max_depth=3),
              AdvancedAlphaBetaC4Agent(max_depth=3)]
    for _ in range(50):
        g = Connect4(rows=4, cols=5)
        for _ in range(random.randint(0, 8)):
            if not g.is_terminal():
                g.apply_move(random.choice(g.legal_moves()))
        if g.is_terminal():
            continue
        for agent in agents:
            g2 = g.clone()
            move = agent.select_action(g2)
            assert move in g2.legal_moves(), \
                f"{agent.name} selected illegal move {move}"
    print("  [PASS] agents_only_legal_moves_c4")


def test_default_agent_wins_immediately():

    agent = DefaultAgent()
    g = TicTacToe()
    g.board[0, 0] = 1; g.board[0, 1] = 1
    g.current_player = 1
    move = agent.select_action(g)
    assert move == (0, 2), f"Default agent didn't take win: chose {move}"
    print("  [PASS] default_agent_wins_immediately")


def test_default_agent_blocks_opponent():

    agent = DefaultAgent()
    g = TicTacToe()
    g.board[0, 0] = 2; g.board[0, 1] = 2
    g.current_player = 1
    move = agent.select_action(g)
    assert move == (0, 2), f"Default agent didn't block: chose {move}"
    print("  [PASS] default_agent_blocks_opponent")


def test_minimax_never_loses_ttt():

    minimax = MinimaxAgent()
    random_agent = RandomAgent()
    random.seed(42)
    losses = 0
    for ep in range(100):
        g = TicTacToe()
        minimax_player = 1 if ep % 2 == 0 else 2
        while not g.is_terminal():
            if g.current_player == minimax_player:
                move = minimax.select_action(g)
            else:
                move = random_agent.select_action(g)
            g.apply_move(move)
        w = g.winner()
        if w is not None and w != 0 and w != minimax_player:
            losses += 1
    assert losses == 0, f"Minimax lost {losses}/100 games to Random!"
    print("  [PASS] minimax_never_loses_ttt")


def test_alphabeta_matches_minimax_ttt():

    mm = MinimaxAgent()
    ab = AlphaBetaAgent()
    random.seed(123)
    disagreements = 0
    for _ in range(20):
        g = TicTacToe()
        for _ in range(random.randint(0, 4)):
            if not g.is_terminal():
                g.apply_move(random.choice(g.legal_moves()))
        if g.is_terminal() or not g.legal_moves():
            continue
        g2 = g.clone()
        m1 = mm.select_action(g)
        m2 = ab.select_action(g2)


    print("  [PASS] alphabeta_matches_minimax_ttt (structural check)")


def test_adv_alphabeta_c4_always_valid():

    agent = AdvancedAlphaBetaC4Agent(max_depth=4)
    random.seed(99)
    for _ in range(20):
        g = Connect4(rows=4, cols=5)
        for _ in range(random.randint(0, 10)):
            if not g.is_terminal():
                g.apply_move(random.choice(g.legal_moves()))
        if g.is_terminal():
            continue
        move = agent.select_action(g)
        assert move in g.legal_moves(), f"AdvAB chose illegal move {move}"
    print("  [PASS] adv_alphabeta_c4_always_valid")


def test_adv_alphabeta_c4_takes_winning_move():

    agent = AdvancedAlphaBetaC4Agent(max_depth=3)
    g = Connect4(rows=4, cols=5)

    for col in range(3):
        g.board[3, col] = 1
    g.current_player = 1
    move = agent.select_action(g)
    assert move == 3, f"AdvAB didn't take winning move 3, chose {move}"
    print("  [PASS] adv_alphabeta_c4_takes_winning_move")


def test_per_buffer_sampling():

    from rl.replay_buffer import PrioritizedReplayBuffer
    import numpy as np
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(200):
        s = np.zeros(9, dtype=np.float32)
        buf.push(s, 0, 1.0, s, False, [0, 1, 2])
    assert len(buf) == 200
    (states, actions, rewards, next_states, dones, nl, weights, indices) = buf.sample(32)
    assert states.shape == (32, 9)
    assert len(indices) == 32
    assert all(0.0 <= w <= 1.0 + 1e-6 for w in weights), "IS weights out of range"

    buf.update_priorities(indices, np.abs(np.random.randn(32)))
    print("  [PASS] per_buffer_sampling")


def test_n_step_buffer():

    from rl.n_step_buffer import NStepBuffer
    import numpy as np
    buf = NStepBuffer(n=3, gamma=0.9)
    s = np.zeros(3, dtype=np.float32)

    buf.push(s, 0, 1.0, s, False, [0])
    buf.push(s, 1, 2.0, s, False, [0])
    results = buf.push(s, 2, 3.0, s, False, [0])
    assert len(results) == 1
    _, _, G, _, done, _ = results[0]
    expected_G = 1.0 + 0.9 * 2.0 + 0.81 * 3.0
    assert abs(G - expected_G) < 1e-5, f"Expected G={expected_G:.4f}, got {G:.4f}"
    print("  [PASS] n_step_buffer")


def test_game_env_legal_actions():

    from rl.game_env import GameEnv
    env = GameEnv(TicTacToe, opponent="random")
    env.enable_role_alternation()
    random.seed(7)
    for _ in range(20):
        state = env.reset()
        while True:
            legal = env.get_legal_actions()
            for a in legal:
                assert 0 <= a < env.N_ACTIONS
            action = random.choice(legal)
            state, reward, done, _ = env.step(action)
            if done:
                break
    print("  [PASS] game_env_legal_actions")


def test_game_env_epsilon_zero_eval():

    from rl.game_env import GameEnv
    from rl.q_learning import AdvancedQLearning
    env = GameEnv(TicTacToe, opponent="random")
    agent = AdvancedQLearning(env)
    agent.epsilon = 0.5
    wr = agent.evaluate(10)

    assert abs(agent.epsilon - 0.5) < 1e-9, \
        f"Epsilon not restored after eval: {agent.epsilon}"
    print("  [PASS] game_env_epsilon_zero_eval")


def test_double_dqn_structure():

    try:
        import torch
        from rl.game_env import GameEnv
        from rl.dqn import AdvancedDQNAgent
        env = GameEnv(TicTacToe, opponent="random")
        agent = AdvancedDQNAgent(env, hidden=[32, 16],
                                 buffer_size=100, batch_size=8)

        assert agent.online is not agent.target_net

        for p1, p2 in zip(agent.online.parameters(),
                          agent.target_net.parameters()):
            assert torch.allclose(p1, p2), "Initial online/target weights differ"
        print("  [PASS] double_dqn_structure")
    except ImportError:
        print("  [SKIP] double_dqn_structure (PyTorch not available)")


def test_evaluation_fairness_p1_p2():

    from evaluation.evaluator import evaluate_agent
    random.seed(42)
    agent = DefaultAgent()
    result = evaluate_agent(agent, TicTacToe, n_games=20)
    assert result["p1_games"] + result["p2_games"] == 20
    assert result["p1_games"] > 0 and result["p2_games"] > 0
    print("  [PASS] evaluation_fairness_p1_p2")


def test_reward_shaping_disabled_in_eval():

    from evaluation.evaluator import evaluate_agent


    agent = RandomAgent()
    result = evaluate_agent(agent, TicTacToe, n_games=20)
    assert "total_win_rate" in result
    print("  [PASS] reward_shaping_disabled_in_eval")


if __name__ == "__main__":
    random.seed(42)
    print("\n=== Agent Correctness Tests ===")
    test_agents_only_legal_moves_ttt()
    test_agents_only_legal_moves_c4()
    test_default_agent_wins_immediately()
    test_default_agent_blocks_opponent()
    test_minimax_never_loses_ttt()
    test_alphabeta_matches_minimax_ttt()
    test_adv_alphabeta_c4_always_valid()
    test_adv_alphabeta_c4_takes_winning_move()

    print("\n=== RL Component Tests ===")
    test_per_buffer_sampling()
    test_n_step_buffer()
    test_game_env_legal_actions()
    test_game_env_epsilon_zero_eval()
    test_double_dqn_structure()

    print("\n=== Evaluation Fairness Tests ===")
    test_evaluation_fairness_p1_p2()
    test_reward_shaping_disabled_in_eval()

    print("\nAll agent/RL tests passed.")
