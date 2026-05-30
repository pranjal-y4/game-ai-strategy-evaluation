"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
TTT_QLEARNING = {
    "episodes": 60_000,
    "lr": 0.15,
    "gamma": 0.95,
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.05,
    "n_step": 3,
    "eval_every": 2_000,
    "curriculum_frac": 0.6,
    "seed": 42,
}


C4_QLEARNING_REDUCED = {
    "episodes": 150_000,
    "rows": 4,
    "cols": 5,
    "lr": 0.1,
    "gamma": 0.95,
    "epsilon_decay": 0.9999,
    "epsilon_min": 0.05,
    "n_step": 3,
    "eval_every": 5_000,
    "curriculum_frac": 0.6,
    "seed": 42,

}


TTT_DQN = {
    "episodes": 20_000,
    "hidden": [128, 64],
    "lr": 5e-4,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "decay_steps": 15_000,
    "batch_size": 64,
    "buffer_size": 20_000,
    "target_update": 200,
    "n_step": 3,
    "use_per": True,
    "eval_every": 1_000,
    "curriculum_frac": 0.6,
    "seed": 42,
}


C4_DQN = {
    "episodes": 100_000,
    "rows": 6,
    "cols": 7,
    "hidden": [256, 128],
    "lr": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "decay_steps": 60_000,
    "batch_size": 64,
    "buffer_size": 50_000,
    "target_update": 500,
    "n_step": 3,
    "use_per": True,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "eval_every": 2_500,
    "curriculum_frac": 0.6,
    "phase2_epsilon_reset": 0.3,
    "reward_shaping": True,
    "shaping_weight": 0.01,
    "shaping_clip": 0.1,
    "seed": 42,
    "grad_clip": 10.0,
}


ADV_ALPHABETA_C4 = {
    "max_depth": 8,
    "time_budget": None,
}


EVALUATION = {
    "n_games": 200,
    "seed": 42,
}
