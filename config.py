MAX_EPISODE_STEPS = 300

reward = {
    'forward': 0.1,
    'normal': -0.1,
    'slow': -0.05,
    'dead': -2.0,
    'goal': 2.0,
    # Alpha in r_t = B_t + alpha * normalized route progress.
    # A complete route contributes approximately +1 before the goal bonus.
    'progress_alpha': 1.0,
}
