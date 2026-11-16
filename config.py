MAX_EPISODE_STEPS = 300

reward = {
    'dead': -2.0,
    'goal': 2.0,
    # Alpha in r_t = B_t + alpha * normalized route progress.
    # A complete route contributes approximately +1 before the goal bonus.
    'progress_alpha': 1.0,
}
