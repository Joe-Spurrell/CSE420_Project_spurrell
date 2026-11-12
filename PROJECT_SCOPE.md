# Project component map

## Used from the starting repository

- **PPO algorithm and policy update loop** - `ppo_train.py` uses Stable-Baselines3 PPO. The policy weights are updated from sampled observations, actions, and rewards.
- **CNN feature extractor** - `customnetwork.py` converts the 72x128 depth image and velocity observation into features for the policy.
- **AirSim environment interface** - `airsim_env.py` resets/arms the drone, captures depth and velocity, applies continuous velocity actions, checks collisions, and ends episodes.
- **Action definition** - three continuous values in [-1, 1], mapped to AirSim target velocity in the environment step method.
- **Windows run helpers** - `ppo_train.py` starts Blocks, waits for RPC, and selects a ClockSpeed settings file.

## Project-specific implementation

- **Progress reward** - normalized change in route Y position, with shared +2 goal and -2 failure/timeout terminal rewards. Coefficients are in `config.py` and calculation is in `airsim_env.py`.
- **Evaluation diagnostics** - `run_progress_model.py` loads the saved policy and logs action values, implied forward velocity, and position.
- **Safe resume support** - this project copy of `ppo_train.py` accepts `--resume_from` and requires a separate `--output_dir`, preventing a continuation run from replacing its source checkpoint.
- **Clock-speed study** - `benchmark_environment.py`, `benchmark_clock_speed.ps1`, and `CLOCK_SPEED_TEST.md` document short throughput tests.

## Not yet implemented

- Time-penalty reward component and its coefficient.
- Depth-based clearance reward and its distance statistic, threshold, and scaling.
- The combined progress + time + clearance condition.
- A multi-episode, multi-seed evaluation report with success rate and confidence intervals.
- A confirmed definition of success, including whether reaching the endpoint without collision is sufficient and how the 300-step limit is treated.

## Not included in this research subset

- AIRL discriminator training and reward-network experiments (`airl_train.py`, `airsim_env_airl.py`). The present study compares hand-designed reward functions using PPO.
- The original demonstration collection/training scripts and unused legacy test environments.
- The packaged Unreal/AirSim game build, Python virtual environment, caches, temporary logs, and untracked checkpoints from unrelated runs.
