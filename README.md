# AirSim Drone Reward Experiments

This repository contains the project-specific code and first baseline artifacts for a fourth-year study of reward shaping for vision-based drone navigation in AirSim.

## Research question

How do normalized route-progress, time, and depth-based obstacle-clearance reward terms affect PPO navigation performance in the same drone environment?

The current experiment is based on the PPO implementation and AirSim interface from [Dronenvai_airl](https://github.com/zitchan/Dronenvai_airl), with an Easy map obtained separately from [AirsimDRL](https://github.com/sunghoonhong/AirsimDRL). The downloaded Unreal environment is not stored in this repository.

## Reward conditions

All variants are intended to use the same terminal rewards: +2 for reaching the goal and -2 for failure or timeout.

| Condition | Reward terms | Status |
| --- | --- | --- |
| Progress | Shared terminal reward + normalized route progress | Implemented in `airsim_env.py`; initial PPO training and evaluation completed |
| Progress + time | Progress + per-step or per-second time cost | Planned; coefficient and time definition need to be agreed |
| Progress + clearance | Progress + depth-derived obstacle-clearance term | Planned; clearance metric and scaling need to be agreed |
| Progress + time + clearance | All terms above | Planned |

The implemented progress term is `alpha * (Y_t - Y_(t-1)) / route_length`, with `alpha = 1.0`. The current episode limit is 300 steps. The existing observation uses a 72x128 depth image and a 3-axis velocity vector; the clearance reward has not been implemented. For a quantitative clearance term, confirm whether to request AirSim `DepthPerspective` or `DepthPlanar` float data rather than the current `DepthVis` image. AirSim documents `DepthVis` as a visualization, while `DepthPlanar` and `DepthPerspective` provide depth values. See [AirSim image APIs](https://microsoft.github.io/AirSimExtensions/image_apis/).

## Current baseline result

The 30,208-step progress-only training run showed gradual improvement in mean episode length and reward. One deterministic evaluation reached Y=31.57 on a 57 m route, then timed out at the 300-step limit with return -1.446. The action trace showed the policy choosing near-zero forward speed around this point. This is preliminary evidence of a stall, not evidence that the baseline reliably completes the route. See [`results/progress_baseline_summary.md`](results/progress_baseline_summary.md) and the training curve in `results/`.

## Project files

| File | Purpose |
| --- | --- |
| `airsim_env.py` | AirSim Gymnasium environment, depth/velocity observation, continuous actions, and progress reward |
| `config.py` | Reward weights and episode-step limit |
| `customnetwork.py` | CNN feature extractor used by PPO |
| `ppo_train.py` | PPO training entry point; includes `--resume_from` and `--output_dir` for safe continuation |
| `run_progress_model.py` | Deterministic policy evaluation with per-step action and position logs |
| `benchmark_environment.py`, `benchmark_clock_speed.ps1` | AirSim interaction throughput and clock-speed checks |
| `environment/clock_test/settings-1.json` | Minimal 1x AirSim settings used by the scripts |
| `requirements-windows.txt`, `SETUP_WINDOWS.md` | Python dependencies and Windows setup instructions |
| `MEETING_BRIEF.md`, `PROJECT_SCOPE.md` | Supervisor discussion brief and project component map |
| `artifacts/progress/` | Snapshot of the initial progress-only PPO checkpoint (30,208 training steps) |
| `results/` | Baseline summary, sanitized episode metrics, and training curve |
| `LICENSE`, `UPSTREAM_ATTRIBUTION.md` | License and source acknowledgements |

The repository excludes the Unreal/AirSim build, Python virtual environment, temporary logs, and future training outputs. Download the compatible Easy environment separately and keep checkpoints locally or attach them as versioned research artifacts when appropriate.

## Quick start

See [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md) for the full setup. To evaluate the included checkpoint after installing dependencies and downloading the Easy AirSim environment:

```powershell
$blocks = (Resolve-Path "C:\path\to\Easy\Blocks.exe").Path
python .\run_progress_model.py --exe_path $blocks --model .\artifacts\progress\ppo_final_progress_30208_steps.zip --log_every 5
```

The evaluation runner starts AirSim at 1x with a visible camera. Close other Blocks instances before starting it.

## Continue the baseline training

The initial checkpoint is preserved in `artifacts/progress/`. To continue it for an additional 30,000 steps, use the `--resume_from` option and save to a new output directory. Substitute the path to your local Blocks executable and Python environment:

```powershell
$blocks = (Resolve-Path "C:\path\to\Easy\Blocks.exe").Path
python .\ppo_train.py --exe_path $blocks --mode progress --seed 3 --train_step 30000 --n_steps 512 --batch_size 64 --clock_speed 1 --resume_from .\artifacts\progress\ppo_final_progress_30208_steps.zip --output_dir .\model\ppo_checkpoints\progress_extended
```

`--train_step` means additional timesteps when resuming. A required, separate `--output_dir` prevents overwriting the source run. The original training log shows about 4.9 hours for 30,208 steps at 1x on the current machine; actual duration depends on the computer and environment.

## Reference paper

Zhe Chen and Junyu Xuan, “Learning unknown reward function for drone navigation based on inverse deep reinforcement learning,” *Neural Computing and Applications* (2026), https://doi.org/10.1007/s00521-025-11698-1. This project uses the paper repository as a code starting point and studies hand-designed reward variants; it is not a reproduction of the paper's AIRL experiment.
