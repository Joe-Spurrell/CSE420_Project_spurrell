# Windows setup

## Prerequisites

- Windows 10 or 11
- Python 3.10
- A compatible packaged AirSim Blocks environment. The project does not redistribute Unreal binaries; obtain the Easy map from the upstream environment source listed in `UPSTREAM_ATTRIBUTION.md`.
- NVIDIA GPU and a CUDA-enabled PyTorch build are recommended for training. CPU can be used for evaluation by passing `--device cpu` to `run_progress_model.py`.

## Create the Python environment

Open PowerShell in this project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir --no-build-isolation msgpack-rpc-python==0.4.1
python -m pip install --no-cache-dir --no-build-isolation -r requirements-windows.txt
```

The CUDA PyTorch command matches the setup used for the baseline machine; select a PyTorch build compatible with your GPU and driver if your hardware differs.

## Evaluate the included progress checkpoint

Close any already-running Blocks simulator, then set the path to the downloaded executable and run:

```powershell
$blocks = (Resolve-Path "C:\path\to\Easy\Blocks.exe").Path
python .\run_progress_model.py --exe_path $blocks --model .\artifacts\progress\ppo_final_progress_30208_steps.zip --log_every 5
```

The runner starts a visible AirSim scene at 1x, waits for RPC, loads the PPO checkpoint, and prints its selected action, commanded forward speed, and position. An episode ending with `timeout` is a failed completion, even if it made partial route progress.

## Continue the existing PPO model

The base upstream training script did not resume from the final progress checkpoint. This project copy adds `--resume_from` and requires `--output_dir` when resuming. The following starts from the included 30,208-step checkpoint and trains for 30,000 additional steps while saving in a new directory:

```powershell
$blocks = (Resolve-Path "C:\path\to\Easy\Blocks.exe").Path
python .\ppo_train.py --exe_path $blocks --mode progress --seed 3 --train_step 30000 --n_steps 512 --batch_size 64 --clock_speed 1 --resume_from .\artifacts\progress\ppo_final_progress_30208_steps.zip --output_dir .\model\ppo_checkpoints\progress_extended
```

Close other Blocks windows first so the script can start the simulator with the requested clock setting. Keep the original checkpoint unchanged so it remains the baseline artifact.

## Add another reward condition

Reward computation is implemented in `airsim_env.py`; tunable coefficients and the episode limit are in `config.py`. Keep PPO, observations, action scaling, map, seeds, training budget, and evaluation procedure fixed across reward conditions. Add each reward as a separate mode or environment class, then record the exact formula and coefficients before comparing results.

For a clearance reward, the current policy observation comes from `DepthVis`. AirSim describes `DepthPlanar` and `DepthPerspective` as depth-valued image types and `DepthVis` as a visualization image. Confirm the image type and compute a defined statistic over a specified forward region, with consistent units and clipping, before assigning a clearance reward.

## Environment speed

The 20-step clock checks are summarized in `CLOCK_SPEED_TEST.md`. They do not establish stable full-episode training at high clock speeds. The current reproducible baseline uses 1x.
