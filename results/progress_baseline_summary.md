# Progress-only baseline snapshot

## Training

- PPO timesteps: 30,208
- Configured AirSim ClockSpeed: 1x
- Wall-clock duration: 17,565 seconds (about 4.9 hours)
- Mean episode length: about 31 steps at the first 512-step report; 216 at the final report
- Mean episode reward: about -1.87 at the first report; -1.53 at the final report

These are the training logger's rolling metrics. They indicate learning progress but do not by themselves establish route success.

## Deterministic evaluation

- Episodes: 1
- Status: timeout
- Steps: 300
- Final Y: 31.57 on a route ending at Y=57 (about 55% of route length)
- Final Z: -0.82
- Return: -1.446

The trace showed forward commands shrinking toward zero around step 225, followed by small positive and negative commands while Y stayed near 31.5-31.8. The episode did not reach the goal. This is a single evaluation episode, not a robust success-rate estimate.

## Artifacts

- `../artifacts/progress/ppo_final_progress_30208_steps.zip` - checkpoint snapshot from the run.
- `progress_training_metrics.csv` - episode return and length, without machine-specific timestamp metadata.
- `progress_training_curve.png` - training curve generated from the sanitized episode metrics.
