# Supervisor meeting brief: drone reward-shaping experiment

## Project question

How do adding time and obstacle-clearance terms to a normalized-progress reward affect PPO drone-navigation performance in a fixed AirSim environment?

## Planned comparison

Keep the PPO policy, CNN, action space, depth/velocity inputs, map, training budget, seeds, and evaluation procedure the same. Change only the reward:

1. Shared goal/failure rewards + normalized progress (implemented baseline).
2. Progress + time cost.
3. Progress + depth-derived clearance.
4. Progress + time + clearance.

The currently implemented progress term is `alpha * (Y_t - Y_(t-1)) / route_length`, with alpha=1.0. Goal reward is +2, failure/timeout is -2, and episodes are limited to 300 steps. Time and clearance coefficients still need a justified scale. For a clearance metric, confirm whether the observation should use `DepthPerspective` or `DepthPlanar` float data rather than `DepthVis`, which is a visualization type in AirSim's API documentation.

## What is working

- The packaged Easy AirSim scene starts and accepts RPC commands.
- PPO can train against the environment at 1x and save a checkpoint.
- The saved progress-only checkpoint loads in the visible deterministic runner.
- Action and position traces make policy stalls distinguishable from RPC failures.

## Baseline evidence so far

- Training completed **30,208 timesteps** at 1x, taking about **4.9 hours** on the current machine.
- PPO's reported mean episode length increased from about **31** to **216** steps; mean episode reward moved from about **-1.87** to **-1.53**.
- One deterministic evaluation reached **Y=31.57 of 57**, then ended at the **300-step timeout**, return **-1.446**.
- Near the stall, the policy chose actions corresponding to roughly zero forward speed. This is a useful baseline failure case, not reliable goal completion.
- Short clock-speed tests did not validate 20-30x real-time training. The project is using 1x for a stable baseline.

## Feedback requested

- Should the study compare all four reward conditions, or narrow the scope to three?
- What is a defensible time cost: per decision or per simulated second, and how should its coefficient be scaled against normalized progress?
- What depth statistic and forward region best represent obstacle clearance? What minimum safe clearance threshold is appropriate?
- Should timeout count as failure with the same terminal penalty as collision? What success criterion and step budget should be used?
- How many random seeds and evaluation episodes per reward condition are realistic for the project schedule?
- Is the substitute Easy map suitable, or should the study use the paper's exact environment if available?

## Next steps

1. Continue the progress-only checkpoint at 1x in a separate output folder; retain the current checkpoint as the baseline.
2. Agree on the time and clearance formulas, units, and coefficient scales before implementing them.
3. Add reward variants while keeping observations, action mapping, PPO settings, map, seeds, and training budget fixed.
4. Evaluate each checkpoint over the same starts and seeds; report success rate, collision rate, final route progress, episode length, and reward.

**Reference:** Zhe Chen and Junyu Xuan, “Learning unknown reward function for drone navigation based on inverse deep reinforcement learning,” *Neural Computing and Applications* (2026), https://doi.org/10.1007/s00521-025-11698-1.
