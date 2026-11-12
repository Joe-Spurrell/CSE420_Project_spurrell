# Upstream attribution

This project is a research derivative of the following code and environment sources:

- **Dronenvai_airl** by Zhe Chen: https://github.com/zitchan/Dronenvai_airl. The upstream repository provides the PPO/AIRL implementation structure, CNN feature extractor, and AirSim interface used as the starting point. The upstream Apache License 2.0 is retained in `LICENSE`.
- **AirsimDRL**: https://github.com/sunghoonhong/AirsimDRL. The packaged Easy AirSim map was obtained separately from this project. The map and Unreal build are not included in this repository.
- **Research paper**: Zhe Chen and Junyu Xuan, “Learning unknown reward function for drone navigation based on inverse deep reinforcement learning,” *Neural Computing and Applications* (2026), https://doi.org/10.1007/s00521-025-11698-1.

Project-specific changes include the normalized-progress reward, shared terminal reward handling, Windows setup and benchmarking utilities, deterministic policy evaluation, and safe PPO checkpoint continuation. The trained checkpoint in `artifacts/progress/` was produced by this project from its progress-reward training run.
