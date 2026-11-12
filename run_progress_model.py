"""Run the trained progress-reward PPO policy in the visible AirSim scene."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from airsim_env import AirSimEnv_Progress
from ppo_train import (
    airsim_port_is_open,
    configure_clock_speed,
    start_external_program,
    wait_for_airsim,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained progress-reward PPO model in AirSim."
    )
    parser.add_argument(
        "--exe_path",
        required=True,
        help="Path to environment\\Easy\\Blocks.exe",
    )
    parser.add_argument(
        "--model",
        default="model/ppo_checkpoints/progress/ppo_final_progress.zip",
        help="PPO checkpoint to run",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Print action and position every N steps (use 1 for every step).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")
    if args.log_every <= 0:
        parser.error("--log_every must be a positive integer")

    if airsim_port_is_open():
        raise RuntimeError(
            "AirSim is already open. Close the existing Blocks window first so this "
            "runner can apply ClockSpeed 1x and the FlyWithMe view."
        )

    configure_clock_speed(1)
    simulator = start_external_program(args.exe_path, view_mode="FlyWithMe")
    print("Started Blocks at 1x with the FlyWithMe view.", flush=True)
    wait_for_airsim(simulator)

    env = DummyVecEnv([lambda: AirSimEnv_Progress()])
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=False,
        clip_reward=100.0,
        training=False,
    )

    try:
        model = PPO.load(str(model_path), env=env, device=args.device)
        print(f"Loaded model: {model_path}", flush=True)
        print("Running deterministic policy episodes...", flush=True)

        for episode in range(1, args.episodes + 1):
            observation = env.reset()
            total_reward = 0.0
            final_info = {}
            steps = 0

            for steps in range(1, config.MAX_EPISODE_STEPS + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                final_info = infos[0]

                if steps % args.log_every == 0 or dones[0]:
                    action_values = action[0].tolist()
                    forward_speed = 1.5 * float(action_values[1]) + 0.5
                    print(
                        f"step={steps:03d} action="
                        f"({action_values[0]:+.3f}, {action_values[1]:+.3f}, "
                        f"{action_values[2]:+.3f}) "
                        f"commanded_forward={forward_speed:+.3f} m/s "
                        f"Y={final_info.get('Y', float('nan')):.2f} "
                        f"Z={final_info.get('Z', float('nan')):.2f} "
                        f"status={final_info.get('status', 'unknown')}",
                        flush=True,
                    )
                if dones[0]:
                    break

            print(
                f"Episode {episode}: status={final_info.get('status', 'unknown')}, "
                f"steps={steps}, Y={final_info.get('Y', float('nan')):.2f}, "
                f"Z={final_info.get('Z', float('nan')):.2f}, "
                f"return={total_reward:.3f}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
