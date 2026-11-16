import argparse
import math
import os
import socket
import time

import numpy as np


def wait_for_api(timeout_seconds=90):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 41451), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("AirSim did not open its API port 41451 within 90 seconds.")


def main():
    parser = argparse.ArgumentParser(description="Measure AirSim environment step throughput.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--clock-speed", type=float, required=True)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    configured_speed = float(os.environ.get("AIRSIM_CLOCK_SPEED", "1"))
    if not math.isclose(configured_speed, args.clock_speed):
        raise ValueError("AIRSIM_CLOCK_SPEED and --clock-speed must match")

    print(f"Waiting for AirSim at 127.0.0.1:41451 ({args.clock_speed:g}x)...", flush=True)
    wait_for_api()

    from airsim_env import AirSimEnv_Progress

    env = AirSimEnv_Progress()
    client = env.client
    completed_steps = 0
    ended_early = False
    terminal_status = None
    terminal_y = None
    terminal_z = None
    try:
        env.reset(seed=123)
        sim_start_ns = client.getMultirotorState().timestamp
        wall_start = time.perf_counter()

        # Zero policy output maps to a gentle 0.5 m/s forward command in this project.
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step(action)
            completed_steps += 1
            if terminated or truncated:
                ended_early = True
                terminal_status = info.get("status", "unknown")
                terminal_y = info.get("Y")
                terminal_z = info.get("Z")
                break

        wall_seconds = time.perf_counter() - wall_start
        sim_end_ns = client.getMultirotorState().timestamp
        sim_seconds = max(0.0, (sim_end_ns - sim_start_ns) / 1_000_000_000)
    finally:
        try:
            client.simPause(True)
            client.enableApiControl(False)
        except Exception:
            pass

    factor = sim_seconds / wall_seconds if wall_seconds > 0 else 0.0
    print(f"Configured ClockSpeed: {args.clock_speed:g}x")
    print(f"Environment steps completed: {completed_steps}/{args.steps}")
    if ended_early:
        print(f"Terminal status: {terminal_status} (Y={terminal_y}, Z={terminal_z})")
    print(f"Simulated time: {sim_seconds:.2f} seconds")
    print(f"Wall time: {wall_seconds:.2f} seconds")
    print(f"Measured environment rate: {factor:.2f}x real time")
    print(f"Environment interaction rate: {completed_steps / wall_seconds:.2f} steps/second")


if __name__ == "__main__":
    main()
