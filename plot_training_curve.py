"""Plot episode performance and PPO training losses for one training run."""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def read_episode_log(path, initial_timesteps):
    episodes = []
    cumulative_steps = 0
    with path.open(newline="", encoding="utf-8") as log_file:
        next(log_file, None)  # Stable-Baselines3 monitor metadata line.
        for row in csv.DictReader(log_file):
            length = int(row["l"])
            cumulative_steps += length
            episodes.append(
                {
                    "timesteps": initial_timesteps + cumulative_steps,
                    "return": float(row["r"]),
                    "length": length,
                }
            )
    return episodes


def read_ppo_log(path):
    with path.open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))

    metrics = {
        "loss": [],
        "value_loss": [],
        "policy_gradient_loss": [],
    }
    for row in rows:
        step_text = row.get("time/total_timesteps", "")
        if not step_text:
            continue
        try:
            step = float(step_text)
        except ValueError:
            continue

        for metric in metrics:
            value_text = row.get(f"train/{metric}", "")
            if not value_text:
                continue
            try:
                metrics[metric].append((step, float(value_text)))
            except ValueError:
                continue
    return metrics


def moving_average(values, window):
    queue = deque()
    total = 0.0
    result = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot PPO episode rewards, episode lengths, and training losses."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="training_log_* directory printed by ppo_train.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/progress_training_curve_with_loss.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    initial_timesteps = int(metadata["initial_timesteps"])
    episodes = read_episode_log(run_dir / "episodes.monitor.csv", initial_timesteps)
    ppo_metrics = read_ppo_log(run_dir / "progress.csv")
    if not episodes:
        raise RuntimeError(f"No completed episodes are logged in {run_dir}.")
    if not any(ppo_metrics.values()):
        raise RuntimeError(f"No PPO loss rows are logged in {run_dir / 'progress.csv'}.")

    fig, (reward_ax, length_ax, loss_ax) = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True
    )
    fig.suptitle(
        f"Progress-reward PPO (alpha={metadata.get('progress_alpha', 1.0):g}) — "
        f"seed {metadata['seed']}, "
        f"AirSim {metadata['clock_speed']:g}x",
        fontsize=17,
        fontweight="bold",
    )

    episode_steps = [row["timesteps"] for row in episodes]
    returns = [row["return"] for row in episodes]
    lengths = [row["length"] for row in episodes]

    reward_ax.scatter(episode_steps, returns, s=18, alpha=0.3, color="#4c78a8", label="Episode return")
    reward_ax.plot(
        episode_steps,
        moving_average(returns, 20),
        color="#174ea6",
        linewidth=2.5,
        label="20-episode mean",
    )
    reward_ax.axhline(0, color="#666666", linestyle="--", linewidth=1.2)
    reward_ax.set_ylabel("Episode return")
    reward_ax.grid(True, alpha=0.25)
    reward_ax.legend(loc="best")

    length_ax.scatter(episode_steps, lengths, s=18, alpha=0.3, color="#72b66b", label="Episode length")
    length_ax.plot(
        episode_steps,
        moving_average(lengths, 20),
        color="#287d3c",
        linewidth=2.5,
        label="20-episode mean",
    )
    length_ax.axhline(300, color="#d62728", linestyle="--", linewidth=1.3, label="300-step limit")
    length_ax.set_ylabel("Environment steps")
    length_ax.set_ylim(0, 315)
    length_ax.grid(True, alpha=0.25)
    length_ax.legend(loc="best")

    colors = {
        "loss": "#7b2cbf",
        "value_loss": "#e07a19",
        "policy_gradient_loss": "#008c95",
    }
    labels = {
        "loss": "Total PPO loss",
        "value_loss": "Value loss",
        "policy_gradient_loss": "Policy-gradient loss",
    }
    for metric, points in ppo_metrics.items():
        if not points:
            continue
        steps, values = zip(*points)
        loss_ax.scatter(steps, values, s=12, alpha=0.2, color=colors[metric])
        loss_ax.plot(
            steps,
            moving_average(values, 10),
            color=colors[metric],
            linewidth=2,
            label=f"{labels[metric]} (10-update mean)",
        )
    loss_ax.axhline(0, color="#666666", linestyle="--", linewidth=1.0)
    loss_ax.set_ylabel("PPO loss")
    loss_ax.set_xlabel("Global PPO timesteps")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend(loc="best")

    end_step = max(
        episode_steps[-1],
        max(point[0] for points in ppo_metrics.values() for point in points),
    )
    loss_ax.xaxis.set_major_locator(MultipleLocator(10000))
    loss_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    loss_ax.set_xlim(initial_timesteps, end_step * 1.02)
    fig.text(
        0.01,
        0.01,
        "PPO losses are recorded once per policy update; episode metrics are recorded when an episode ends.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Training curve saved to {args.output}")


if __name__ == "__main__":
    main()
