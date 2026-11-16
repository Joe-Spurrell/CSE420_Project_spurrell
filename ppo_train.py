from gymnasium.envs.registration import register
import gymnasium as gym
import torch
import random
import subprocess
import socket
import time
import json
import shutil
import numpy as np
import os
import config
import argparse
from stable_baselines3.common.utils import get_device
from customnetwork import feature_extractor as rnetwork
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor


def start_external_program(exe_path, view_mode=None):
    command = [exe_path, '-windowed', '-ResX=1280', '-ResY=720']
    settings_path = os.environ.get('AIRSIM_SETTINGS_PATH')
    if not settings_path:
        project_settings = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'environment',
            'clock_test',
            'settings-1.json',
        )
        if os.path.isfile(project_settings):
            settings_path = project_settings

    if settings_path:
        settings_path = os.path.abspath(settings_path)
        project_root = os.path.dirname(os.path.abspath(__file__))
        working_directory = os.path.dirname(os.path.abspath(settings_path))
        active_settings_path = os.path.join(
            project_root, 'environment', 'Easy', 'Custom', 'Binaries', 'Win64', 'settings.json'
        )
        os.makedirs(os.path.dirname(active_settings_path), exist_ok=True)
        if view_mode:
            with open(settings_path, 'r', encoding='utf-8') as settings_file:
                settings = json.load(settings_file)
            settings['ViewMode'] = view_mode
            with open(active_settings_path, 'w', encoding='ascii') as settings_file:
                json.dump(settings, settings_file, indent=2)
        elif os.path.normcase(settings_path) != os.path.normcase(active_settings_path):
            shutil.copyfile(settings_path, active_settings_path)
        print(f"Using AirSim settings: {active_settings_path}", flush=True)
        command.append(f'-settings={active_settings_path}')
    else:
        working_directory = None
    return subprocess.Popen(command, cwd=working_directory)


def airsim_port_is_open(host='127.0.0.1', port=41451):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def wait_for_airsim(process, host='127.0.0.1', port=41451, timeout_seconds=120):
    """Wait until the launched simulator exposes its AirSim RPC port."""
    deadline = time.monotonic() + timeout_seconds
    print(f"Waiting for AirSim RPC at {host}:{port} (up to {timeout_seconds}s)...", flush=True)

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"AirSim simulator exited before opening RPC port {port} "
                f"(exit code {exit_code}). Check the Blocks window and simulator logs."
            )

        try:
            with socket.create_connection((host, port), timeout=1.0):
                print("AirSim RPC port is ready.", flush=True)
                return
        except OSError:
            time.sleep(1.0)

    raise TimeoutError(
        f"AirSim did not open RPC port {port} within {timeout_seconds}s. "
        "Check whether Blocks is still running and whether it loaded the expected settings."
    )


def configure_clock_speed(requested_speed=None):
    """Keep the simulator ClockSpeed and the environment's step timing aligned."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    settings_dir = os.path.join(project_root, 'environment', 'clock_test')
    existing_settings = os.environ.get('AIRSIM_SETTINGS_PATH')
    existing_speed = os.environ.get('AIRSIM_CLOCK_SPEED')
    explicit_request = requested_speed is not None or bool(existing_settings) or bool(existing_speed)

    if requested_speed is not None:
        speed = float(requested_speed)
        settings_path = os.path.join(settings_dir, f"settings-{speed:g}.json")
    elif existing_settings:
        settings_path = os.path.abspath(existing_settings)
        with open(settings_path, 'r', encoding='utf-8') as settings_file:
            speed = float(json.load(settings_file).get('ClockSpeed', 1))
        if existing_speed is not None and not np.isclose(float(existing_speed), speed):
            raise ValueError(
                f"AIRSIM_CLOCK_SPEED={existing_speed} does not match "
                f"ClockSpeed={speed:g} in {settings_path}"
            )
    elif existing_speed is not None:
        speed = float(existing_speed)
        settings_path = os.path.join(settings_dir, f"settings-{speed:g}.json")
    else:
        speed = 1.0
        settings_path = os.path.join(settings_dir, 'settings-1.json')

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('AirSim clock speed must be a positive finite number')
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(f"AirSim settings file not found: {settings_path}")

    os.environ['AIRSIM_CLOCK_SPEED'] = f'{speed:g}'
    os.environ['AIRSIM_SETTINGS_PATH'] = os.path.abspath(settings_path)
    return speed, os.path.abspath(settings_path), explicit_request


def main():
    # arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--clip_range_vf', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.15)
    parser.add_argument('--train_step', type=int, default=260000)
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Continue PPO training from a saved PPO zip checkpoint.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override the checkpoint directory; required with --resume_from to preserve the source checkpoint.',
    )
    parser.add_argument('--clock_speed', type=int, choices=[1, 2, 3, 4, 5, 10, 20, 30], default=None,
                        help='AirSim ClockSpeed; the matching settings file is selected automatically')
    parser.add_argument('--view', action='store_true',
                        help='Show the AirSim scene with the FlyWithMe camera during training')
    parser.add_argument('--mode', choices=['progress'], default='progress',
                        help='Reward function to train (the progress-only baseline)')
    args = parser.parse_args()
    if args.resume_from and not args.output_dir:
        parser.error('--output_dir is required with --resume_from so the source run is preserved.')

    clock_speed, settings_path, explicit_clock_request = configure_clock_speed(args.clock_speed)
    explicit_clock_request = explicit_clock_request or args.view
    print(f"AirSim ClockSpeed: {clock_speed:g}x ({settings_path})", flush=True)
    print(f"AirSim view: {'FlyWithMe' if args.view else 'NoDisplay'}", flush=True)
    if airsim_port_is_open():
        if explicit_clock_request:
            raise RuntimeError(
            "AirSim is already running. ClockSpeed and view settings only take effect when Blocks starts. "
            "Close the existing Blocks simulator, then rerun this command."
            )
        simulator_process = None
        print("AirSim RPC is already listening; reusing that simulator.", flush=True)
    else:
        simulator_process = start_external_program(
            args.exe_path, view_mode='FlyWithMe' if args.view else None
        )
        print(f"Started AirSim simulator process (PID {simulator_process.pid}).", flush=True)
        wait_for_airsim(simulator_process)
    mode = args.mode
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    print("Training PPO with shared terminal rewards plus normalized route progress.", flush=True)
    register(
        id='AirSim/AirSimEnv-progress',
        entry_point='airsim_env:AirSimEnv_Progress',
        max_episode_steps=config.MAX_EPISODE_STEPS,
    )
    env = gym.make('AirSim/AirSimEnv-progress')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=100.0)
    checkpoint_saving_path = './model/ppo_checkpoints/progress'
    if clock_speed != 1.0:
        checkpoint_saving_path = os.path.join(
            './model/ppo_checkpoints', f'{mode}_clock{clock_speed:g}'
        )
    if args.output_dir:
        checkpoint_saving_path = args.output_dir
    os.makedirs(checkpoint_saving_path, exist_ok=True)
    print(f"Checkpoints will save in {checkpoint_saving_path}", flush=True)
    env = VecMonitor(env, filename='monitor_log.txt')

    device = get_device()
    policy_kwargs = dict(
        features_extractor_class=rnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], qf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-1.8
    )

    print(device)
    if args.resume_from:
        print(f"Resuming PPO from: {args.resume_from}")
        model = PPO.load(args.resume_from, env=env, device="cuda")
    else:
        model = PPO(
            env=env,
            policy=ActorCriticPolicy,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.9,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            normalize_advantage=True,
            vf_coef=0.15,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.n_epochs,
            seed=random_seed,
            device="cuda",
            verbose=2,
        )
        model.policy.log_std.data[1] += 0.1
        print(model.policy.log_std)

    model.set_random_seed(random_seed)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_saving_path,
                                             name_prefix='ppo_model')

    model.learn(
        total_timesteps=args.train_step,
        reset_num_timesteps=not bool(args.resume_from),
        callback=checkpoint_callback,
    )
    final_model_path = os.path.join(checkpoint_saving_path, f"ppo_final_{mode}")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}.zip")


if __name__ == '__main__':
    main()
