import time
import numpy as np
import airsim
import config
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

step_duration = 0.5
goalY = 57
outY = -0.5
floorZ = 1.18
MAX_EPISODE_STEPS = config.MAX_EPISODE_STEPS


def obs_combine(img, vel):
    vel = np.tile(vel[np.newaxis, np.newaxis, :], (1, 72, 1))
    combined = np.concatenate((img, vel), axis=-1)
    return combined


class AirSimEnvBase(gym.Env):
    def __init__(self):
        super().__init__()

        # connect to the AirSim simulator
        # Keep RPC failures bounded so a missing simulator does not hang for
        # AirSim's one-hour default request timeout.
        self.client = airsim.MultirotorClient(timeout_value=30)
        self.client.confirmConnection()
        self.img_seq = None
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.infty, high=np.infty, shape=(1, 72, 131),
                                            dtype=np.float32)

    def transform_input(self, osb):
        responses = osb[0]
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        image = Image.fromarray(img2d)
        image = np.array(image.resize((72, 128)).convert('L'))
        image = np.float32(image.reshape(1, 72, 128))
        image /= 255.
        vel = np.array(osb[1], dtype=np.float32)

        return [image, vel]

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * step_duration).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * step_duration).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        observation = self.get_state()
        image = observation[0]
        vel = observation[1]
        self.img_seq = image
        history = self.img_seq.copy()
        obs = obs_combine(history, vel)
        info = {}
        return obs, info

    def get_state(self):
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        observation = self.transform_input(observation)
        return observation

    def step(self, quad_offset):
        # print("action: ", quad_offset)
        # move with given velocity
        quad_offset = [float(i) for i in quad_offset]
        x, y, z = quad_offset
        # print(x, y, z)
        self.client.simPause(False)
        has_collided = False
        landed = False
        movement = self.client.moveByVelocityAsync(
            1.5 * x, 1.5 * y + 0.5, 1.5 * z, step_duration
        )
        # Wait for the requested simulated duration, regardless of ClockSpeed.
        movement.join()
        self.client.simPause(True)
        collision_count = 0
        for _ in range(11):
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided
            # A stationary/hovering drone can have zero velocity; use its floor
            # position to detect landing instead of treating that as a terminal.
            landed = quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        try:
            observation = self.get_state()
        except Exception as e:
            time.sleep(10)
            observation = self.get_state()

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position

        # decide whether done
        dead = has_collided or quad_pos.y_val <= outY
        done = dead or quad_pos.y_val >= goalY

        # compute reward
        reward = self.compute_reward(quad_pos, observation[1], dead)
        self.img_seq = observation[0]
        history_ = self.img_seq.copy()
        obs_ = obs_combine(history_, observation[1])

        # log info
        info = {}
        info['Y'] = quad_pos.y_val
        info['Z'] = quad_pos.z_val
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.y_val >= goalY:
            info['status'] = 'goal'
        else:
            info['status'] = 'going'

        # print(f'action: {quad_offset}')

        return obs_, reward, done, False, info

    def render(self):
        pass

    def compute_reward(self, quad_pos, vel, dead):
        raise NotImplementedError('A reward function must be provided by the environment subclass.')

    def close(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')


class AirSimEnv_Progress(AirSimEnvBase):
    """Progress-only reward with the same success/failure terminal rewards."""

    def __init__(self):
        super().__init__()
        self.route_start_y = None
        self.previous_y = None
        self.episode_steps = 0
        self.last_progress_fraction = 0.0
        self.last_progress_reward = 0.0
        self.last_terminal_reward = 0.0

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        position = self.client.getMultirotorState().kinematics_estimated.position
        self.route_start_y = float(position.y_val)
        self.previous_y = self.route_start_y
        self.episode_steps = 0
        self.last_progress_fraction = 0.0
        self.last_progress_reward = 0.0
        self.last_terminal_reward = 0.0
        return observation, info

    def compute_reward(self, quad_pos, vel, dead):
        current_y = float(quad_pos.y_val)
        route_length = goalY - self.route_start_y
        if route_length <= 0:
            raise RuntimeError(
                f"Route length must be positive; start y={self.route_start_y}, goal y={goalY}"
            )

        progress_fraction = (current_y - self.previous_y) / route_length
        self.previous_y = current_y

        self.last_progress_fraction = float(progress_fraction)
        self.last_progress_reward = float(
            config.reward['progress_alpha'] * progress_fraction
        )
        self.last_terminal_reward = 0.0

        if dead:
            self.last_terminal_reward = float(config.reward['dead'])
        elif current_y >= goalY:
            self.last_terminal_reward = float(config.reward['goal'])

        return self.last_progress_reward + self.last_terminal_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_steps += 1

        # The Gymnasium TimeLimit wrapper also uses this limit. Giving the
        # timeout its shared failure penalty makes the terminal reward explicit.
        if (
            self.episode_steps >= MAX_EPISODE_STEPS
            and not terminated
            and not truncated
        ):
            self.last_terminal_reward = float(config.reward['dead'])
            reward += self.last_terminal_reward
            terminated = True
            info['status'] = 'timeout'

        info['progress_fraction'] = self.last_progress_fraction
        info['progress_reward'] = self.last_progress_reward
        info['terminal_reward'] = self.last_terminal_reward
        return observation, float(reward), terminated, truncated, info
