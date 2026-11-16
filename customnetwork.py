import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def obs_split(combined):
    batch_size = combined.shape[0]
    img = combined[:, :, :, :128]
    vel = combined[:, :, :, 128:]
    vel = vel.reshape(batch_size, 1, 72, 3)
    vel = vel[:, 0, 0, :]
    return img, vel


class feature_extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim=96):
        super().__init__(observation_space, features_dim=features_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3[0].weight.data.normal_(0, 0.5)  # initialization
        self.image_fc1 = nn.Linear(256, 64, bias=False)
        self.image_fc2 = nn.Linear(64, 48, bias=False)
        self.image_fc1.weight.data.normal_(0, 0.5)
        self.image_fc2.weight.data.normal_(0, 0.5)

        self.vel_fc1 = nn.Linear(3, 48, bias=False)
        self.vel_fc1.weight.data.normal_(0, 0.5)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, obs):
        image, vel = obs_split(obs)

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.image_fc1(x)
        x = self.relu(x)
        x = self.image_fc2(x)

        vel = self.vel_fc1(vel)

        state_process = torch.cat((x, vel), dim=1)
        state_process = self.tanh(state_process)

        return state_process

