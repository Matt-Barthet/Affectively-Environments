from stable_baselines3 import PPO

from affectively_environments.envs.heist import HeistEnvironment
from affectively_environments.envs.solid import SolidEnvironment
from affectively_environments.envs.pirates import PiratesEnvironment

import sys
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    if len(sys.argv) != 4:
        print(f"Incorrect number of arguments specified, was expecting 7, found {len(sys.argv)}")
        exit()
    else:
        run = int(sys.argv[1])
        weight = float(sys.argv[2])
        game = sys.argv[3]

    env = HeistEnvironment(id_number=run, weight=weight, graphics=True, logging=True, path="./Builds/Heist.app")
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard", device='cpu')
    model.learn(total_timesteps=2000000, progress_bar=True)
    model.save(f"ppo_solid_{label}_{run}")
