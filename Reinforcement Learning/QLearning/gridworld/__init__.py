from gridworld.gridworld_env import GridworldEnv

from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gridworld:GridworldEnv',
)
