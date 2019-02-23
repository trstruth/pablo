from pablo.envs import Canvas
from pablo.agents import Agent

from gym.envs.registration import register

register(
    id='pablo-v0',
    entry_point='pablo.envs:Canvas',
)
