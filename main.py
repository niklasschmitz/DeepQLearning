import jax.numpy as np
from jax import jit, grad
from jax.experimental import minmax

import gym
from gym import wrappers

from model import Agent

GAMMA = 0.95
EPSILON = 1.0
EPS_END = 0.05
ALPHA = 0.001
ACTION_SPACE = [0,1,2,3,4,5]
MEM_SIZE = 5000

if __name__ == '__main__':
	env = gym.make('SpaceInvaders-v0')

	opt_init, opt_update = minmax.rmsprop(ALPHA)

