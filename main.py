import jax.numpy as np
from jax import jit, grad
from jax.experimental import minmax

import gym
from gym import wrappers

from model import DeepQNetwork

GAMMA = 0.95
EPSILON = 1.0
EPS_END = 0.05
ALPHA = 0.001
ACTION_SPACE = [0,1,2,3,4,5]
MEM_SIZE = 5000

def store_transition(memory, mem_cnt, state, action, reward, state_):
    if mem_cnt < MEM_SIZE:
        memory.append([state, action, reward, state_])
    else:
        memory[mem_cnt % MEM_SIZE] = [state, action, reward, state_]
    mem_cnt += 1
    return memory, mem_cnt

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')

    memory = []
    mem_cnt = 0

    # fill memory with random interactions with the environment
    while len(memory) < MEM_SIZE:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            state = np.mean(observation[15:200, 30:125], axis=2)
            state_ = np.mean(observation_[15:200, 30:125], axis=2)
            memory, mem_cnt = store_transition(memory, mem_cnt, state, action, reward, state_)
            observation = observation_
    print('done initializing memory')

    # two seperate Q-Table approximations for Double Q-Learning
    init_Q_eval, pred_Q_eval = DeepQNetwork()
    init_Q_next, pred_Q_next = DeepQNetwork()

    opt_init, opt_update = minmax.rmsprop(ALPHA)

    scores = []
    eps_history = []
    num_games = 50
    batch_size = 32

    for i in range(num_games):
        pass
