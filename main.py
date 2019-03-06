import jax.numpy as np
from jax import jit, grad
from jax.experimental import optimizers
import numpy as onp
import gym

from model import DeepQNetwork

GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
ALPHA = 0.001
ACTION_SPACE = [0, 1, 2, 3, 4, 5]
MEM_SIZE = 5000


def preprocess(observation):
    return np.mean(observation[15:200, 30:125], axis=2, keepdims=True)


def store_transition(memory, mem_cnt, state, action, reward, state_):
    if mem_cnt < MEM_SIZE:
        memory.append([state, action, reward, state_])
    else:
        memory[mem_cnt % MEM_SIZE] = [state, action, reward, state_]
    mem_cnt += 1
    return memory, mem_cnt


def choose_action(observation, pred_Q, params_Q_eval, eps):
    rand = onp.random.random()
    actions = pred_Q(params_Q_eval, observation)
    print(actions)
    if rand < 1 - eps:
        action = np.argmax(actions[1])
    else:
        action = onp.random.choice(ACTION_SPACE)
    return action


def main():
    env = gym.make('SpaceInvaders-v0')

    memory = []
    mem_cnt = 0

    # # fill memory with random interactions with the environment
    # while len(memory) < MEM_SIZE:
    #     observation = env.reset()
    #     done = False
    #     while not done:
    #         # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
    #         action = env.action_space.sample()
    #         observation_, reward, done, info = env.step(action)
    #         if done and info['ale.lives'] == 0:
    #             reward = -100
    #         state = np.mean(observation[15:200, 30:125], axis=2, keepdims=True)
    #         state_ = np.mean(observation_[15:200, 30:125], axis=2, keepdims=True)
    #         memory, mem_cnt = store_transition(memory, mem_cnt, state, action, reward, state_)
    #         observation = observation_
    # print('done initializing memory')

    init_Q, pred_Q = DeepQNetwork()

    # two separate Q-Table approximations (eval and next) for Double Q-Learning
    # initialize parameters, not committing to a batch size (NHWC)
    # we choose 3 channels as we want to pass stacks of 3 consecutive frames
    in_shape = (-1, 185, 95, 3)
    _, params_Q_eval = init_Q(in_shape)
    _, params_Q_next = init_Q(in_shape)

    opt_init, opt_update = optimizers.rmsprop(ALPHA)

    # Define a simple squared-error loss
    def loss(params, batch):
        inputs, targets = batch
        predictions = pred_Q(params, inputs)
        return np.sum((predictions - targets) ** 2)

    # Define a compiled update step
    @jit
    def step(j, opt_state, batch):
        params = optimizers.get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(j, g, opt_state)

    scores = []
    eps_history = []
    num_games = 50
    batch_size = 2  # 32

    steps = 0
    eps = EPS_START
    learn_step_counter = 0

    def learn(j, params_Q_eval, params_Q_next):
        opt_state_Q_eval = opt_init(params_Q_eval)

        if mem_cnt + batch_size < MEM_SIZE:
            mem_start = int(onp.random.choice(range(mem_cnt)))
        else:
            mem_start = int(onp.random.choice(range(MEM_SIZE - batch_size - 1)))
        mini_batch = memory[mem_start: mem_start + batch_size]
        mini_batch = np.array(mini_batch)

        input_states = np.stack(tuple(mini_batch[:, 0][:]))
        next_states = np.stack(tuple(mini_batch[:, 3][:]))

        predicted_Q = pred_Q(params_Q_eval, input_states)
        predicted_Q_next = pred_Q(params_Q_next, next_states)

        max_action = np.argmax(predicted_Q_next, axis=1)
        rewards = np.array(list(mini_batch[:, 2]))

        Q_target = onp.array(predicted_Q)
        Q_target[:, max_action] = rewards + GAMMA * np.max(predicted_Q_next[0])

        # batch = (inputs, targets)
        opt_state_Q_eval = step(j, opt_state_Q_eval, (input_states, Q_target))
        params_Q_eval = optimizers.get_params(opt_state_Q_eval)

        return params_Q_eval, params_Q_next

    for i in range(num_games):
        print('starting game ', i + 1, 'epsilon: %.4f' % eps)
        eps_history.append(eps)
        done = False
        observation = env.reset()
        frames = [preprocess(observation)]
        score = 0
        last_action = 0
        while not done:
            j = 0
            # only choose an action at every 3rd frame
            if len(frames) == 3:
                action = choose_action(frames, pred_Q, params_Q_eval, eps)
                steps += 1
                frames = []
            else:
                action = last_action
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(preprocess(observation))
            if done and info['ale.lives'] == 0:
                reward = -100
            memory, mem_cnt = store_transition(memory, mem_cnt,
                                               preprocess(observation),
                                               action, reward,
                                               preprocess(observation_))
            observation = observation_

            # TODO learn step
            params_Q_eval, params_Q_next = learn(j, params_Q_eval, params_Q_next)
            j += 1

            if steps > 500:
                if eps - 1e-4 > EPS_END:
                    eps -= 1e-4
                else:
                    eps = EPS_END

            last_action = action
        scores.append(score)
        print('score: ', score)


if __name__ == '__main__':
    main()
