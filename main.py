import jax.numpy as np
from jax import jit, grad
from jax.experimental import optimizers
import numpy as onp
import gym
import os
from collections import deque

from model import DeepQNetwork

GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
ALPHA = 0.001
ACTION_SPACE = [0, 1, 2, 3, 4, 5]
MEM_SIZE = 5000
STACK_SIZE = 4  # number of consecutive frames to stack as input to the network
NUM_GAMES = 50
BATCH_SIZE = 4
TAU = 5  # replacement of Qnext

LOAD = True  # load pretrained weights to continue training
WEIGHTS_PATH = "weights"

LEARN = False  # disable for evaluation without further training
RENDER = True  # display the game during training


def load_params(path):
    """ load pretrained params/weights of the DQN """
    p = onp.load(path)
    q = [(np.array(t[0]), np.array(t[1])) if len(t) == 2 else () for t in p]
    return q


def preprocess(observation):
    return np.mean(observation[15:200, 30:125], axis=2)


def store_transition(memory, state, action, reward, state_):
    memory.append([state, action, reward, state_])
    return memory


def sample(memory, batch_size):
    indices = onp.random.choice(onp.arange(len(memory)), size=batch_size, replace=False)
    batch = [memory[i] for i in indices]
    return batch


def choose_action(observation, pred_Q, params_Q_eval, eps):
    rand = onp.random.random()
    actions = pred_Q(params_Q_eval, observation)
    if rand < 1 - eps:
        action = np.argmax(actions[0])
    else:
        action = onp.random.choice(ACTION_SPACE)
    return action


def stack_frames(frames):
    """ Takes a deque of frames and stacks them into a numpy array """
    return np.stack(frames, axis=-1)


def main():
    env = gym.make('SpaceInvaders-v0')

    memory = deque(maxlen=MEM_SIZE)

    # fill memory with random interactions with the environment
    while len(memory) < MEM_SIZE:
        observation = env.reset()
        frames = deque([np.zeros((185, 95)) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        frames.append(preprocess(observation))
        state = stack_frames(frames)
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            frames.append(preprocess(observation_))
            state_ = stack_frames(frames)
            memory = store_transition(memory, state, action, reward, state_)
            state = state_
    print('done initializing memory')

    init_Q, pred_Q = DeepQNetwork()

    # two separate Q-Table approximations (eval and next)
    # initialize parameters, not committing to a batch size (NHWC)
    # we choose 3 channels as we want to pass stacks of 3 consecutive frames
    in_shape = (-1, 185, 95, STACK_SIZE)
    _, params_Q_next = init_Q(in_shape)

    if LOAD:
        path = os.path.join(WEIGHTS_PATH, "params_Q_eval.npy")
        params_Q_eval = load_params(path)
    else:
        _, params_Q_eval = init_Q(in_shape)
    params_Q_next = params_Q_eval.copy()

    # Initialize RMSProp optimizer
    opt_init, opt_update = optimizers.rmsprop(ALPHA)
    opt_state = opt_init(params_Q_eval)
    opt_step = 0

    # Define a simple mean-squared-error loss
    def loss(params, batch):
        inputs, targets = batch
        predictions = pred_Q(params, inputs)
        return np.mean((predictions - targets) ** 2)

    # Define a compiled update step
    @jit
    def step(j, opt_state, batch):
        params = optimizers.get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(j, g, opt_state)

    def learn(opt_step, opt_state, params_Q_eval, params_Q_next):
        mini_batch = sample(memory, BATCH_SIZE)

        if opt_step % TAU == 0:
            params_Q_next = params_Q_eval.copy()

        input_states = np.stack([transition[0] for transition in mini_batch])
        next_states = np.stack([transition[3] for transition in mini_batch])

        predicted_Q = pred_Q(params_Q_eval, input_states)
        predicted_Q_next = pred_Q(params_Q_next, next_states)

        max_action = np.argmax(predicted_Q_next, axis=1)
        rewards = np.array([transition[2] for transition in mini_batch])

        Q_target = onp.array(predicted_Q)
        Q_target[:, max_action] = rewards + GAMMA * np.max(predicted_Q_next, axis=1)

        opt_state = step(opt_step, opt_state, (input_states, Q_target))
        params_Q_eval = optimizers.get_params(opt_state)

        return opt_state, params_Q_eval, params_Q_next

    scores = []
    eps_history = []
    eps = EPS_START if LEARN else 0

    for i in range(NUM_GAMES):
        print('starting game ', i + 1, 'epsilon: %.4f' % eps)
        eps_history.append(eps)
        done = False
        observation = env.reset()
        frames = deque([np.zeros((185, 95)) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        frames.append(preprocess(observation))
        state = stack_frames(frames)
        score = 0
        while not done:
            action = choose_action(state.reshape((1, 185, 95, STACK_SIZE)), pred_Q, params_Q_eval, eps)
            observation_, reward, done, info = env.step(action)
            score += reward

            if RENDER:
                env.render()

            if LEARN:
                frames.append(preprocess(observation))
                state_ = stack_frames(frames)
                memory = store_transition(memory, state, action, reward, state_)
                state = state_
                opt_state, params_Q_eval, params_Q_next = learn(opt_step, opt_state,
                                                                params_Q_eval, params_Q_next)
                opt_step += 1

                if opt_step > 500:
                    if eps - 1e-4 > EPS_END:
                        eps -= 1e-4
                    else:
                        eps = EPS_END

        if LEARN:
            out_path = os.path.join(WEIGHTS_PATH, 'params_Q_eval_' + str(i))

            onp.save(out_path, params_Q_eval)
        scores.append(score)
        print('score: ', score)


if __name__ == '__main__':
    main()
