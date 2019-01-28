import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Dense, Conv, Relu, Flatten


def DeepQNetwork():
    weights_init, predict = stax.serial(
        Conv(32, (8,8)), Relu,
        Conv(64, (4,4)), Relu,
        Conv(128, (3,3)), Relu,
        Dense(512), Relu,
        Flatten,
        Dense(6)
    )
    return weights_init, predict


def main():
    pass
