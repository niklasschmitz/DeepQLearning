import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Dense, Conv, Relu, Flatten


def DeepQNetwork():
    init_fun, predict_fun = stax.serial(
        Conv(32, (8, 8)), Relu,
        Conv(64, (4, 4)), Relu,
        Conv(128, (3, 3)), Relu,
        Flatten,
        Dense(512), Relu,
        Dense(6)
    )
    return init_fun, predict_fun
