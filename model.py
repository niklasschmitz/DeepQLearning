from jax.experimental import stax
from jax.experimental.stax import Dense, Conv, Relu, Flatten


def DeepQNetwork():
    init_fun, predict_fun = stax.serial(
        Conv(16, (8, 8), strides=(4, 4)), Relu,
        Conv(32, (4, 4), strides=(2, 2)), Relu,
        Conv(64, (3, 3)), Relu,
        Flatten,
        Dense(256), Relu,
        Dense(6)
    )
    return init_fun, predict_fun
