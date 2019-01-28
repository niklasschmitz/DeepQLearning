import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Dense, GeneralConv, Relu


def DeepQNetwork():
    weights_init, predict = stax.serial(
        GeneralConv(), Relu,
        GeneralConv(), Relu,
        GeneralConv(), Relu,
        Dense(512), Relu,
        stax.Flatten(),
        Dense(6)
    )
    return weights_init, predict


def main():
    pass
