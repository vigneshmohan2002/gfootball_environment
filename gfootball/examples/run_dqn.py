"""Runs football_env on OpenAI's DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from stable_baselines import DQN, logger
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import monitor
import multiprocessing
import os
import gfootball.env as football_env
from gfootball.examples import models
from stable_baselines.common.callbacks import CheckpointCallback
import time


def create_single_football_env():
    """Creates gfootball environment."""
    env = football_env.create_environment(
        env_name="11_vs_11_hard_stochastic",
        stacked=(True),
        rewards="scoring",
        logdir=logger.get_dir(),
        write_goal_dumps=True,
        write_full_episode_dumps=False,
        render=False,
        dump_frequency=50,
    )
    env = monitor.Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(0))
    )
    return env


def train(gamma, lr, buffer_size, num_timesteps, save_interval, num_envs):
    """Trains a DQN."""
    logger.configure()
    vec_env = create_single_football_env()

    # Import tensorflow after we create environments. TF is not fork sake, and
    # we could be using TF as part of environment if one of the players is
    # controlled by an already trained model.. TF is not fork sake, and
    # we could be using TF as part of environment if one of the players is
    # controlled by an already trained model.
    import tensorflow.compat.v1 as tf

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    # Define the DQN model
    model = DQN(
        "MlpPolicy",
        vec_env,
        gamma=gamma,
        learning_rate=lr,
        buffer_size=buffer_size,
        verbose=1,
    )
    # Train the DQN model
    verbose_callback = CheckpointCallback(
        verbose=1, save_freq=10000000, save_path="./checkpoints/"
    )
    model.learn(
        total_timesteps=num_timesteps, log_interval=1, callback=verbose_callback
    )

    # Save the trained model
    model.save(".")


if __name__ == "__main__":
    gamma = 0.997
    lr = 0.00011879
    buffer_size = 50000
    num_timesteps = 50000000
    save_interval = 10000000
    num_envs = 8
    train(gamma, lr, buffer_size, num_timesteps, save_interval, num_envs)
