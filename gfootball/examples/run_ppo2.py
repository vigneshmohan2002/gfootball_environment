"""Runs football_env on OpenAI's ppo2."""

import multiprocessing
import os
from absl import app


from stable_baselines import PPO2, logger

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import monitor
from stable_baselines.common.callbacks import BaseCallback

import gfootball.env as football_env
from gfootball.examples import models


def create_single_football_env(iprocess, custom_rewards=False):
    """Creates gfootball environment."""
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        stacked=False,
        rewards="scoring,custom" if custom_rewards else "scoring",
        logdir=logger.get_dir(),
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        dump_frequency=50,
    )
    env = monitor.Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess))
    )
    return env


def train(_):
    """Trains two PPO2 policies. One with custom rewards and one with scoring rewards"""
    # vec_env = DummyVecEnv(
    #     [
    #         (lambda _i=i: create_single_football_env(_i, custom_rewards=True))
    #         for i in range(8)
    #     ],
    # )

    # # Import tensorflow after we create environments. TF is not fork sake, and
    # # we could be using TF as part of environment if one of the players is
    # # controlled by an already trained model.
    # import tensorflow.compat.v1 as tf

    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # ncpu = multiprocessing.cpu_count()
    # config = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     intra_op_parallelism_threads=ncpu,
    #     inter_op_parallelism_threads=ncpu,
    # )
    # config.gpu_options.allow_growth = True
    # tf.Session(config=config).__enter__()

    # model = PPO2(
    #     policy=CnnPolicy,
    #     env=vec_env,
    #     seed=42,
    #     n_steps=512,
    #     nminibatches=4,
    #     noptepochs=2,
    #     max_grad_norm=0.76,
    #     gamma=0.997,
    #     ent_coef=0.00155,
    #     learning_rate=0.00011879,
    #     cliprange=0.115,
    #     tensorboard_log="./ppo2_11_vs_11_easy_stochastic_tensorboard/",
    # )
    # model.learn(
    #     total_timesteps=50_000_000,
    #     log_interval=1,
    #     tb_log_name="11_vs_11_easy_stochastic_custom",
    # )
    # model.save("11_vs_11_easy_stochastic_custom")

    vec_env = DummyVecEnv(
        [
            (lambda _i=i: create_single_football_env(_i, custom_rewards=False))
            for i in range(8)
        ],
    )

    # Import tensorflow after we create environments. TF is not fork sake, and
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

    model = PPO2(
        policy=CnnPolicy,
        env=vec_env,
        seed=42,
        n_steps=512,
        nminibatches=4,
        noptepochs=2,
        max_grad_norm=0.76,
        gamma=0.997,
        ent_coef=0.00155,
        learning_rate=0.00011879,
        cliprange=0.115,
        tensorboard_log="./ppo2_11_vs_11_easy_stochastic_tensorboard/",
    )
    model.learn(
        total_timesteps=50_000_000,
        log_interval=1,
        tb_log_name="11_vs_11_easy_stochastic",
    )
    model.save("11_vs_11_easy_stochastic")


if __name__ == "__main__":
    app.run(train)
