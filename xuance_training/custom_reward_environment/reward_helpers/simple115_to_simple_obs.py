import numpy as np
import typing


def observation_converter(observation):
    reward_shaping_observation = {}
    reward_shaping_observation["left_team_coords"] = observation[0:22].reshape(-1, 2)
    reward_shaping_observation["right_team_coords"] = observation[44:66].reshape(-1, 2)
    reward_shaping_observation["ball_coords"] = observation[88:91]
    reward_shaping_observation["ball_direction"] = observation[91:94]
    reward_shaping_observation["ball_owned_team"] = (
        np.argmax(observation[94:96] == 1) if 1 in observation[94:96] else -1
    )
    reward_shaping_observation["ball_owned_player"] = (
        np.argmax(observation[96:107] == 1) if 1 in observation[96:107] else -1
    )

    return reward_shaping_observation
