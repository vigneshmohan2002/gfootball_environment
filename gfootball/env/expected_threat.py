from gfootball.env import expected_goals as xG
import numpy as np
from math import floor


def calculate_threat_from_pass_points(pass_origin, pass_target):
    pass_origin = xG.to_normalized_space(pass_origin)
    pass_target = xG.to_normalized_space(pass_target)

    xT = np.load("gfootball/env/xT.npy")

    # xT is a 2D array, (16,12), need to get the appropriate x and y indices from the pass_origin and pass_target
    x_origin = floor(pass_origin[0] * 16)
    y_origin = floor(pass_origin[1] * 12)

    x_target = floor(pass_target[0] * 16)
    y_target = floor(pass_target[1] * 12)

    try:
        origin_xT = xT[x_origin, y_origin]
        target_xT = xT[x_target, y_target]
        return target_xT - origin_xT
    except IndexError:
        return 0.0
