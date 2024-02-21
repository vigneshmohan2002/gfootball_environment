from expected_goals import to_normalized_space
import numpy as np
from math import floor

def calculate_threat_from_pass_points(pass_played_frame, pass_received_frame):
    pass_origin = to_normalized_space(pass_origin)
    pass_target = to_normalized_space(pass_target)

    xT = np.load('gfootball/env/reward_helpers/xT.npy')

    # xT is a 2D array, (16,12), need to get the appropriate x and y indices from the pass_origin and pass_target
    x_origin = floor(pass_origin[0]*16)
    y_origin = floor(pass_origin[1]*12)

    x_target = floor(pass_target[0]*16)
    y_target = floor(pass_target[1]*12)

    origin_xT = xT[x_origin, y_origin]
    target_xT = xT[x_target, y_target]

    return target_xT - origin_xT