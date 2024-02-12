import pandas as pd
import numpy as np


def to_normalized_space(p):
    # This function converts the position from the original space to the normalized space
    # In the normalized space, the centre of the field is [0.5, 0.5] and the top left corner is [0, 0] and the bottom right corner is [1, 1]
    xn = (p[0] + 1) / 2
    yn = (p[1] + 0.42) / 0.84
    return xn, yn


def calculate_xG(sh):
    model_variables = ["Angle", "Distance", "X", "C", "X2", "C2", "AX"]
    b = pd.read_csv("gfootball/env/reward_helpers/xg_model_params.csv")
    bsum = b[0]
    for i, v in enumerate(model_variables):
        bsum = bsum + b[i + 1] * sh[v]

    xG = 1 / (1 + np.exp(bsum))
    return xG


def get_xg_from_normalized_point(point):

    point = to_normalized_space(point)

    point = point[0] + 0.5, -point[1] + 0.5

    point = min(1, max(0, point[0])), min(1, max(0, point[1]))

    x = point[0] * 105
    y = point[1] * 68

    a = np.arctan(7.32 * x / (x**2 + abs(y - 68 / 2) ** 2 - (7.32 / 2) ** 2))

    xg_p = calculate_xG(
        {
            "X": x,
            "Y": y,
            "Distance": np.sqrt(x**2 + abs(y - 68 / 2) ** 2),
            "Angle": a,
            "C": abs(y - 68 / 2),
            "X2": x**2,
            "C2": (y - 68 / 2) ** 2,
            "AX": x * a,
        }
    )
    return xg_p
