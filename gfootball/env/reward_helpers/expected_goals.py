import pandas as pd
import numpy as np


def to_normalized_space(p):
    # This function converts the position from the original space to the normalized space
    # In the normalized space, the centre of the field is [0.5, 0.5] and the top left corner is [0, 0] and the bottom right corner is [1, 1]
    xn = (p[0] + 1) / 2
    yn = (p[1] + 0.42) / 0.84
    print("Normalized space point: ", xn, ", ", yn)
    return xn, yn


model_params = {
    "Intercept": -0.286152965059171,
    "Angle": -0.8018059793904666,
    "Distance": 0.31175546461618064,
    "X": -0.15816408816458669,
    "C": -0.002443541559558661,
    "X2": -0.0013774910615196073,
    "C2": -0.003384650940474773,
    "AX": 0.12075110246468955,
}


def calculate_xG(values):
    model_variables = ["Angle", "Distance", "X", "C", "X2", "C2", "AX"]
    bsum = model_params["Intercept"]
    for v in model_variables:
        bsum += model_params[v] * values[v]
    xG = 1 / (1 + np.exp(bsum))
    return xG


def get_xg_from_game_obs_point(point):

    point = to_normalized_space(point)

    x = point[0] * 105 if point[0] < 0.5 else (1 - point[0]) * 105
    y = point[1] * 68

    values = {}
    a = np.where(
        np.arctan(7.32 * x / (x**2 + (abs(y - 68 / 2)) ** 2 - (7.32 / 2) ** 2)) > 0,
        np.arctan(7.32 * x / (x**2 + (abs(y - 68 / 2)) ** 2 - (7.32 / 2) ** 2)),
        np.arctan(7.32 * x / (x**2 + (abs(y - 68 / 2)) ** 2 - (7.32 / 2) ** 2)) + np.pi,
    )
    if a < 0:
        a = np.pi + a
    values["Angle"] = a
    values["Distance"] = np.sqrt(x**2 + abs(y - 68 / 2) ** 2)
    values["D2"] = x**2 + abs(y - 68 / 2) ** 2
    values["X"] = x
    values["AX"] = x * a
    values["X2"] = x**2
    values["C"] = abs(y - 68 / 2)
    values["C2"] = (y - 68 / 2) ** 2

    return calculate_xG(values)
