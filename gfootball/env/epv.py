import typing
from typing import Tuple
from gfootball.env import expected_goals as xG
import numpy as np
from math import floor
import os


def get_epv_for_ball_position(ball_position: Tuple[float, float]) -> float:
    """Returns the expected possession value (EPV) for a given ball position."""
    ball_position = xG.to_normalized_space(ball_position)
    EPV = np.loadtxt("gfootball/env/EPV_grid.csv", delimiter=",")

    ny, nx = EPV.shape

    x = floor(ball_position[0] * nx)
    y = floor(ball_position[1] * ny)

    try:
        return EPV[x, y]
    except IndexError:
        return 0.0
