import typing
from typing import Tuple
from expected_goals import to_normalized_space
import numpy as np
from math import floor


def get_epv_for_ball_position(ball_position: Tuple[float, float]) -> float:
    """Returns the expected possession value (EPV) for a given ball position."""
    ball_position = to_normalized_space(ball_position)
    EPV = np.loadtxt("epv_grid.csv", delimiter=",")

    ny, nx = EPV.shape

    x_origin = floor(ball_position[0] * nx)
    y_origin = floor(ball_position[1] * ny)

    return EPV[x_origin, y_origin]
