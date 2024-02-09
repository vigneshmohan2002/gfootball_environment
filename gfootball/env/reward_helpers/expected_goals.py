def to_normalized_space(p):
    # This function converts the position from the original space to the normalized space
    # In the normalized space, the centre of the field is [0.5, 0.5] and the top left corner is [0, 0] and the bottom right corner is [1, 1]
    xn = (p[0] + 1) / 2
    yn = (p[1] + 0.42) / 0.84
    return xn, yn


def get_xg_from_normalized_point(p, epv):

    p = to_normalized_space(p)

    p = p[0] + 0.5, -p[1] + 0.5

    p = min(1, max(0, p[0])), min(1, max(0, p[1]))
    
    xg_p = 0
    # if 0 <= pos_x < epv.shape[1] and 0 <= pos_y < epv.shape[0]:
    #     xg_p = epv[pos_y, pos_x]

    return xg_p
