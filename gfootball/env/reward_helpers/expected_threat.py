from expected_goals import to_normalized_space

def calculate_threat_from_pass_points(pass_origin, pass_target):
    pass_origin = to_normalized_space(pass_origin)
    pass_target = to_normalized_space(pass_target)

    # Need to know what side we are scoring in

    