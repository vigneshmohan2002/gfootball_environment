import numpy as np
import math

distance_shot_threshold = 14
field_size = np.array([106, 68])
target_positions = np.array[
    (0.08333333333333333, 0.125),
    (0.08333333333333333, 0.375),
    (0.08333333333333333, 0.625),
    (0.08333333333333333, 0.875),
    (0.25, 0.125),
    (0.25, 0.375),
    (0.25, 0.625),
    (0.25, 0.875),
    (0.41666666666666663, 0.125),
    (0.41666666666666663, 0.375),
    (0.41666666666666663, 0.625),
    (0.41666666666666663, 0.875),
    (0.5833333333333333, 0.125),
    (0.5833333333333333, 0.375),
    (0.5833333333333333, 0.625),
    (0.5833333333333333, 0.875),
    (0.75, 0.125),
    (0.75, 0.375),
    (0.75, 0.625),
    (0.75, 0.875),
    (0.9166666666666667, 0.125),
    (0.9166666666666667, 0.375),
    (0.9166666666666667, 0.625),
    (0.9166666666666667, 0.875),
]

x_axes = np.array(
    [[0, 0], [1 / 6, 0], [2 / 6, 0], [3 / 6, 0], [4 / 6, 0], [5 / 6, 0], [1, 0]]
)
y_axes = np.array([[0, 0], [0, 1 / 4], [0, 2 / 4], [0, 3 / 4], [0, 1]])


def position_to_reward_factor(position):
    """
    This function takes the position of the player and returns the reward factor according to the position
    We are going to 'colour' the field according to the pitch control value we wish to assign
    The colours are used to group areas of the field together and assign the same reward factor to them
    @args:
    position: tuple, the position of the player in the normalized space
    @returns:
    reward_factor: float, the reward factor assigned to the position
    """
    colour_reward_map = {
        "green": 0,
        "blue": 0,
        "yellow": 0,
        "pink": 0,
        "purple": 0,
        "orange": 0,
        "red": 0,
    }
    x = position[0]
    y = position[1]
    if y < 0.25 or y > 0.75:
        if x < 0.16:
            return colour_reward_map["green"]
        elif x < 0.33:
            return colour_reward_map["blue"]
        elif x < 0.66:
            return colour_reward_map["yellow"]
        elif x < 0.83:
            return colour_reward_map["pink"]
        elif x <= 1:
            return colour_reward_map["orange"]
    else:
        if x < 0.33:
            return colour_reward_map["green"]
        elif x < 0.66:
            return colour_reward_map["blue"]
        elif x <= 1:
            return colour_reward_map["purple"]


def to_normalized_space_for_players(players):
    # This function converts the position of the player from the original space to the normalized space
    normalized_players = np.zeros_like(players)
    normalized_players[:, 0] = (players[:, 0] + 1) / 2
    normalized_players[:, 1] = (players[:, 1] + 0.42) / 0.84
    return normalized_players


# Since we target areas of the pitch rather than players we should exclude offside areas from the reward.
def get_onside_positions(
    controlled_player_pos_normalized,
    players_pos_right_normalized,
    target_positions,
):

    th = 1 / 105
    onside_positions = []
    for target_position in target_positions:
        target_position_x = target_position[0]
        passer_x = controlled_player_pos_normalized[0]
        last_defender_x = np.max(
            players_pos_right_normalized[1:, 0]
        )  # Take the maximum x position of the defending team except the goalkeeper

        # If player is in own field i.e behind the halfway line thus offside won't apply
        if target_position_x <= 0:
            onside_positions.append(target_position)
            continue

        # If last defender is between the passer and reciever then it's offside, so continue
        if target_position_x + th > last_defender_x and passer_x < last_defender_x:
            continue

    return onside_positions


def calculate_pitch_control_at_target(
    target_position,
    attacking_players,
    defending_players,
    ball_start_pos,
    params,
    field_size=[106, 68],
):

    if (
        target_position[0] < -field_size[0] / 2
        or target_position[0] > field_size[0] / 2
        or target_position[1] < -field_size[1] / 2
        or target_position[1] > field_size[1] / 2
    ):
        return 0.0, 1.0

    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(
        np.isnan(ball_start_pos)
    ):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = (
            np.linalg.norm(target_position - ball_start_pos)
            / params["average_ball_speed"]
        )

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin(
        [p.simple_time_to_intercept(target_position) for p in attacking_players]
    )
    tau_min_def = np.nanmin(
        [p.simple_time_to_intercept(target_position) for p in defending_players]
    )

    # check whether we actually need to solve equation 3
    if (
        tau_min_att - max(ball_travel_time, tau_min_def)
        >= params["time_to_control_def"]
    ):
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0.0, 1.0
    elif (
        tau_min_def - max(ball_travel_time, tau_min_att)
        >= params["time_to_control_att"]
    ):
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1.0, 0.0
    else:
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [
            p
            for p in attacking_players
            if p.time_to_intercept - tau_min_att < params["time_to_control_att"]
        ]
        defending_players = [
            p
            for p in defending_players
            if p.time_to_intercept - tau_min_def < params["time_to_control_def"]
        ]
        # set up integration arrays
        dT_array = np.arange(
            ball_travel_time - params["int_dt"],
            ball_travel_time + params["max_int_time"],
            params["int_dt"],
        )
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1 - ptot > params["model_converge_tol"] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (
                    (1 - PPCFatt[i - 1] - PPCFdef[i - 1])
                    * player.probability_intercept_ball(T)
                    * player.lambda_att
                )
                # make sure it's greater than zero
                assert (
                    dPPCFdT >= 0
                ), "Invalid attacking player probability (calculate_pitch_control_at_target)"
                player.PPCF += (
                    dPPCFdT * params["int_dt"]
                )  # total contribution from individual player
                PPCFatt[
                    i
                ] += (
                    player.PPCF
                )  # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (
                    (1 - PPCFatt[i - 1] - PPCFdef[i - 1])
                    * player.probability_intercept_ball(T)
                    * player.lambda_def
                )
                # make sure it's greater than zero
                assert (
                    dPPCFdT >= 0
                ), "Invalid defending player probability (calculate_pitch_control_at_target)"
                player.PPCF += (
                    dPPCFdT * params["int_dt"]
                )  # total contribution from individual player
                PPCFdef[
                    i
                ] += player.PPCF  # add to sum over players in the defending team
            ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
            i += 1
        if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))
        return PPCFatt[i - 1], PPCFdef[i - 1]


def default_model_params(time_to_control_veto=3):

    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params["max_player_speed"] = 5.0  # maximum player speed m/s
    params["reaction_time"] = (
        0.7  # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    )
    params["tti_sigma"] = (
        0.45  # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    )
    params["kappa_def"] = (
        1.0  # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    )
    params["lambda_att"] = 4.3  # ball control parameter for attacking team
    params["lambda_def"] = (
        4.3 * params["kappa_def"]
    )  # ball control parameter for defending team
    params["lambda_gk"] = (
        params["lambda_def"] * 3.0
    )  # make goal keepers must quicker to control ball (because they can catch it)
    params["average_ball_speed"] = 15.0  # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params["int_dt"] = 0.04  # integration timestep (dt)
    params["max_int_time"] = 10  # upper limit on integral time
    params["model_converge_tol"] = (
        0.01  # assume convergence when PPCF>0.99 at a given location.
    )
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start.
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params["time_to_control_att"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )
    print(params)
    return params


class Player(object):
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self, pid, pos, teamname, params, GKid):
        self.id = pid
        self.is_gk = self.id == GKid
        self.teamname = teamname
        self.playername = "%s_%s_" % (teamname, pid)
        self.vmax = params[
            "max_player_speed"
        ]  # player max speed in m/s. Could be individualised
        self.reaction_time = params[
            "reaction_time"
        ]  # player reaction time in 's'. Could be individualised
        self.tti_sigma = params[
            "tti_sigma"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params[
            "lambda_att"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = (
            params["lambda_gk"] if self.is_gk else params["lambda_def"]
        )  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.position = pos
        self.PPCF = 0.0  # initialise this for later

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.0
        r_reaction = self.position
        self.time_to_intercept = (
            self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax
        )
        return self.time_to_intercept

    def probability_intercept_ball(self, T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1 / (
            1.0
            + np.exp(
                -np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)
            )
        )
        return f


def observation_to_pitch_control_reward(obs):

    # Getting positions and directions of players and ball
    left_team_positions = np.array(obs["left_team"]).reshape((-1, 2))

    right_team_positions = np.array(obs["right_team"]).reshape((-1, 2))

    ball_position = np.array([obs["ball"][0], obs["ball"][1]])

    # Getting the controlled player and team
    controlled_player_id = obs["active"]
    controlled_team_id = obs["ball_owned_team"]  # + 1 (Original code)

    # Normalize players and balls position and directions
    normalized_left_team_positions = to_normalized_space_for_players(
        left_team_positions
    )
    normalized_right_team_positions = to_normalized_space_for_players(
        right_team_positions
    )
    ball_pos_normalized = to_normalized_space_for_players(
        ball_position.reshape((-1, 2))
    )

    # Further pre processing to work with the pitch control model
    def to_metric_space(positions):
        positions[:, 0] -= 0.5
        positions[:, 1] = 1 - positions[:, 1] - 0.5
        return positions

    normalized_left_team_positions = to_metric_space(normalized_left_team_positions)

    normalized_right_team_positions = to_metric_space(normalized_right_team_positions)

    ball_pos_normalized = to_metric_space(ball_pos_normalized)

    controlled_player_pos_normalized = left_team_positions[controlled_player_id]

    params = default_model_params()

    # In a real-world scenario we consider velocity of the players and the ball
    # However for the sake of similicty and due to the realism lost in the environment, the effect of velocity in the model is negligible
    left_team_players = []
    right_team_players = []
    for idx, p in enumerate(normalized_left_team_positions):
        left_team_players.append(Player(idx, p * field_size, "Home", params, 0))

    for idx, p in enumerate(normalized_right_team_positions):
        right_team_players.append(Player(idx, p * field_size, "Away", params, 0))

    onside_positions = get_onside_positions(
        controlled_player_pos_normalized,
        normalized_right_team_positions,
        target_positions,
    )

    onside_positions = to_metric_space(onside_positions)
    # Classify left and right team into attacking and defending team based on possession of the ball
    attacking_players, defending_players, possession = (
        (left_team_players, right_team_players, True)
        if controlled_team_id == 0
        else (right_team_players, left_team_players, False)
    )

    reward = 0
    for target in onside_positions:
        ppcf_att, _ = calculate_pitch_control_at_target(
            target,
            attacking_players,
            defending_players,
            ball_pos_normalized.flatten(),
            params,
        )
        reward += ppcf_att * position_to_reward_factor(target)

    if possession:
        return reward
    return -reward


if __name__ == "__main__":
    x_axes = np.array(
        [[0, 0], [1 / 6, 0], [2 / 6, 0], [3 / 6, 0], [4 / 6, 0], [5 / 6, 0], [1, 0]]
    )
    y_axes = np.array([[0, 0], [0, 1 / 4], [0, 2 / 4], [0, 3 / 4], [0, 1]])
    rectangle_centers = []
    for i in range(len(x_axes) - 1):
        for j in range(len(y_axes) - 1):
            x_center = (x_axes[i][0] + x_axes[i + 1][0]) / 2
            y_center = (y_axes[j][1] + y_axes[j + 1][1]) / 2
            rectangle_centers.append((x_center, y_center))
    print(len(rectangle_centers))
    print(rectangle_centers)
