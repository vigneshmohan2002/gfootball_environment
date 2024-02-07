import numpy
import math
import os
import numpy as np
import math
import cv2

distance_shot_threshold = 14
field_size = numpy.array([106, 68])
wpc = 1
wepv = 0.3
wxg = 1
th_pass_d = 30  # meters
th_angle = 15  # degrees


#######################
# Utilities functions #
######################


def to_normalized_space(p):
    xn = (p[0] + 1) / 2
    yn = (p[1] + 0.42) / 0.84
    return xn, yn


def to_normalized_space_for_players(players):
    normalized_players = numpy.zeros_like(players)
    normalized_players[:, 0] = (players[:, 0] + 1) / 2
    normalized_players[:, 1] = (players[:, 1] + 0.42) / 0.84
    return normalized_players


# Consider changing this to a function that takes a list of players and returns only onside player, or can simply use a filter
# Offside detection to avoid pass
def offside(
    controlled_player_pos_normalized,
    player_pos_normalized,
    players_pos_right_normalized,
):

    th = 1 / 105
    reciever_x = player_pos_normalized[0]
    passer_x = controlled_player_pos_normalized[0]
    last_defender_x = numpy.max(
        players_pos_right_normalized[1:, 0]
    )  # Take the maximum x position of the defending team except the goalkeeper

    # if player is in own field i.e behind the halfway line thus offside won't apply
    if reciever_x <= 0:
        return False

    # If last defender is between the [asser and reciever, then offside
    if reciever_x + th > last_defender_x and passer_x < last_defender_x:
        return True

    return False


#######################
# Pitch Control Model #
######################


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
        self.velocity = numpy.array([0, 0])
        self.PPCF = 0.0  # initialise this for later

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.0  # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity * self.reaction_time
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


def observation_to_pitch_control(obs):

    # Getting positions and directions of players and ball
    left_team_positions = numpy.array(obs["left_team"]).reshape((-1, 2))
    left_team_directions = numpy.array(obs["left_team_direction"]).reshape((-1, 2))

    right_team_positions = numpy.array(obs["right_team"]).reshape((-1, 2))
    right_team_directions = numpy.array(obs["right_team_direction"]).reshape((-1, 2))

    ball_position = numpy.array([obs["ball"][0], obs["ball"][1]])
    ball_direction = numpy.array([obs["ball_direction"][0], obs["ball_direction"][1]])

    # Getting the controlled player and team
    controlled_player_id = obs["active"]
    controlled_team_id = obs["ball_owned_team"]  # + 1 (Original code)

    steps_left = obs["steps_left"]
    is_dribbling = False
    is_sprinting = False

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

    normalized_left_team_positions[:, 0] -= 0.5
    normalized_left_team_positions[:, 1] = (
        1 - normalized_left_team_positions[:, 1] - 0.5
    )

    normalized_right_team_positions[:, 0] -= 0.5
    normalized_right_team_positions[:, 1] = (
        1 - normalized_right_team_positions[:, 1] - 0.5
    )
    ball_pos_normalized[:, 0] -= 0.5
    ball_pos_normalized[:, 1] = 1 - ball_pos_normalized[:, 1] - 0.5

    left_team_directions[:, 1] = -left_team_directions[:, 1]
    players_dirs_left_normalized = left_team_directions / numpy.linalg.norm(
        left_team_directions, axis=1
    ).reshape((-1, 1))

    controlled_player_pos = left_team_positions[controlled_player_id]
    controlled_player_pos_normalized = left_team_positions[controlled_player_id]
    controlled_player_dir_normalized = players_dirs_left_normalized[
        controlled_player_id
    ]

    params = default_model_params()

    left_team_players = []
    right_team_players = []
    for idx, p in enumerate(normalized_left_team_positions):
        left_team_players.append(Player(idx, p * field_size, "Home", params, 0))

    for idx, p in enumerate(normalized_right_team_positions):
        right_team_players.append(Player(idx, p * field_size, "Away", params, 0))

    # TODO: generate 24 target positions on the field and calculate the pitch control at each of these positions
    target_position = []

    # Classify left and right team into attacking and defending team based on possession of the ball
    attacking_players, defending_players, possession = (
        (left_team_players, right_team_players, True)
        if controlled_team_id == 0
        else (right_team_players, left_team_players, False)
    )

    # TODO: Calculate pitch control at each target position
    # ppcf_att, ppcf_def = calculate_pitch_control_at_target(target, attacking_players, defending_players, ball_pos_normalized.flatten(), params)

    # Depending on possession, we choose to reward or punish the model
