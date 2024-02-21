from gfootball.env import football_action_set
import numpy as np
from reward_helpers.pitch_control import observation_to_pitch_control_reward
from reward_helpers.expected_goals import get_xg_from_game_obs_point
from reward_helpers.expected_threat import calculate_threat_from_pass_points

import deepcopy
from xuance.environment.football.gfootball_env import GFootball_Env


class CustomRewardGFootball_env(GFootball_Env):
    """A wrapper that adds a custom reward utilizing advanced football metrics
    Pitch Control, Expected Goals, Expected Possession Value and Expected Threat
    to the environment."""

    def __init__(self, env):
        super().__init__(self, env)
        self.pass_played_frame = None

    def _expected_threat_pass(self, pass_received_ball_pos):
        """
        Calculate the expected threat of a pass.
        Note: xT is a metric that can also evaluate the quality of a dribble/carry play.
        """

        pass_played_location = (
            self.pass_played_frame[0]["ball"][0],
            self.pass_played_frame[0]["ball"][1],
        )
        pass_played_location = (
            pass_received_ball_pos["ball"][0],
            pass_received_ball_pos["ball"][1],
        )
        xT = calculate_threat_from_pass_points(
            pass_played_location, pass_received_ball_pos
        )
        return xT

    def _expected_possession_value(self, ball_location):
        """
        Calculate the expected possession value solely based on the location of the ball.
        """
        return None

    def _expected_goals(self, ball_location):
        """
        Calculate the xG value of a shot.
        """
        shot_pos = ball_location[0], ball_location[1]
        return get_xg_from_game_obs_point(shot_pos)

    def _pitch_control(self, observation):
        """
        Calculate the pitch control of the team.
        """
        return observation_to_pitch_control_reward(observation)

    def _reward_fn(self, xT, EPV, xG, pitch_control):
        """This will be an aggregation of the stats calculated by the above functions."""
        weight_xT = 0.002  # Given per pass
        weight_EPV = 0.0016  # This is a very small value as EPV is given frequently
        weight_xG = 0.05  # Given per shot
        weight_pitch_control = (
            0.0016  # This is a very small value as PC is given frequently
        )
        add_reward = (
            weight_xT * xT
            + weight_EPV * EPV
            + weight_xG * xG
            + weight_pitch_control * pitch_control
        )
        return add_reward

    def step(self, actions):
        """One-step transition of the environment.

        Args:
            actions: the actions for all agents.
        """
        obs, reward, terminated, truncated, info = self.env.step(actions)
        reward_shaping_observation = deepcopy(obs)
        obs = obs.reshape([self.n_agents, -1])
        state = self.get_state()
        self._episode_step += 1
        self._episode_score += reward.mean()
        info["episode_step"] = self._episode_step
        info["episode_score"] = self._episode_score
        truncated = True if self._episode_step >= self.max_cycles else False

        xG = 0
        xT = 0
        epv = 0
        pitch_control = 0
        # If action is a pass save the frame
        pass_actions = [
            football_action_set.action_short_pass,
            football_action_set.action_long_pass,
            football_action_set.action_high_pass,
        ]
        ball_owned_player_action = actions[
            reward_shaping_observation["ball_owned_player"]
        ]  # Get the action of the player who owns the ball
        if ball_owned_player_action in pass_actions:
            self.pass_played_frame = (
                reward_shaping_observation,
                reward_shaping_observation["ball_owned_player"],
            )
        # Detect when the pass is received
        observation_bop = reward_shaping_observation["ball_owned_player"]
        # If the ball is owned by a different player than the one who played the pass, then the pass was received
        if self.pass_played_frame is not None and observation_bop not in [
            -1,
            self.pass_played_frame[1],
        ]:
            # We only need the x and y coordinates of the ball
            xT = self._expected_threat_pass(reward_shaping_observation["ball"])
            self.pass_played_frame = None

        # If action is a shot calculate the xG
        if ball_owned_player_action == football_action_set.action_shot:
            xG = self._expected_goals(reward_shaping_observation["ball"])

        # If frame_cnt is not available, 3000-steps_left is used as a proxy
        k = 10
        if reward_shaping_observation["frame_cnt"] % k == 0:
            pitch_control = self._pitch_control(reward_shaping_observation)
            epv = self._expected_possession_value(
                reward_shaping_observation["ball"][0],
                reward_shaping_observation["ball"][1],
            )

        reward += self._reward_fn(xT, epv, xG, pitch_control)

        return obs, state, reward, terminated, truncated, info
