#!/bin/bash

python3 -u -m gfootball.examples.run_dqn \
  --level 11_vs_11_hard_stochastic \
  --reward_experiment scoring \
  "$@"

