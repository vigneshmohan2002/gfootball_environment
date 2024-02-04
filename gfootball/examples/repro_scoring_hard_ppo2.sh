#!/bin/bash

python3 -u -m gfootball.examples.run_ppo2 \
  --level 11_vs_11_hard_stochastic \
  --reward_experiment scoring \
  --policy gfootball_impala_cnn \
  --cliprange 0.115 \
  --gamma 0.997 \
  --ent_coef 0.00155 \
  --num_timesteps 50000000 \
  --max_grad_norm 0.76 \
  --lr 0.00011879 \
  --num_envs 16 \
  --noptepochs 2 \
  --nminibatches 4 \
  --nsteps 512 \
  "$@"

# Good but unsettable defaults:
# Optimizer: adam
# Value-function coefficient is 0.5
# GAE (lam): 0.95
