#!/bin/bash

python3 -u -m gfootball.examples.run_dqn scoring 2>&1 | tee 11_vs_11_easy_stochastic_dqn_output.txt
python3 -u -m gfootball.examples.run_dqn custom 2>&1 | tee 11_vs_11_easy_stochastic_custom_dqn_output.txt
