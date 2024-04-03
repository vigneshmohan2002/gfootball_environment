#!/bin/bash

python3 -u -m gfootball.examples.run_ppo2 scoring 2>&1 | tee 11_vs_11_easy_stochastic_output.txt
python3 -u -m gfootball.examples.run_ppo2 custom 2>&1 | tee 11_vs_11_easy_stochastic_custom_output.txt

