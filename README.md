This zip file contains the code provided by the Google Research Football team as well as fles written by me for various parts of the experiment.
As a result of my code being an adaptation on the GRF code, I have included the GRF code in this zip file as well. My code aims to adapt the reward function of the GFootball environment as well as add files which are repsonsible for the training.

To run this code:
Download the zip file and extract it to a folder.
Build the docker image by running the appopriate command in the terminal.
Run the docker container by running the appropriate command in the terminal.
The requisite docker commands are not listed as they are system dependent. In certain cases you may need to change certain flags to ensure that the docker container is run correctly and is able to use the host machine's GPU in training.
This code was run on a machine with an NVIDIA RTX 3070 Super GPU and an AMD Ryzen 5 CPU, and Ubuntu 20.04 LTS as the operating system. It is thus optimized for this setup and may require changes to run on other systems.
The code was run on the available setup using the following commands:
sudo docker build —build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . t gfootball
sudo docker run --runtime=nvidia --device /dev/dri/card0 -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix:rw gfootball bash
Note that the nvidia runtime is used to ensure that the GPU is used for training, the arguments for the flags are system dependent and may need to be changed to run on other systems.
The below commands run shell scripts which train the agent using the PPO2 and DQN algorithms respectively. The shell scripts are located in the `examples` folder.
./gfootball/examples/scoring_easy_custom_dqn.sh
./gfootball/examples/scoring_easy_custom_ppo2.sh

To run multi-agent training, the files can be found in the xuance_training folder. They haven't been run yet due to time and computational constraints. 