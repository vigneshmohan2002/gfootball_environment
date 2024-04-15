This zip file contains the code provided by the Google Research Football team as well as fles written by me for various parts of the experiment.
As a result of my code being an adaptation on the GRF code, I have included the GRF code in this zip file as well. My code aims to adapt the reward function of the GFootball environment as well as add files which are repsonsible for the training.

To run this code:
Download the zip file and extract it to a folder.
Build the docker image by running the appopriate command in the terminal.
Run the docker container by running the appropriate command in the terminal.
The requisite docker commands are system dependent. In certain cases you may need to change certain flags to ensure that the docker container is run correctly and is able to use the host machine's GPU in training.
This code was run on a machine with an NVIDIA RTX 3070 Super GPU and an AMD Ryzen 5 CPU, and Ubuntu 20.04 LTS as the operating system. It is thus optimized for this setup and may require changes to run on other systems.
Run the training script which can be found in the `examples` folder and the `xuance_training` folder.
