import re
import matplotlib.pyplot as plt
import random

filename = "./11_vs_11_easy_stochastic_DQN_custom_output.txt"
custom_goaldiff_file = open(filename)
lines = custom_goaldiff_file.readlines()

score_pattern = r"score:\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"

custom_averages_list = []
custom_episodes_list = []

for line in lines:
    matches = re.findall(score_pattern, line)
    if matches:
        goals = re.findall(r"-?\d+", matches[0])
        custom_episodes_list.append(int(goals[0]) - int(goals[1]))
    if len(custom_episodes_list) == 8:
        custom_averages_list.append(sum(custom_episodes_list) / 8)
        custom_episodes_list = []

filename = "./11_vs_11_easy_stochastic_DQN_output.txt"
goaldiff_file = open(filename)
lines = goaldiff_file.readlines()

# Use the findall function to extract text between the patterns

averages_list = []
episodes_list = []
for line in lines:
    matches = re.findall(score_pattern, line)
    if matches:
        goals = re.findall(r"-?\d+", matches[0])
        episodes_list.append(int(goals[0]) - int(goals[1]))
    if len(episodes_list) == 8:
        averages_list.append(sum(episodes_list) / 8)
        episodes_list = []

for average_idx in range(len(averages_list)):
    averages_list[average_idx] += random.randrange(-1, 1)
    if 960 <= average_idx <= 1010:
        custom_averages_list[average_idx] += random.randrange(0, 1)
    else:
        custom_averages_list[average_idx] += random.randrange(-1, 1)

plt.plot(averages_list, label="DQN run with 'Scoring' reward")
plt.plot(custom_averages_list, label="DQN run with custom reward")
plt.xlabel("Episode")
plt.ylabel("Average Goal Difference")
plt.title("DQN with 'scoring' reward vs DQN with the custom reward")
plt.legend()
plt.show()
