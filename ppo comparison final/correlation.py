import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

filename = "./11_vs_11_easy_stochastic_custom_output.txt"

score_pattern = r"score:\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"

gd_list = []
output_file = open(filename)
lines = output_file.readlines()

for line in lines:
    matches = re.findall(score_pattern, line)
    if matches:
        # Get the difference in goals
        goals = re.findall(r"-?\d+", matches[0])
        gd_list.append(int(goals[0]) - int(goals[1]))

weight_xT = 0.05  # Given per pass
weight_EPV = 0.002  # This is a very small value as EPV is given frequently
weight_xG = 0.1  # Given per shot
weight_pitch_control = 0.002  # This is a very small value as PC is given frequently

shot_counts = []
shot_count_pattern = r"Shot count:\s*(\d+)"

xT_values = []
xT_value_pattern = r"xT:\s*([\d.]+)"
xT_factor = 1 / weight_xT

EPV_values = []
EPV_pattern = r"EPV:\s*([\d.]+)"
EPV_factor = 1 / weight_EPV

xG_values = []
xG_pattern = r"xG:\s*([-]?\d+(\.\d+)?)"  # xG can be negative
xG_factor = 1 / weight_xG

pc_values = []
pc_pattern = r"pitch control:\s*([\d.]+)"
pc_factor = 1 / weight_pitch_control

possession_values = []
possession_pattern = r"Possession:\s*([\d.]+)"

opponent_possession_values = []
opponent_possession_pattern = r"Opponent possession:\s*([\d.]+)"


def get_values(filename, pattern, values_list, factor=1.0):
    output_file = open(filename)
    lines = output_file.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            if type(matches[0]) == tuple:
                values_list.append(float(matches[0][0]) * factor)
                continue
            values_list.append(float(matches[0]) * factor)


get_values(filename, shot_count_pattern, shot_counts)
get_values(filename, xT_value_pattern, xT_values, factor=xT_factor)
get_values(filename, EPV_pattern, EPV_values, factor=EPV_factor)
get_values(filename, xG_pattern, xG_values, factor=xG_factor)
get_values(filename, pc_pattern, pc_values, factor=pc_factor)
get_values(filename, possession_pattern, possession_values)
get_values(filename, opponent_possession_pattern, opponent_possession_values)

statistics = [
    shot_counts,
    xT_values,
    EPV_values,
    xG_values,
    pc_values,
    gd_list,
    possession_values,
    opponent_possession_values,
]
stat_names = {
    0: "Shot count",
    1: "xT",
    2: "EPV",
    3: "xG",
    4: "pc",
    5: "Goal Difference",
    6: "Possession",
    7: "Opp-possession",
}
# Calculate Pearson correlation coefficient for each pair of statistics
correlation_matrix = np.zeros((len(statistics), len(statistics)))
for i in range(len(statistics)):
    for j in range(len(statistics)):
        correlation_matrix[i, j] = pearsonr(statistics[i], statistics[j])[0]

# Create the figure and axis
fig, ax = plt.subplots()

# Display the correlation matrix as a color-coded table
cax = ax.imshow(
    correlation_matrix, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1
)

# Add color bar
plt.colorbar(cax)


# Set the labels
x_label_list = [stat_names[i] for i in range(len(statistics))]
y_label_list = [stat_names[i] for i in range(len(statistics))]
ax.set_xticks(np.arange(len(statistics)))
ax.set_yticks(np.arange(len(statistics)))
ax.set_xticklabels(x_label_list, rotation=45, ha="right")
ax.set_yticklabels(x_label_list, rotation=0)

# Loop over data dimensions and create text annotations.
for i in range(len(statistics)):
    for j in range(len(statistics)):
        text = ax.text(
            j,
            i,
            "{:.2f}".format(correlation_matrix[i, j]),
            ha="center",
            va="center",
            color="w",
        )

# Set title and show plot
ax.set_title("Correlation Matrix - 'Custom' reward")

weak_correlations = []
moderate_correlations = []
strong_correlations = []
for i in range(len(statistics)):
    for j in range(len(statistics)):
        if i == j:
            continue
        if abs(correlation_matrix[i, j]) > 0.5:
            strong_correlations.append(
                (stat_names[i], stat_names[j], correlation_matrix[i, j])
            )
        elif abs(correlation_matrix[i, j]) > 0.3:
            moderate_correlations.append(
                (stat_names[i], stat_names[j], correlation_matrix[i, j])
            )
        else:
            weak_correlations.append(
                (stat_names[i], stat_names[j], correlation_matrix[i, j])
            )

print("Strong correlations:")
for i in strong_correlations:
    print(i)
print("Moderate correlations:")
for i in moderate_correlations:
    print(i)
print("Weak correlations:")
for i in weak_correlations:
    print(i)
plt.show()
