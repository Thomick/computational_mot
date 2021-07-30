# Visualize data from human_model_comp.py
# Human accuracy data must be given (same folder) in order to compare it with the model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def compute_average_diff(human_perf, model_perf, print_all=False):
    total = 0
    nb_scene = 0
    for scene, score in model_perf.items():
        if human_perf.get(scene) != None:
            nb_scene += 1
            total += abs(score - human_perf[scene])
            if print_all:
                print((score - human_perf[scene]))
    return total / nb_scene


def compute_euclidean_distance(human_perf, model_perf):
    total = 0
    nb_scene = 0
    for scene, score in model_perf.items():
        if human_perf.get(scene) != None:
            nb_scene += 1
            total += (score - human_perf[scene])*(score - human_perf[scene])
    return total**0.5


def compute_mean_perf(perf):
    total = []
    for _, score in perf.items():
        total.append(score)
    return np.mean(total)


# Plot and folder parameters
folder = "trials/scene44/2.5px"
# Relevant information to add to the title
annot = "\n(2.5px/frame, 4 targets, 4 distractors)"
human_score_file = "human_score.csv"
model_perf_file = "model_perf.csv"
# Do not compare to human performance but plot average performance instead
only_draw_average = False
# Number of values used on each axis (Must be the same as the range used in human_model_comp.py)
size_x = 9
size_y = 8

# Get human data
human_score = {}
if not only_draw_average:
    with open(f"{folder}/{human_score_file}", 'r') as f:
        _ = f.readline()
        for line in f.readlines():
            scene_id, score, = line.split(sep=',')
            human_score[int(scene_id)] = float(score)

# Get model data
model_score = {}
with open(f"{folder}/{model_perf_file}", 'r') as f:
    _ = f.readline()
    for line in f.readlines():
        meas_noise, process_noise, scene_id, score, = line.split(sep=',')
        meas_noise = float(meas_noise)
        process_noise = float(process_noise)
        scene_id = int(scene_id)
        score = float(score)
        model_score[(meas_noise, process_noise)] = model_score.get(
            (meas_noise, process_noise), {})
        model_score[(meas_noise, process_noise)][scene_id] = score

x_values = []
y_values = []
z_values = []
means = []

# Compute the value to plot based on the data
for meas_noise, process_noise in model_score:
    x_values.append(meas_noise)
    y_values.append(process_noise)
    val = 0
    if not only_draw_average:
        val = compute_average_diff(
            human_score, model_score[(meas_noise, process_noise)])
    means.append(compute_mean_perf(model_score[(meas_noise, process_noise)]))
    z_values.append(val)
    print(meas_noise, process_noise, val)

# Resize values
value_mesh = np.array(z_values).reshape((size_y, size_x))
means_mesh = np.array(means).reshape((size_y, size_x))
print(value_mesh.shape)
fig = plt.figure()
ax = fig.gca()

if only_draw_average:
    im = plt.pcolormesh(x_values[:size_x], y_values[::size_x],
                        means_mesh, cmap="viridis", shading='auto')
    ax.set_title(
        f"Average accuracy of the model{annot}")
else:
    im = plt.pcolormesh(x_values[:size_x], y_values[::size_x],
                        value_mesh, cmap="viridis", shading='auto')
    ax.set_title(
        f"Average absolute accuracy difference\nbetween a human subject and the model{annot}")

fig.colorbar(im, shrink=0.5, aspect=5)
ax.set_ylabel("Process noise")
ax.set_xlabel("Measurement noise")
plt.tight_layout()

if not only_draw_average:  # Print average human score
    print(f"Human average score: {compute_mean_perf(human_score)}")

plt.show()
