# Plot average across all speeds as a function of 2 parameters
# Needs the right folder structure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def compute_average_diff(human_perf, model_perf, print_all=False):
    diffs = []
    for scene, scores in model_perf.items():
        for i in range(len(scores)):
            if human_perf.get(scene) != None:
                diffs.append(abs(scores[i] - human_perf[scene][i]))
                if print_all:
                    print((scores[i] - human_perf[scene][i]))
    return np.mean(diffs)


def get_data(base_folder, speeds, human_score_file="human_score.csv", model_perf_file="model_perf.csv"):
    human_score = {}
    for s in speeds:
        with open(f"{base_folder}/{s}px/{human_score_file}", 'r') as f:
            _ = f.readline()
            for line in f.readlines():
                scene_id, score, = line.split(sep=',')
                human_score[int(scene_id)] = human_score.get(int(scene_id), [])
                human_score[int(scene_id)].append(float(score))

    model_score = {}
    for s in speeds:
        with open(f"{base_folder}/{s}px/{model_perf_file}", 'r') as f:
            _ = f.readline()
            for line in f.readlines():
                meas_noise, process_noise, scene_id, score, = line.split(
                    sep=',')
                meas_noise = float(meas_noise)
                process_noise = float(process_noise)
                scene_id = int(scene_id)
                score = float(score)
                model_score[(meas_noise, process_noise)] = model_score.get(
                    (meas_noise, process_noise), {})
                model_score[(meas_noise, process_noise)][scene_id] = model_score[(meas_noise, process_noise)].get(
                    scene_id, [])
                model_score[(meas_noise, process_noise)
                            ][scene_id].append(score)

    x_values = []
    y_values = []
    z_values = []

    for meas_noise, process_noise in model_score:
        x_values.append(meas_noise)
        y_values.append(process_noise)
        val = compute_average_diff(
            human_score, model_score[(meas_noise, process_noise)])
        z_values.append(val)
        #print(meas_noise, process_noise, val)

    return x_values, y_values, z_values


def plot_flat(x_values, y_values, z_values, size_x, size_y, annot):
    value_mesh = np.array(z_values).reshape((size_y, size_x))
    fig = plt.figure()
    ax = fig.gca()

    im = plt.pcolormesh(x_values[:size_x], y_values[::size_x],
                        value_mesh, cmap="viridis", shading='auto')
    ax.set_title(
        f"Average absolute accuracy difference\nbetween a human subject and the model\nacross all speeds{annot}")
    fig.colorbar(im, shrink=0.5, aspect=5)
    # plt.clim(0.05, 0.45)
    ax.set_ylabel("Process noise")
    ax.set_xlabel("Measurement noise")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_folder = "scene44"
    human_score_file = "human_score.csv"
    model_perf_file = "model_perf.csv"
    annot = " (4 targets, 4 distractors)"
    size_x = 9
    size_y = 8

    speeds = [1.0, 2.5, 4.0, 5.5, 7.0]
    x_values, y_values, z_values = get_data(
        base_folder, speeds, human_score_file, model_perf_file)

    plot_flat(x_values, y_values, z_values, size_x, size_y, annot)
