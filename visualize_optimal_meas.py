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


def get_data(model_base_folder, human_score_base_folder, speeds, human_score_file="human_score.csv", model_perf_file="model_perf.csv"):
    # Get human data
    human_score = {}
    for s in speeds:
        with open(f"{human_score_base_folder}/{s}px/{human_score_file}", 'r') as f:
            _ = f.readline()
            for line in f.readlines():
                scene_id, score, = line.split(sep=',')
                human_score[int(scene_id)] = human_score.get(int(scene_id), [])
                human_score[int(scene_id)].append(float(score))

    # Get model data
    model_score = {}
    for s in speeds:
        with open(f"{model_base_folder}/{s}px/{model_perf_file}", 'r') as f:
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

    # Compute the average absolute difference for each point
    for meas_noise, process_noise in model_score:
        x_values.append(meas_noise)
        y_values.append(process_noise)
        val = compute_average_diff(
            human_score, model_score[(meas_noise, process_noise)])
        z_values.append(val)
        #print(meas_noise, process_noise, val)

    return x_values, y_values, z_values


if __name__ == '__main__':
    model_base_folder = "trials/scene55_higher_res"
    human_score_base_folder = "trials/scene55"
    human_score_file = "human_score.csv"
    model_perf_file = "model_perf.csv"
    annot = " (5 targets, 5 distractors)"

    speeds = [1.0, 2.5, 4.0]
    x_values, y_values, z_values = get_data(
        model_base_folder, human_score_base_folder, speeds, human_score_file, model_perf_file)

    ax = plt.gca()
    plt.plot(x_values, z_values, '-o')
    ax.set_xlabel("Measurement noise")
    ax.set_ylabel("Absolute accuracy difference")
    ax.set_title(
        f"Average absolute accuracy difference\nbetween a human subject and the model{annot}")
    plt.show()
