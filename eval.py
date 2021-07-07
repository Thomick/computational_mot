import numpy as np
import matplotlib.pyplot as plt
from cal_mot_kalman import MotKalman
from make_scenes import Trajectory, OcclusionSettings, ColorSettings, Scene, Objs, Canvas


def error_sum(gt_pos, est_pos, frame_range):
    s = 0
    for f in range(frame_range[0], frame_range[1]):
        for i in range(est_pos.shape[0]):
            s += np.linalg.norm(np.array(est_pos[i]
                                [f])-np.array(gt_pos[f][i]))
    return s


def error_max(gt_pos, est_pos, frame_range):
    s = 0
    for f in range(frame_range[0], frame_range[1]):
        for i in range(est_pos.shape[0]):
            s = max(s, np.linalg.norm(np.array(est_pos[i]
                                               [f])-np.array(gt_pos[f][i])))
    return s


def attention_window_score(gt_pos, est_pos, frame, threshold):
    s = 0
    for i in range(est_pos.shape[0]):
        if np.min([np.linalg.norm(np.array(est_pos[j][frame])-np.array(gt_pos[frame][i])) for j in range(len(est_pos))]) < threshold:
            s += 1

    return s/(est_pos.shape[0])


def visualize_performance(data, metric="cumulated_error", bar_chart=True, save_flag=True, x_axis_label="", y_axis_label="", threshold=20):
    # data : list of dictionaries
    # metric : cumulated_error,max_error, last_frame_error, attention_window_score
    labels = []
    values = []
    for d in data:
        print(d["label"])
        labels.append(d["label"])
        if metric == "cumulated_error":
            values.append(error_sum(d["gt_pos"], d["est_pos"],
                          [0, d["est_pos"].shape[1]]))
        elif metric == "max_error":
            values.append(error_max(d["gt_pos"], d["est_pos"],
                          [0, d["est_pos"].shape[1]]))
        elif metric == "last_frame_error":
            values.append(error_sum(d["gt_pos"], d["est_pos"], [
                          d["est_pos"].shape[1]-1, d["est_pos"].shape[1]]))
        elif metric == "attention_window_score":
            values.append(attention_window_score(
                d["gt_pos"], d["est_pos"], d["est_pos"].shape[1]-1, threshold))

    ax = plt.gca()
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    if bar_chart:
        ax.bar(labels, values)
    else:
        ax.plot(labels, values)
    plt.tight_layout()
    if save_flag:
        plt.savefig("last_plot.png")
    plt.show()


def visualize_performance_multiple_runs(parameters, num_runs, metric="cumulated_error", bar_chart=True, save_flag=True, x_axis_label="", y_axis_label="", threshold=20, title=""):
    # parameters : list of dictionaries
    # metric : cumulated_error,max_error, last_frame_error, attention_window_score
    labels = []
    means = []
    med = []
    stds = []
    for d in parameters:
        print(d["label"])
        labels.append(d["label"])
        values = []
        for _ in range(num_runs):
            class_scene = Scene(d["class_canvas"], d["class_baseobj"],
                                d["max_frames"], d["num_obj"], d["occlusion_settings"], d["std_measure"], render=False)
            class_scene.update_scenes_all()
            # Kalman parameters
            class_motkalman = MotKalman(class_scene, num_targets=d["num_targets"],
                                        dt=d["dt"], std_measure=d["std_measure"], std_pred=d["std_pred"], covP=d["covP"])
            gt_pos = class_scene.get_gt_pos()
            est_pos = class_motkalman.est_pos

            if metric == "cumulated_error":
                values.append(error_sum(gt_pos, est_pos,
                                        [0, est_pos.shape[1]]))
            elif metric == "max_error":
                values.append(error_max(gt_pos, est_pos,
                                        [0, est_pos.shape[1]]))
            elif metric == "last_frame_error":
                values.append(error_sum(gt_pos, est_pos, [
                    est_pos.shape[1]-1, est_pos.shape[1]]))
            elif metric == "attention_window_score":
                values.append(attention_window_score(
                    gt_pos, est_pos, est_pos.shape[1] - 1, threshold))
        means.append(np.mean(values))
        med.append(np.median(values))
        stds.append(np.std(values))

    ax = plt.gca()
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)
    if bar_chart:
        ax.bar(labels, means, yerr=stds, align='center',
               alpha=0.5, ecolor='black', capsize=10)
    else:
        ax.errorbar(labels, means, yerr=stds)
    plt.tight_layout()
    if save_flag:
        plt.savefig(title + ".png")
    plt.show()


if __name__ == '__main__':
    parameters = []

    num_obj = 8
    canvas_x = 256
    canvas_y = 256
    max_frames = 500
    diameter = 20
    trajectory_type = "bouncing"  # "bouncing" or "mean-reverting"
    speed_range = [7, 7]  # in pixels per time
    speedvar_prob = 0.
    speedvar_std = 2  # in pixels per time
    directionvar_prob = 0.
    directionvar_std = 15  # in degrees
    inertia_param = 0.01  # between 0 and 1
    accelnoise_std = 0
    spring_constant = 0.01  # > 0
    occlusion_settings = OcclusionSettings(
        -300, -320, [0, 0, canvas_x, canvas_y])
    color_settings = ColorSettings(color_type="white", init_hue_range=[0, 1],
                                   hue_drift_range=[0.01, 0.01])

    # Kalman parameters
    dt = 1
    std_pred = 0.3
    std_measure = 0.3
    num_targets = 4
    covP = 200

    for s in range(0, 11):
        params = {}
        class_trajectory = Trajectory(trajectory_type, speed_range, speedvar_prob, speedvar_std,
                                      directionvar_prob, directionvar_std, inertia_param, accelnoise_std, spring_constant)
        class_canvas = Canvas(canvas_x, canvas_y)
        class_baseobj = Objs(class_canvas, diameter,
                             class_trajectory, color_settings)
        params["max_frames"] = max_frames
        params["std_measure"] = std_measure
        params["num_obj"] = num_targets + s
        params["occlusion_settings"] = occlusion_settings
        params["dt"] = dt
        params["std_pred"] = std_pred
        params["std_measure"] = std_measure
        params["num_targets"] = num_targets
        params["covP"] = covP
        params["class_trajectory"] = class_trajectory
        params["class_canvas"] = class_canvas
        params["class_baseobj"] = class_baseobj
        params["label"] = s
        parameters.append(params)

    visualize_performance_multiple_runs(
        parameters, 10, metric="attention_window_score", bar_chart=False, save_flag=True,
        y_axis_label="Correctness\n(Tracked target / Total number of targets)", x_axis_label="Standard deviation of measure noise", title="Kalman filter model performance\nas a function of standard deviation of measure noise")
