# Evaluation script for a single parameter

import numpy as np
import matplotlib.pyplot as plt
from cal_mot_kalman import MotKalman
from cal_mot_kalman_accel import MotKalmanAccel
from cal_mot_kalman_meanreverting import MotKalmanMR
from cal_mot_kalman_bouncing import MotKalmanB
from make_scenes import Trajectory, OcclusionSettings, ColorSettings, Scene, Objs, Canvas
import csv


def count_target_switch(corresp_filter, scene, num_target, frame_range):
    s = 0
    for i in range(num_target):
        current_correspondence = i
        last_correspondence = i
        current_cor_duration = 1
        for f in range(max(frame_range[0], 1), frame_range[1]):
            cur = scene.correspondence[f][corresp_filter[f][i]]
            # print(cur)
            if cur != current_correspondence:
                current_cor_duration = 0
            current_cor_duration += 1
            current_correspondence = cur
            if current_cor_duration == 5 and current_correspondence != last_correspondence:
                s += 1
                last_correspondence = current_correspondence
        # print()
    return s


def count_approach(gt_pos, num_target, num_object, frame_range, threshold):
    s = 0
    for i in range(num_target):
        last_approach = []
        for f in range(frame_range[0], frame_range[1]):
            current_approach = []
            for obj in range(num_object):
                if np.linalg.norm(gt_pos[f][i] - gt_pos[f][obj]) < threshold and obj != i:
                    if obj not in last_approach:
                        s += 1
                    current_approach.append(obj)
            last_approach = current_approach
    return s


# number of target switch / number of occlusion
def target_switch_ratio(class_kalman, class_scene, threshold):
    num_approach = count_approach(class_scene.get_gt_pos(
    ), class_kalman.num_targets, class_scene.num_obj, [0, class_scene.max_frames], 20)
    num_switch = count_target_switch(
        class_kalman.correspondence, class_scene, class_kalman.num_targets, [0, class_scene.max_frames])
    # print(num_approach)
    # print(num_switch, "\n")
    if(num_approach > 0):
        return num_switch / num_approach
    else:
        return num_switch


# return the sum of the estimation error in the frame range
def error_sum(gt_pos, est_pos, frame_range):
    s = 0
    for f in range(frame_range[0], frame_range[1]):
        for i in range(est_pos.shape[0]):
            s += np.linalg.norm(np.array(est_pos[i]
                                [f])-np.array(gt_pos[f][i]))
    return s


# return the max of the estimation error in the frame range
def error_max(gt_pos, est_pos, frame_range):
    s = 0
    for f in range(frame_range[0], frame_range[1]):
        for i in range(est_pos.shape[0]):
            s = max(s, np.linalg.norm(np.array(est_pos[i]
                                               [f])-np.array(gt_pos[f][i])))
    return s


def get_accuracy(gt_pos, est_pos, frame, threshold):
    s = 0
    for i in range(est_pos.shape[0]):
        if np.min([np.linalg.norm(np.array(est_pos[j][frame])-np.array(gt_pos[frame][i])) for j in range(len(est_pos))]) < threshold:
            s += 1

    return s / (est_pos.shape[0])


def successful_tracking(gt_pos, est_pos, frame, threshold):
    if get_accuracy(gt_pos, est_pos, frame, threshold) == 1:
        return 1
    else:
        return 0


def errorfill(x, y, yerr, alpha_fill=0.3, ax=None):
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    base_line, = ax.plot(x, y)
    ax.fill_between(x, ymax, ymin, facecolor=base_line.get_color(),
                    alpha=alpha_fill)
    return base_line


def visualize_performance_multiple_runs(parameters, num_subject, num_run, metric="cumulated_error", bar_chart=True, with_error=True, save_graph=True, save_data=True, model="normal", x_axis_label="", y_axis_label="", threshold=20, title="", show=True, legend=""):
    # parameters : list of dictionaries
    # metric : cumulated_error, last_frame_error, get_accuracy, target_switch_ratio, success_rate
    labels = []
    means = []
    stds = []
    all_data = []

    ax = plt.gca()
    for d in parameters:
        print(d["label"])
        labels.append(d["label"])
        values = []
        for i in range(1, num_subject + 1):
            print(f"Subject {i}")
            subject_scores = []
            for _ in range(num_run):
                class_scene = Scene(d["class_canvas"], d["class_baseobj"],
                                    d["max_frames"], d["num_obj"], d["occlusion_settings"], d["std_measure"], render=False)
                class_scene.update_scenes_all(quiet=True)
                # Kalman parameters
                if(model == "normal"):
                    class_motkalman = MotKalman(class_scene, num_targets=d["num_targets"],
                                                dt=d["dt"], std_measure=d["std_measure"], std_pred=d["std_pred"], covP=d["covP"])
                elif (model == "acceleration"):
                    class_motkalman = MotKalmanAccel(class_scene, num_targets=d["num_targets"],
                                                     dt=d["dt"], std_measure=d["std_measure"], std_pred=d["std_pred"], covP=d["covP"])
                elif (model == "mean-reverting"):
                    class_motkalman = MotKalmanMR(class_scene, num_targets=d["num_targets"],
                                                  dt=d["dt"], std_measure=d["std_measure"], std_pred=d["std_pred"], covP=d["covP"], inertia_param=d["inertia"], spring_constant=d["spring_constant"])
                elif (model == "bouncing"):
                    class_motkalman = MotKalmanB(class_scene, num_targets=d["num_targets"],
                                                 dt=d["dt"], std_measure=d["std_measure"], std_pred=d["std_pred"], covP=d["covP"], inertia_param=d["inertia"], min_speed=d["min_speed"])
                else:
                    print("Unknown model")
                    break
                gt_pos = class_scene.get_gt_pos()
                est_pos = class_motkalman.est_pos

                if metric == "cumulated_error":
                    subject_scores.append(error_sum(gt_pos, est_pos,
                                                    [0, est_pos.shape[1]]))
                elif metric == "max_error":
                    subject_scores.append(error_max(gt_pos, est_pos,
                                                    [0, est_pos.shape[1]]))
                elif metric == "last_frame_error":
                    subject_scores.append(error_sum(gt_pos, est_pos, [
                        est_pos.shape[1]-1, est_pos.shape[1]]))
                elif metric == "get_accuracy":
                    subject_scores.append(get_accuracy(
                        gt_pos, est_pos, est_pos.shape[1] - 1, threshold))
                    ax.set_ylim(-0.05, 1.1)
                elif metric == "target_switch_ratio":
                    subject_scores.append(target_switch_ratio(
                        class_motkalman, class_scene, 30))
                elif metric == "success_rate":
                    with_error = False
                    subject_scores.append(successful_tracking(
                        gt_pos, est_pos, est_pos.shape[1] - 1, threshold))
                    ax.set_ylim(-0.05, 1.1)
            values.append(np.mean(subject_scores))
            all_data.append([f"Subject {i}"] + subject_scores)
        means.append(np.mean(values))
        stds.append(np.std(values))

    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)
    if with_error:
        if bar_chart:
            ax.bar(labels, means, yerr=stds, align='center',
                   alpha=0.5, ecolor='black', capsize=10)
        else:
            line = errorfill(labels, means, stds)
    else:
        if bar_chart:
            ax.bar(labels, means, align='center',
                   alpha=0.5, ecolor='black', capsize=10)
        else:
            line, = plt.plot(labels, means)
    if not legend == "":
        line.set_label(legend)
        ax.legend()
    plt.tight_layout()
    if save_graph:
        plt.savefig(f"{title}-{model}.png")
    if save_data:
        with open(f"{title}-{model}", 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_data)
    if show:
        plt.show()
    return means


if __name__ == '__main__':
    parameters = []

    num_target = 3
    num_distractor = 3
    canvas_x = 256
    canvas_y = 256
    max_frames = 500
    diameter = 20
    trajectory_type = "mean-reverting"
    speed_range = [2, 10]
    speedvar_prob = 0.5
    speedvar_std = 1
    directionvar_prob = 0.0
    directionvar_std = 30
    inertia_param = 0.05
    accelnoise_std = 1
    spring_constant = 0.001

    num_obj = num_distractor + num_target
    occlusion_settings = OcclusionSettings(
        -300, -320, [0, 0, canvas_x, canvas_y])
    color_settings = ColorSettings(color_type="white", init_hue_range=[0, 1],
                                   hue_drift_range=[0.01, 0.01])

    # Kalman parameters
    dt = 1
    std_pred = 10
    std_measure = 1
    num_targets = 4
    covP = 200

    # Register all scene parameters used during evaluation
    for s in range(0, 1):
        params = {}
        class_trajectory = Trajectory(trajectory_type, speed_range, speedvar_prob, speedvar_std,
                                      directionvar_prob, directionvar_std, inertia_param, accelnoise_std, spring_constant)
        class_canvas = Canvas(canvas_x, canvas_y)
        class_baseobj = Objs(class_canvas, diameter,
                             class_trajectory, color_settings)
        params["max_frames"] = max_frames
        params["num_obj"] = num_obj
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
        params["inertia"] = inertia_param
        params["min_speed"] = speed_range[0]
        params["spring_constant"] = spring_constant
        parameters.append(params)

    num_subject = 10
    num_run = 10

    means = visualize_performance_multiple_runs(
        parameters, num_subject, num_run, metric="get_accuracy", bar_chart=False, save_graph=False, save_data=False, model="normal", with_error=True, show=False,
        y_axis_label="Accuracy", x_axis_label="Process noise (standard deviation)", title="Kalman filter model performance\nas a function of process noise", legend="Classic Kalman filter")
    print(means[0])
