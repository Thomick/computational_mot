# Same as eval.py but for two parameters and 3d plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from cal_mot_kalman import MotKalman
from cal_mot_kalman_accel import MotKalmanAccel
from cal_mot_kalman_meanreverting import MotKalmanMR
from cal_mot_kalman_bouncing import MotKalmanB
from make_scenes import Trajectory, OcclusionSettings, ColorSettings, Scene, Objs, Canvas
from eval import count_target_switch, count_approach, target_switch_ratio, error_sum, error_max, get_accuracy, successful_tracking
import csv


def visualize_performance_multiple_runs(parameters, num_subject, num_run, metric="cumulated_error", save_graph=True, save_data=True, model="normal", x_axis_label="", y_axis_label="", z_axis_label="", threshold=20, title="", show=True):
    # parameters : list of dictionaries
    # metric : cumulated_error, last_frame_error, attention_window_score, target_switch_ratio, success_rate
    xlabels = []
    ylabels = []
    means = []
    stds = []
    all_data = []

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for d in parameters:
        print(d["xlabel"], d["ylabel"])
        xlabels.append(d["xlabel"])
        ylabels.append(d["ylabel"])
        values = []
        for _ in range(1, num_subject+1):
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
                    print("Unknown model type")
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
                elif metric == "attention_window_score":
                    subject_scores.append(get_accuracy(
                        gt_pos, est_pos, est_pos.shape[1] - 1, threshold))
                    ax.set_zlim(-0.05, 1.1)
                elif metric == "target_switch_ratio":
                    subject_scores.append(target_switch_ratio(
                        class_motkalman, class_scene, 30))
                elif metric == "success_rate":
                    subject_scores.append(successful_tracking(
                        gt_pos, est_pos, est_pos.shape[1] - 1, threshold))
                    ax.set_zlim(-0.05, 1.1)
            values.append(np.mean(subject_scores))
            all_data.append(
                [d["xlabel"], d["ylabel"], np.mean(subject_scores)])
        means.append(np.mean(values))
        stds.append(np.std(values))

    surf = ax.plot_trisurf(np.array(xlabels), np.array(ylabels),
                           np.array(means), cmap='viridis', linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_zlabel(z_axis_label)
    ax.set_title(title)
    plt.tight_layout()
    if save_graph:
        plt.savefig(f"{title}-{model}.png")
    if save_data:
        with open(f"{title}-{model}.csv", 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([x_axis_label, y_axis_label, z_axis_label])
            writer.writerows(all_data)
    if show:
        plt.show()


if __name__ == '__main__':
    parameters = []

    num_target = 4
    num_distractor = 4
    canvas_x = 256
    canvas_y = 256
    max_frames = 500
    diameter = 20
    trajectory_type = "bouncing"
    speed_range = [4, 4]
    speedvar_prob = 0.0
    speedvar_std = 1
    directionvar_prob = 0.0
    directionvar_std = 15
    inertia_param = 0.01
    accelnoise_std = 0
    spring_constant = 0.01

    num_obj = num_target + num_distractor
    occlusion_settings = OcclusionSettings(
        -300, -320, [0, 0, canvas_x, canvas_y])
    color_settings = ColorSettings(color_type="white", init_hue_range=[0, 1],
                                   hue_drift_range=[0.01, 0.01])

    # Kalman parameters
    dt = 1
    std_pred = 0.3
    std_measure = 1
    num_targets = 4
    covP = 200

    for i in range(0, 5):
        for s in range(0, 5):
            std_pred = s * 30 + 0.1
            std_measure = i*2 + 0.1
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
            params["inertia"] = inertia_param
            params["min_speed"] = speed_range[0]
            params["spring_constant"] = spring_constant
            params["xlabel"] = s * 30  # corresponding x on the plot
            params["ylabel"] = i*2  # corresponding y on the plot
            parameters.append(params)

    num_subject = 10
    num_run = 5

    visualize_performance_multiple_runs(
        parameters, num_subject, num_run, metric="attention_window_score", save_graph=True, save_data=True, model="normal", show=True,
        y_axis_label="Measure noise\n(standard deviation)", x_axis_label="Process noise\n(standard deviation)", z_axis_label="Accuracy", title="Kalman filter model performance\nas a function of process noise and measure noise")
