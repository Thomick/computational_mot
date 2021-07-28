# Generate scenes and run a model on them for different parameters (2) (Saved to files)
# Should be visualized with visualize_human_model_comp.py

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from cal_mot_kalman import MotKalman, visualize_frames_est
from cal_mot_kalman_accel import MotKalmanAccel
from cal_mot_kalman_meanreverting import MotKalmanMR
from cal_mot_kalman_bouncing import MotKalmanB
from make_scenes import Trajectory, OcclusionSettings, ColorSettings, Scene, Objs, Canvas
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
from eval import get_accuracy


def draw_circle(frame, pos, diameter, color):
    w1 = np.arange(0, frame.shape[0], 1)
    h1 = np.arange(0, frame.shape[1], 1)
    wx, wy = np.meshgrid(w1, h1)
    wx = wx - pos[0]
    wy = wy - pos[1]
    r = np.sqrt(wx ** 2 + wy ** 2)

    tmp_r = frame[:, :, 0]
    tmp_r[np.where(r < diameter/2)
          ] = color[0]
    tmp_g = frame[:, :, 1]
    tmp_g[np.where(r < diameter/2)
          ] = color[1]
    tmp_b = frame[:, :, 2]
    tmp_b[np.where(r < diameter/2)
          ] = color[2]


def draw_number(number, frame, pos, size):
    text = str(number)
    pil_font = ImageFont.truetype("Arial.TTF", size=size // len(text),
                                  encoding="unic")
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    white = "#FF0000"
    draw.text(pos, text, font=pil_font, fill=white)
    return np.array(canvas)


def make_video_for_humans(name, scene, num_target):
    frames = scene.output_frames
    starting_frames = np.stack([frames[0] for _ in range(30)], 0)
    for i in range(30):
        for j in range(num_target):
            draw_circle(
                starting_frames[i], scene.output_centerpt[0][j], scene.stored_objs[0].diameter, (0, 0, 255))
    num_finalframes = 200
    final_frames = np.stack([frames[scene.max_frames-1]
                            for _ in range(num_finalframes)], 0)
    perm = rnd.permutation(range(scene.num_obj))
    for i in range(num_finalframes):
        for j in range(scene.num_obj):
            final_frames[i] = draw_number(
                perm[j], final_frames[i], scene.output_centerpt[scene.max_frames-1][j]-np.array([5, 10]), 20)
    output_frames = np.concatenate([starting_frames, frames, final_frames], 0)
    print(output_frames.shape)
    clip = ImageSequenceClip(list(output_frames), fps=30)
    clip.write_gif(f'{name}.gif', fps=30, logger=None)
    return perm[:num_target]


def compute_average_diff(reference, scores):
    total = 0
    for i in range(len(reference)):
        total += reference[i] - scores[i]
    return total/len(reference)


if __name__ == '__main__':

    # Trial parameters
    folder = "scene33/4.0px"
    generate_new_scenes = True  # Use already generated scenes from the folder
    nb_scene = 30
    nb_rep = 5

    # Default parameters
    num_target = 3
    num_distractor = 3
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

    num_obj = num_distractor + num_target
    occlusion_settings = OcclusionSettings(
        -300, -320, [0, 0, canvas_x, canvas_y])
    color_settings = ColorSettings(color_type="white", init_hue_range=[0, 1],
                                   hue_drift_range=[0.01, 0.01])

    # Kalman parameters
    dt = 1
    std_pred = 10
    std_measure = 1
    covP = 200

    scenes = []
    for i in range(nb_scene):
        # Generate scene
        class_canvas = Canvas(canvas_x, canvas_y)
        class_trajectory = Trajectory(trajectory_type, speed_range, speedvar_prob, speedvar_std,
                                      directionvar_prob, directionvar_std, inertia_param, accelnoise_std, spring_constant)
        class_baseobj = Objs(class_canvas, diameter,
                             class_trajectory, color_settings)
        if generate_new_scenes:
            class_scene = Scene(class_canvas, class_baseobj,
                                max_frames, num_obj, occlusion_settings, std_measure, render=True)
        else:
            class_scene = Scene(class_canvas, class_baseobj,
                                max_frames, num_obj, occlusion_settings, std_measure, render=False)
        class_scene.update_scenes_all()

        if generate_new_scenes:
            class_scene.save_scene(f"{folder}/scene{i}")
            file_name = i
            answers = make_video_for_humans(
                f"{folder}/{file_name}", class_scene, num_target)
            with open(f"{folder}/answer{i}.txt", 'w') as f:  # Write answers
                f.write(f"answer = {answers}\n")
        else:
            class_scene.load_scene(f"{folder}/scene{i}")

        scenes.append(class_scene)

    proc_noises = []
    meas_noises = []
    scores = []
    scene_id = []
    for proc in range(8):
        for meas in range(9):
            for i in range(nb_scene):
                total = 0
                std_pred = proc*20
                std_measure = meas
                for j in range(nb_rep):
                    scenes[i].make_measurements(std_measure + 0.1)
                    class_motkalman = MotKalman(
                        scenes[i], num_target, dt, std_measure + 0.1, std_pred + 0.1, covP)
                    gt_pos = scenes[i].get_gt_pos()
                    est_pos = class_motkalman.est_pos
                    score = get_accuracy(
                        gt_pos, est_pos, max_frames - 1, 20)
                    total += score
                proc_noises.append(std_pred)
                meas_noises.append(std_measure)
                scores.append(total/nb_rep)
                scene_id.append(i)
                print(std_measure, std_pred, total / nb_rep)
    # Write model performance to a csv file
    with open(f"{folder}/model_perf.csv", "w") as f:
        f.write("Measurement noise,Prediction noise,Scene id, Score\n")
        for i in range(len(scores)):
            f.write(
                f"{meas_noises[i]},{proc_noises[i]},{scene_id[i]},{scores[i]}\n")
