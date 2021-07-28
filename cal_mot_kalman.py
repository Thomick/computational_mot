import numpy as np
from make_scenes import Canvas, Scene, Objs, ColorSettings, OcclusionSettings, Trajectory
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MotKalman():
    def __init__(self, class_scene, num_targets=1, dt=1., std_measure=0.3, std_pred=0.3, covP=100., use_correspondence=True):
        self.class_scene = class_scene
        self.num_measures = class_scene.max_frames
        self.num_targets = num_targets
        self.std_measure = std_measure
        self.std_pred = std_pred
        self.covP = covP
        self.dt = dt
        self.measured_pos = []
        self.K = [[]]*self.num_targets
        self.x = []
        self.P = []
        self.estimation = np.zeros((self.num_targets, self.num_measures, 4))
        self.est_pos = np.zeros((self.num_targets, self.num_measures, 2))
        self.correspondence = [[]]*self.num_measures
        # Use the correspondence part of the model (if not the measurement are not shuffled)
        self.use_correspondence = use_correspondence

        self.init_F()
        q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.std_pred)
        self.Q = block_diag(q, q)
        # to extract position coordinates
        self.H = np.array([[1., 0., 0, 0], [0., 0., 1., 0.]])
        # measurement noise
        self.R = np.array([[self.std_measure**2, 0], [0, self.std_measure**2]])
        self.R_missing_data = np.array([[10000, 0], [0, 10000]])
        self.I = np.identity(4)  # identity matrix

        self.extract_measure()
        self.init_kalmanparams_all()
        self.exec_kalmanfilter_all()

        # for visualize
        self.col_obj = [255, 0, 0]
        self.est_frames = np.ones(
            (class_scene.output_frames.shape))*class_scene.output_frames

    def init_F(self):
        self.F = np.array([[1., self.dt, 0., 0.], [0., 1., 0., 0.], [
            0., 0., 1., self.dt], [0., 0., 0., 1.]])

    def extract_measure(self):
        if self.use_correspondence:
            self.measured_pos = self.class_scene.get_measurements(
                permutation=True)
        else:
            self.measured_pos = self.class_scene.get_measurements(
                permutation=False)

    def init_kalmanparams_all(self):
        [self.init_kalmanparams_eachobj(i) for i in range(self.num_targets)]

    def init_kalmanparams_eachobj(self, ind_obj):
        self.x.append(np.array([[self.measured_pos[0][ind_obj][0]], [0.], [self.measured_pos[0][ind_obj][1]], [
                      0.]]))  # initial [position x,velocity x, velocity y, position y]
        self.P.append(np.array([[0., 0., 0., 0.], [0., self.covP, 0., 0.], [
                      0., 0., 0., 0.], [0., 0., 0., self.covP]]))  # covariance matrix

    # Link the measurement to their respecting target
    def build_correspondence(self, ind_frame):
        assignment = [-1] * self.num_targets
        x = []
        P = []
        for i in range(self.num_targets):
            x_temp, P_temp = self.predictNext(i)
            x.append(x_temp[[0, 2]])
            P.append(P_temp)
        for i in range(self.num_targets):
            me = sorted(range(len(self.measured_pos[ind_frame])),
                        key=lambda j: np.linalg.norm(x[i].T-np.array(self.measured_pos[ind_frame][j])))
            for j in me:
                if j not in assignment:
                    assignment[i] = j
                    break
        self.correspondence[ind_frame] = assignment

    def exec_kalmanfilter_all(self):
        for n in range(self.num_measures):
            if n > 0:
                self.build_correspondence(n)
            [self.exec_kalmanfilter_eachobj(n, i)
             for i in range(self.num_targets)]

    def predictNext(self, ind_obj):
        x = np.array(self.x[ind_obj])
        P = np.array(self.P[ind_obj])

        x = np.dot(self.F, x)
        P = np.dot(np.dot(self.F, P), self.F.T) + self.Q
        return x, P

    def exec_kalmanfilter_eachobj(self, ind_frame, ind_obj):
        x, P = self.predictNext(ind_obj)
        R = self.R

        if (self.use_correspondence):   # Wether or not
            if ind_frame == 0:
                z = np.array([self.measured_pos[ind_frame][ind_obj]])
                R = np.array([[0., 0.], [0., 0.]])
            elif self.correspondence[ind_frame][ind_obj] == -1:
                z = np.array([self.est_pos[ind_obj, ind_frame-1]])
                R = self.R_missing_data
            else:
                z = np.array([self.measured_pos[ind_frame]
                              [self.correspondence[ind_frame][ind_obj]]])
        else:
            z = np.array([self.measured_pos[ind_frame]
                         [ind_obj]])

        y = z.T - np.dot(self.H, x)
        S = np.dot(np.dot(self.H, P), self.H.T) + R
        K = np.dot(np.dot(P, self.H.T), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = np.dot((self.I - np.dot(K, self.H)), P)
        self.estimation[ind_obj, ind_frame, :] = np.resize(x, x.shape[0])
        self.est_pos[ind_obj, ind_frame, :] = np.resize(
            np.array([x[0], x[2]]), 2)

        self.x[ind_obj] = x.tolist()
        self.P[ind_obj] = P.tolist()
        self.K[ind_obj].append(K)

    def make_estframes(self):
        for t in range(self.num_measures-1):
            [self.draw_circle_estimated(t+1, i)
             for i in range(self.num_targets)]

    def draw_circle_estimated(self, ind_frame, ind_obj):
        w1 = np.arange(0, self.class_scene.class_canvas.canvas_x, 1)
        h1 = np.arange(0, self.class_scene.class_canvas.canvas_y, 1)
        wx, wy = np.meshgrid(w1, h1)
        wx = wx-(self.class_scene.class_canvas.center_x)
        wy = wy-(self.class_scene.class_canvas.center_y)
        wx = wx - (self.est_pos[ind_obj][ind_frame]
                   [0] - self.class_scene.class_canvas.center_x)
        wy = wy - (self.est_pos[ind_obj][ind_frame]
                   [1] - self.class_scene.class_canvas.center_y)
        r = np.sqrt(wx**2 + wy**2)

        tmp_r = self.est_frames[ind_frame, :, :, 0]
        tmp_r[np.where(
            r < self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[0]
        tmp_g = self.est_frames[ind_frame, :, :, 1]
        tmp_g[np.where(
            r < self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[1]
        tmp_b = self.est_frames[ind_frame, :, :, 2]
        tmp_b[np.where(
            r < self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[2]


def visualize_frames_est(class_motkalman, canvas_x, canvas_y, flag_save=False, fname_save='sample_est.gif', show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tmp = class_motkalman.est_frames.astype('uint8')
    imgs = [[ax.imshow(np.reshape(tmp[j, :, :, :], [canvas_y, canvas_x, 3]))]
            for j in range(class_motkalman.class_scene.max_frames)]
    ani = animation.ArtistAnimation(fig, imgs, interval=1)
    if (flag_save == True):
        ani.save(fname_save, writer="pillow", fps=60)
    if show:
        plt.show()


if __name__ == '__main__':
    num_obj = 8
    canvas_x = 256
    canvas_y = 256
    max_frames = 500
    diameter = 20
    trajectory_type = "mean-reverting"  # "bouncing" or "mean-reverting"
    speed_range = [4, 7]  # in pixels per time
    speedvar_prob = 0.2
    speedvar_std = 1  # in pixels per time
    directionvar_prob = 0
    directionvar_std = 15  # in degrees
    inertia_param = 0.0  # between 0 and 1
    accelnoise_std = 0
    spring_constant = 0.001  # > 0

    occlusion_settings = OcclusionSettings(
        -10, -500, [200, 0, canvas_x, canvas_y])
    color_settings = ColorSettings(color_type="white", init_hue_range=[0, 1],
                                   hue_drift_range=[0.01, 0.01])

    # Kalman parameters
    dt = 1
    std_pred = 0.3
    std_measure = 0.3
    num_targets = 4
    covP = 200

    ##############################
    # make scenes
    class_trajectory = Trajectory(trajectory_type, speed_range, speedvar_prob, speedvar_std,
                                  directionvar_prob, directionvar_std, inertia_param, accelnoise_std, spring_constant)
    class_canvas = Canvas(canvas_x, canvas_y)
    class_baseobj = Objs(class_canvas, diameter,
                         class_trajectory, color_settings)
    class_scene = Scene(class_canvas, class_baseobj,
                        max_frames, num_obj, occlusion_settings, std_measure)
    class_scene.update_scenes_all()

    # Kalman parameters
    class_motkalman = MotKalman(class_scene, num_targets=num_targets,
                                dt=dt, std_measure=std_measure, std_pred=std_pred, covP=covP)
    # def __init__(self,class_scene,num_targets=1,dt=1.,std_measure=0.3,std_pred=0.3,covP=100.):
    class_motkalman.make_estframes()

    flag_save = True
    fname_save_est = 'sample_mot_est.gif'
    visualize_frames_est(class_motkalman, canvas_x,
                         canvas_y, flag_save, fname_save_est)
