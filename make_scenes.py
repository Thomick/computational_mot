import numpy as np
import numpy.random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys


class Canvas():
    def __init__(self, canvas_x=256, canvas_y=256):
        self.canvas_x = canvas_x
        self.canvas_y = canvas_y
        self.center_x = int(canvas_x/2)
        self.center_y = int(canvas_y/2)


class ColorSettings():
    def __init__(self, color_type="white", init_hue_range=[0, 0], hue_drift_range=[0, 0]):
        self.type = color_type
        self.init_hue_range = init_hue_range
        self.hue_drift_range = hue_drift_range

    def init_value(self):
        if self.type == "white":
            return 255
        elif self.type == "hue":
            return rnd.uniform(self.init_hue_range[0], self.init_hue_range[1])

    def get_modifier(self):
        if self.type == "white":
            return 0
        elif self.type == "hue":
            return rnd.uniform(self.hue_drift_range[0], self.hue_drift_range[1]) * rnd.choice([-1, 1])

    def get_rgb(self, val):
        if self.type == "white":
            return (255, 255, 255)
        elif self.type == "hue":
            return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(val, 1, 1))


class OcclusionSettings():
    def __init__(self, start=-1, end=-1, occlusion_rectangle=[-1, -1, -1, -1], color=[42, 157, 143]):
        self.start = start  # Frame in which occlusion start
        self.end = end  # last frame of occlusion
        self.color = color
        # coords of upper left corner and lower right corner [x_ll,y_ll,x_lr,y_lr]
        self.rectangle = occlusion_rectangle


class Scene():
    def __init__(self, class_canvas, class_baseobj, max_frames=100, num_obj=1, occlusion_settings=OcclusionSettings(), measure_noise_std=0.3, render=True):
        self.class_canvas = class_canvas

        # about time
        self.max_frames = max_frames
        self.current_frame = 0
        # about objects
        self.num_obj = num_obj
        self.stored_objs = []
        self.init_stored_centerpt = []
        self.tmp_centerpt = []
        self.output_centerpt = []
        self.init_stored_color = []
        self.tmp_color = []
        self.output_color = []
        # Set to false if the scene will never be visualized (speed up generation)
        self.render = render
        # occlusion
        # Removed because of conflicts with correspondence, should be reimplemented in make_measurement
        self.occlusion_settings = occlusion_settings
        # initialize
        self.initialize_scenes(class_baseobj)
        self.measure_noise_std = measure_noise_std
        self.correspondence = [list(range(self.num_obj))] + [rnd.permutation(
            list(range(self.num_obj))) for _ in range(self.max_frames - 1)]  # Contains the association between measurement and objects
        self.measurements = []

    def initialize_scenes(self, class_baseobj):
        # initialize output frames
        self.output_frames = np.zeros(
            (self.max_frames, self.class_canvas.canvas_y, self.class_canvas.canvas_x, 3), dtype=np.uint8)

        # make objects
        [self.add_newobj(class_baseobj) for i in range(self.num_obj)]
        #
        [self.draw_circle(0, i) for i in range(self.num_obj)]
        self.output_centerpt.append(self.init_stored_centerpt)
        self.output_color.append(self.init_stored_color)

    def update_scene(self):
        self.current_frame += 1
        self.tmp_centerpt = []
        self.tmp_color = []
        [self.stored_objs[i].update_pos() for i in range(self.num_obj)]
        [self.draw_circle(self.current_frame, i)
         for i in range(self.num_obj)]

        if self.render:
            self.draw_occlusion(self.current_frame)
        self.output_centerpt.append(self.tmp_centerpt)
        self.output_color.append(self.tmp_color)

    def update_scenes_all(self, quiet=False):
        for _ in range(self.max_frames-self.current_frame - 1):
            self.update_scene()
        if not quiet:
            print("All frames were generated")
        self.make_measurements()

    def add_newobj(self, class_baseobj):
        new_obj = Objs(class_baseobj.class_canvas, class_baseobj.diameter,
                       class_baseobj.class_trajectory, class_baseobj.color_settings)
        flag_loop = True
        flag_overlap = False
        while flag_loop == True:
            new_obj.move_to_random_pos()
            for j in range(len(self.stored_objs)):
                flag_overlap = False
                d = math.dist([new_obj.pos[0], new_obj.pos[1]], [
                              self.stored_objs[j].pos[0], self.stored_objs[j].pos[1]])
                if (d < self.stored_objs[j].diameter):
                    flag_overlap = True
                    break
            if flag_overlap == False:
                self.stored_objs.append(new_obj)
                self.init_stored_centerpt.append(
                    [new_obj.pos[0], new_obj.pos[1]])
                self.init_stored_color.append(new_obj.color_value)
                flag_loop = False

    def draw_circle(self, ind_frame, ind_obj):
        col = self.stored_objs[ind_obj].get_color()
        if self.render:
            w1 = np.arange(0, self.class_canvas.canvas_x, 1)
            h1 = np.arange(0, self.class_canvas.canvas_y, 1)
            wx, wy = np.meshgrid(w1, h1)
            wx = wx-(self.class_canvas.center_x)
            wy = wy-(self.class_canvas.center_y)
            wx = wx - (self.stored_objs[ind_obj].pos[0] -
                       self.class_canvas.center_x)
            wy = wy - (self.stored_objs[ind_obj].pos[1] -
                       self.class_canvas.center_y)
            r = np.sqrt(wx**2 + wy**2)

            tmp_r = self.output_frames[ind_frame, :, :, 0]
            tmp_r[np.where(r < self.stored_objs[ind_obj].diameter/2)
                  ] = col[0]
            tmp_g = self.output_frames[ind_frame, :, :, 1]
            tmp_g[np.where(r < self.stored_objs[ind_obj].diameter/2)
                  ] = col[1]
            tmp_b = self.output_frames[ind_frame, :, :, 2]
            tmp_b[np.where(r < self.stored_objs[ind_obj].diameter/2)
                  ] = col[2]

        # center position stored
        self.tmp_centerpt.append(
            [self.stored_objs[ind_obj].pos[0], self.stored_objs[ind_obj].pos[1]])
        # Hue stored
        self.tmp_color.append(self.stored_objs[ind_obj].color_value)

    def draw_occlusion(self, ind_frame):
        if ind_frame >= self.occlusion_settings.start and ind_frame <= self.occlusion_settings.end:
            x_l = self.occlusion_settings.rectangle[0]
            x_r = self.occlusion_settings.rectangle[2]
            y_l = self.occlusion_settings.rectangle[1]
            y_r = self.occlusion_settings.rectangle[3]
            self.output_frames[ind_frame, x_l:x_r, y_l:y_r,
                               0] = self.occlusion_settings.color[0]
            self.output_frames[ind_frame, x_l:x_r, y_l:y_r,
                               1] = self.occlusion_settings.color[1]
            self.output_frames[ind_frame, x_l:x_r, y_l:y_r,
                               2] = self.occlusion_settings.color[2]

    def get_gt_pos(self):
        return np.array([self.output_centerpt[ind] for ind in range(self.max_frames)])

    def make_measurements(self, measure_std=None):  # Generate new measurements
        if not measure_std == None:
            self.measure_noise_std = measure_std
        self.measurements = np.array([self.output_centerpt[ind] + rnd.normal(0, self.measure_noise_std, (self.num_obj, 2))
                                      for ind in range(self.max_frames)])

    # Get measurements (with or without permutation)
    def get_measurements(self, permutation=False):
        if permutation:
            return [self.measurements[ind_frame][[self.correspondence[ind_frame][ind_obj] for ind_obj in range(self.num_obj)]] for ind_frame in range(self.max_frames)]
        else:
            return self.measurements

    def in_rectangle(self, pos, rectangle):
        return pos[0] >= rectangle[0] and pos[0] < rectangle[2] and pos[1] >= rectangle[1] and pos[1] < rectangle[3]

    def is_visible(self, ind_frame, ind_obj):
        return self.in_rectangle(self.output_centerpt[ind_frame][ind_obj], [0, 0, self.class_canvas.canvas_x, self.class_canvas.canvas_y]) and \
            not (ind_frame >= self.occlusion_settings.start and ind_frame <= self.occlusion_settings.end
                 and self.in_rectangle(self.output_centerpt[ind_frame][ind_obj], self.occlusion_settings.rectangle))

    def save_scene(self, name):  # Save scene center points
        np.save(name, self.output_centerpt)

    # Load scene center points, can't be used before generating an animation
    # Run make_measurement before anything else
    # The scene must only be used to run one of the models using make_measurement/get_measurement
    def load_scene(self, name):
        self.output_centerpt = np.load(f"{name}.npy")

# Trajectory parameters and transition function


class Trajectory():
    def __init__(self, trajectory_type="bouncing", speed_range=[3, 7], speedvar_prob=0., speedvar_std=4,
                 directionvar_prob=0., directionvar_std=10, inertia_param=0, accelnoise_std=0, spring_constant=0.01):
        self.trajectory_type = trajectory_type  # "bouncing" or "mean-reverting"
        self.speed_range = speed_range
        self.speedvar_prob = speedvar_prob
        self.speedvar_std = speedvar_std
        self.directionvar_prob = directionvar_prob
        self.directionvar_std = directionvar_std
        self.inertia_param = inertia_param
        self.accelnoise_std = accelnoise_std
        self.spring_constant = spring_constant

    def init_speed(self):
        return (self.speed_range[0] + self.speed_range[1])/2

    def get_speed(self, obj):
        speed = obj.speed
        if self.trajectory_type == "bouncing":
            speed -= speed * self.inertia_param
            if rnd.uniform() < self.speedvar_prob:
                speed_norm = np.linalg.norm(speed, 2)
                if speed_norm > 0:
                    increment = rnd.normal(0, self.speedvar_std)
                    if increment > -speed_norm:
                        speed += increment / \
                            speed_norm * speed
                    else:
                        speed = self.speed_range[0] / speed_norm * speed
            if rnd.uniform() < self.directionvar_prob:
                rot_angle = rnd.normal(0, self.directionvar_std)
                s = np.sin(np.deg2rad(rot_angle))
                c = np.cos(np.deg2rad(rot_angle))
                speed = speed.dot(np.array([[c, -s], [s, c]]))

            if obj.pos[0] < obj.diameter/2:
                speed[0] = np.abs(speed[0])
            if obj.pos[1] < obj.diameter/2:
                speed[1] = np.abs(speed[1])
            if obj.pos[0] > obj.class_canvas.canvas_x - obj.diameter/2:
                speed[0] = -np.abs(speed[0])
            if obj.pos[1] > obj.class_canvas.canvas_y - obj.diameter/2:
                speed[1] = -np.abs(speed[1])
            speed_norm = np.linalg.norm(speed, 2)
            if speed_norm > 0:
                if speed_norm > self.speed_range[1]:
                    speed = self.speed_range[1]*speed / speed_norm
                elif speed_norm < self.speed_range[0]:
                    speed = self.speed_range[0]*speed / speed_norm

        elif self.trajectory_type == "mean-reverting":
            spring_comp = self.spring_constant * \
                (np.array([obj.class_canvas.center_x, obj.class_canvas.center_y])
                 - obj.pos)
            speed += -self.inertia_param*speed + \
                spring_comp + rnd.normal(0, self.accelnoise_std, 2)
            if obj.pos[0] < obj.diameter/2:
                obj.pos[0] = obj.diameter/2
            if obj.pos[1] < obj.diameter/2:
                obj.pos[1] = obj.diameter/2
            if obj.pos[0] > obj.class_canvas.canvas_x - obj.diameter/2:
                obj.pos[0] = obj.class_canvas.canvas_x - obj.diameter/2
            if obj.pos[1] > obj.class_canvas.canvas_y - obj.diameter/2:
                obj.pos[1] = obj.class_canvas.canvas_y - obj.diameter/2

        return speed


class Objs():
    def __init__(self, class_canvas, diameter=50, class_trajectory=Trajectory(), color_settings=ColorSettings()):
        self.class_canvas = class_canvas
        self.diameter = diameter
        self.class_trajectory = class_trajectory
        self.move_to_random_pos()
        self.speed = self.class_trajectory.init_speed()
        direction = math.floor(rnd.uniform(0, 360))
        self.speed = np.array(
            [np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))], dtype=np.float64)
        self.color_settings = color_settings
        self.color_value = self.color_settings.init_value()
        self.color_modifier = self.color_settings.get_modifier()

    def update_pos(self):
        self.speed = self.class_trajectory.get_speed(self)
        self.pos += self.speed
        self.color_value += self.color_modifier

    def get_color(self):
        return self.color_settings.get_rgb(self.color_value)

    def move_to_random_pos(self):
        x = np.random.randint(
            int(self.diameter/2), self.class_canvas.canvas_x-int(self.diameter/2))
        y = np.random.randint(
            int(self.diameter/2), self.class_canvas.canvas_y-int(self.diameter/2))
        self.pos = np.array([x, y], dtype=np.float64)


def visualize_frames(class_scene, canvas_x, canvas_y, flag_save=False, fname_save='sample.gif'):
    print("Preparing animation")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tmp = class_scene.output_frames.astype('uint8')
    imgs = [[ax.imshow(np.reshape(tmp[j, :, :, :], [canvas_y, canvas_x, 3]))]
            for j in range(class_scene.max_frames)]
    ani = animation.ArtistAnimation(fig, imgs, interval=1)
    # ani.save("sample.gif", writer="pillow")
    if (flag_save == True):
        ani.save(fname_save, writer="pillow", fps=60)
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

    class_trajectory = Trajectory(trajectory_type, speed_range, speedvar_prob, speedvar_std,
                                  directionvar_prob, directionvar_std, inertia_param, accelnoise_std, spring_constant)
    class_canvas = Canvas(canvas_x, canvas_y)
    class_baseobj = Objs(class_canvas, diameter,
                         class_trajectory, color_settings)
    class_scene = Scene(class_canvas, class_baseobj,
                        max_frames, num_obj, occlusion_settings, 3)
    class_scene.update_scenes_all()

    flag_save = True
    fname_save = 'sample_mot.gif'
    visualize_frames(class_scene, canvas_x, canvas_y, flag_save, fname_save)
