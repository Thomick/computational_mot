# Plot average across all speeds and all numbers of targets as a function of 2 parameters
# Needs the right folder structure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from visualize_human_model_comp_allspeeds import plot_flat, get_data


human_score_file = "human_score.csv"
model_perf_file = "model_perf.csv"
annot = " and all numbers of targets"
size_x = 9
size_y = 8

base_folder = "trials/scene33"
speeds = [1.0, 2.5, 4.0, 5.5, 7.0]
x_values, y_values, z_values3 = get_data(
    base_folder, speeds, human_score_file, model_perf_file)

base_folder = "trials/scene44"
speeds = [1.0, 2.5, 4.0, 5.5, 7.0]
_, _, z_values4 = get_data(
    base_folder, speeds, human_score_file, model_perf_file)

base_folder = "trials/scene55"
speeds = [1.0, 2.5, 4.0]
_, _, z_values5 = get_data(
    base_folder, speeds, human_score_file, model_perf_file)

z_values = (np.array(z_values3)+np.array(z_values4)+np.array(z_values5))/3

plot_flat(x_values, y_values, z_values, size_x, size_y, annot)
