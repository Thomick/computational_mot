# Small script to plot human and model performance as a function of speed

import matplotlib.pyplot as plt
import numpy as np


def get_model_perf(basefolder, measstd, procstd, speed):
    res = []
    filename = f"{basefolder}/{speed}px/model_perf.csv"
    with open(filename, 'r') as f:
        _ = f.readline()
        for l in f.readlines():
            meas, proc, _, score, = l.split(',')
            if float(meas) == measstd and float(proc) == procstd:
                res.append(float(score))
    return np.mean(res)


filename = "trials/human_average44.csv"
title = "Average accuracy as a function of target speed (4 targets, 4 distractors)"


fig = plt.figure()
ax = fig.gca()
x_label = ""
y_label = ""
x = []
y = []
# Read human performances
with open(filename, 'r') as f:
    x_label, y_label, = f.readline().split(',')
    for line in f.readlines():
        xval, yval, = line.split(',')
        x.append(float(xval))
        y.append(float(yval))

line, = plt.plot(x, y, '-o')
line.set_label("Human subject")
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title(title)
ax.set_ylim(-0.05, 1.05)

# Configure other lines of the plot
# Line 2
measstd = 1.1
procstd = 0.1
y_model = []
basefolder = "trials/scene44_higher_res"
x = [1.0, 2.5, 4.0, 5.5, 7.0]
for speed in x:
    print(get_model_perf(basefolder, measstd, procstd, speed))
    y_model.append(get_model_perf(basefolder, measstd, procstd, speed))

line, = ax.plot(x, y_model, '-o')
line.set_label(
    f"Linear Kalman filter (process noise = {procstd}, measurement noise = {measstd})")

# Line 3
measstd = 1.0
procstd = 0
y_model = []
basefolder = "trials/scene44"
x = [1.0, 2.5, 4.0, 5.5, 7.0]
for speed in x:
    print(get_model_perf(basefolder, measstd, procstd, speed))
    y_model.append(get_model_perf(basefolder, measstd, procstd, speed))

line, = ax.plot(x, y_model, '-o')
line.set_label(
    f"Linear Kalman filter (process noise = {procstd+0.1}, measurement noise = {measstd})")

ax.set_xlim(0.75, 7.25)
ax.legend()

plt.show()
