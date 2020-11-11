import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


big_data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data-synced.npz',
                   allow_pickle=True)['data'].item()

poses = big_data['10']['SLS1R']['positions_3d'][0]
poses = np.load(
    '/home/filipkr/Documents/xjob/pose-data/orig-dims/18-100e.npy')
poses = poses - np.expand_dims(poses[:, 0, :], 1)


def update_graph(i):
    # data=df[df['time']==num]
    graph.set_data(poses[i, :, 0], poses[i, :, 1])
    graph.set_3d_properties(poses[i, :, 2])
    lines.set_data(poses[i, skeleton, 0], poses[i, skeleton, 1])
    lines.set_3d_properties(poses[i, skeleton, 2])
    # title.set_text('3D Test, time={}'.format(num))
    # print(poses[i, skeleton, 0], poses[i, skeleton, 1], poses[i, skeleton, 2])

    # xmin = np.min(poses[i, skeleton, 0])
    # xmax = np.max(poses[i, skeleton, 0])
    # ymin = np.min(poses[i, skeleton, 1])
    # ymax = np.max(poses[i, skeleton, 1])
    # zmin = np.min(poses[i, skeleton, 2])
    # zmax = np.max(poses[i, skeleton, 2])

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_zlim(zmin, zmax)
    return title, graph


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=10., azim=40)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_zlim(-10, 10)

skeleton = [3, 2, 1, 0, 4, 0, 6, 5, 7, 5, 8]

graph, = ax.plot(poses[0, :, 0], poses[0, :, 1],
                 poses[0, :, 2], linestyle="", marker="o")
lines, = ax.plot(poses[0, skeleton, 0], poses[0, skeleton, 1],
                 poses[0, skeleton, 2])
# lines2, = ax.plot(poses[0, [0, 6, 5], 0], poses[0, [0, 6, 5], 1],
#                   poses[0, [0, 6, 5], 2])

Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, poses.shape[0],
                                         interval=10, blit=False)

ani.save('/home/filipkr/Desktop/animation.mp4', writer=writer)

plt.show()
