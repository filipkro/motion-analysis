
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

# x = np.random.normal(size=(80, 3))
# df = pd.DataFrame(x, columns=["x", "y", "z"])

data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data-synced.npz',
               allow_pickle=True)['data'].item()

poses = data['10']['SLS1R']['positions_3d'][0]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='darkblue', alpha=0.5)
pc = ax.plot([], [], [], c='darkblue', alpha=0.5)
print(sc)
print(pc)


def update(i):
    sc._offsets3d = (poses[i, :, 0], poses[i, :, 1], poses[i, :, 2])
    pc._offsets3d = (poses[i, [3, 2, 1, 0, 4], 0], poses[i, [
        3, 2, 1, 0, 4], 1], poses[i, [3, 2, 1, 0, 4], 2])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ani = matplotlib.animation.FuncAnimation(
    fig, update, frames=poses.shape[0], interval=3)

plt.tight_layout()
plt.show()
