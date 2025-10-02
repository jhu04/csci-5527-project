import scipy
import numpy as np
import matplotlib.pyplot as plt

data = np.load("./data/advection/1D_Advection_Sols_beta1.0.npy")
t_coords = np.load("./data/advection/t_coordinate.npy")
x_coords = np.load("./data/advection/x_coordinate.npy")
print(data.shape)
print(np.min(t_coords))
print(np.max(t_coords))

fig, (axes) = plt.subplots(2,5, figsize=(20, 8))

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.set_title(f"Image {i}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_box_aspect(1)
    ax.imshow(data[i], extent=[np.min(t_coords), np.max(t_coords), np.min(x_coords), np.max(x_coords)], aspect='auto')

plt.show()
