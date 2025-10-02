import time
import torch
import numpy as np
import scipy.io

import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_min = -1
    x_max = 1
    x_num = 256
    t_min = -1
    t_max = 1
    t_num = 256

    out_data = {}
    x = np.linspace(x_min, x_max, num=x_num)[:,None]
    t = np.linspace(t_min, t_max, num=t_num)[:,None]

    out_data['x'] = x
    out_data['t'] = t

    xv, tv = np.meshgrid(x, t)
    xtv = np.stack([xv, tv])
    xtv = np.transpose(xtv, (1, 2, 0))

    sol_eq = lambda x: np.sin(np.pi * x[0]) * np.sin(4 * np.pi * x[1])
    usol = np.apply_along_axis(sol_eq, 2, xtv)

    out_data['usol'] = usol

    scipy.io.savemat('./data/helmholtz.mat', out_data)

    # Plot predictions, GT, and error over the full range
    fig, ((ax1)) = plt.subplots(1,1, figsize=(18, 12))

    ax1.set_title("Ground truth solution")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_box_aspect(1)
    ax1.imshow(usol, vmin=np.min(usol), vmax=np.max(usol), extent=[-1, 1, 1, -1], aspect='auto')
    plt.show()
