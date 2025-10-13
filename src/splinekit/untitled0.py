#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:45:42 2025

@author: pthevena
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_video(n):
    global X
    X = np.random.binomial(1, 0.3, size = (n, n))

    fig = plt.figure()
    im = plt.imshow(X, cmap = plt.cm.gray)

    def animate(t):
        global X
        # X = np.roll(X, +1, axis = 0)
        X = np.random.binomial(1, 0.3, size = (n, n))
        im.set_array(X)
        return im, 

    anim = FuncAnimation(
        fig,
        animate,
        cache_frame_data = False,
        frames = None,
        interval = 500,
        blit = True
    )

    plt.show()

    return anim

anim = create_video(11)
