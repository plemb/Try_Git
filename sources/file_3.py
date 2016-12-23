# -*- coding: utf-8 -*-
__author__ = 'plemberger'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import randn

x = [3,2,1,0]
y = [1,2,4,3]
data = randn.random(30).cumsum()

# =======================================
# construction de figures avec matplotlib
# =======================================

fig = plt.figure()

# ajout obligatoire de sous-graphes blabla
ax1  = fig.add_subplot(3, 6, 1)
ax2  = fig.add_subplot(3, 6, 2)
ax3  = fig.add_subplot(3, 6, 3)
ax4  = fig.add_subplot(3, 6, 4)
ax5  = fig.add_subplot(3, 6, 5)
ax6  = fig.add_subplot(3, 6, 6)
ax7  = fig.add_subplot(3, 6, 7)
ax8  = fig.add_subplot(3, 6, 8)
ax9  = fig.add_subplot(3, 6, 9)