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

# ajout obligatoire de sous-graphes
ax1  = fig.add_subplot(3, 6, 1)
ax2  = fig.add_subplot(3, 6, 2)
ax3  = fig.add_subplot(3, 6, 3)
ax4  = fig.add_subplot(3, 6, 4)
ax5  = fig.add_subplot(3, 6, 5)
ax6  = fig.add_subplot(3, 6, 6)
ax7  = fig.add_subplot(3, 6, 7)
ax8  = fig.add_subplot(3, 6, 8)
ax9  = fig.add_subplot(3, 6, 9)

# construction des sous-graphes
_ = ax1.hist(randn(1000), bins=20, color='b', alpha=0.3)
ax1.set_title('Titre figure 1')

_ = ax2.scatter(np.arange(30), np.arange(30) + 3 * randn.random(30), color='g')

_ = ax3.plot(x, y, linestyle = "--", color = 'g')

_ = ax4.plot(x, y, linestyle = "-", color = 'r', marker = 'o')

_ = ax5.plot(x, y, linestyle = "-", color = 'c', drawstyle = 'steps-post')
ax5.set_title('Titre figure 5')

_ = ax6.plot(randn(100).cumsum(), color = 'y')
ax6.set_title('Titre figure 6')

ax7.set_xticks([0, 25, 50, 75, 100])
ax7.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
_ = ax7.plot(randn.random(101).cumsum(), linestyle = "--", color = 'm', label = 'un')
_ = ax7.plot(randn.random(101).cumsum(), linestyle = "-", color = 'b', label = 'deux')
_ = ax7.legend(loc = 'best')

# ===================================
# construction de figures avec pandas
# ===================================

serie = pd.Series(np.random.random(10).cumsum(), index=np.arange(0, 100, 10))
_ = ax8.plot(serie, color = 'r', linestyle = "--")
_ = ax8.set_title('Titre figure 8')

df = pd.DataFrame(np.random.random(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
_ = ax9.plot(df)
_ = ax9.set_title('Titre figure 9')

ax10 = fig.add_subplot(3, 6, 10)
ax10.set_title('Titre figure 10')
df.boxplot()
#df.plot(kind='box')  # ne fonctionne pas !

ax11 = fig.add_subplot(3, 6, 11)
ax11.set_title('Titre figure 11')
serie.plot(kind = 'bar', ax=plt.gca())

ax12 = fig.add_subplot(3, 6, 12)
ax12.set_title('Titre figure 12')
df.plot(kind = 'bar', ax=plt.gca())

df = pd.DataFrame(np.random.rand(6, 4), index=['one', 'two', 'three', 'four', 'five', 'six'], columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
ax13 = fig.add_subplot(3, 6, 13)
ax13.set_title('Titre figure 13')
sum_by_line = df.sum(1)
df_rescaled = df.div(sum_by_line, axis = 0)
df_rescaled.plot(kind='barh', stacked=True, ax=plt.gca())

series = pd.Series(np.random.randn(100))
ax14 = fig.add_subplot(3, 6, 14)
ax14.set_title('Titre figure 14')
series.hist(bins=5, color = 'c')

x = np.random.randn(100)
y = np.random.randn(100)
ax15 = fig.add_subplot(3, 6, 15)
ax15.set_title('Titre figure 15')
plt.scatter(x,y)

# construction d'un scatter plot entre 4 variables avec pandas sur une nouvelle figure
figure_2 = plt.figure()
df = pd.DataFrame(np.random.rand(100, 4), columns=pd.Index(['v1', 'v2', 'v3', 'v4']))
pd.scatter_matrix(df, diagonal='kde', color='k', ax=plt.gca())

# autres essais avec pandas
figure_3 = plt.figure()

# une time series
ax = figure_3.add_subplot(2, 3, 1)
ax.set_title('time series')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot(ax=plt.gca())

# histogrammes superposés
ax = figure_3.add_subplot(2, 3, 2)
ax.set_title('3 histogrammes superposes')
df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000), 'c': randn(1000) - 1}, columns=['a', 'b', 'c'])
df4.plot(kind='hist', alpha=0.5, ax=plt.gca())

# un box plot
ax = figure_3.add_subplot(2, 3, 3)
ax.set_title('un box plot')
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
colors = dict(boxes='DarkGreen', whiskers='DarkOrange',  medians='DarkBlue', caps='Gray')
df.plot(kind='box', color = colors, ax=plt.gca())

# scatter plots de plusieurs groupes
ax = figure_3.add_subplot(2, 3, 4)
ax.set_title('scatters plots superposes')
df_1 = pd.DataFrame(np.random.randn(50, 2), columns=['a', 'b'])
df_2 = pd.DataFrame(np.random.randn(50, 2), columns=['c', 'd'])
df_1.plot(kind='scatter', x='a', y='b', color='Blue', label='Group 1', ax=plt.gca())
df_2.plot(kind='scatter', x='c', y='d', color='Green', label='Group 2', ax=plt.gca())

# un histogramme simple
ax = figure_3.add_subplot(2, 3, 5)
ax.set_title('histogramme simple')
series = pd.Series(np.random.rand(5), index=list('abcde'))
series.plot(kind='bar', color='k', alpha=0.7 )

# ajustement des espacements entre sous graphes et affichage
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

print("fini")

"""
couleurs
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white

niveaux de gris
color = '0.75'

code
color = '#eeefff'
"""

"""
style de ligne
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
"""






