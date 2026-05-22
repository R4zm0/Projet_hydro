#from pentes import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# truc à changer 
N_Taille_Grid = 101


# Dimensions des terrains artificiels
x = np.arange(0, N_Taille_Grid)
y = np.arange(0, N_Taille_Grid)
X, Y = np.meshgrid(x, y)



mnt = np.loadtxt('txt/double_sin.txt')


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
col = plt.get_cmap('gist_earth')(mnt)


ax.plot_surface(X, Y, mnt, cmap = 'viridis')
ax.contour(X, Y, mnt, zdir='z', offset=np.min(mnt), cmap='viridis')


print(f"Stats du fichier : moy{np.average(mnt)}, min : {np.min(mnt)}, max : {np.max(mnt)}, ecartype : {np.std(mnt, ddof=1)}")
plt.show()


plt.hist(
    mnt.flatten(),
    bins=100,           # nombre de barres
    density=False,     # True = densité (aire = 1)
    alpha=0.7,         # transparence
    edgecolor='blue'  # bordures des barres
)
plt.title("Histogramme des données")
plt.xlabel("Valeurs")
plt.ylabel("Fréquence")
plt.legend()
plt.grid(True)
plt.show()