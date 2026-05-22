#from pentes import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualisation as vis

# truc à changer 
N_Taille_Grid = 101


# Dimensions des terrains artificiels
x = np.arange(0, N_Taille_Grid)
y = np.arange(0, N_Taille_Grid)
X, Y = np.meshgrid(x, y)


manouvellevariable = "test de branch"
mnt = np.loadtxt('txt/double_sin.txt')


vis.afficher_2D(X, Y, mnt, title="double sin double monstre", Zname="Altitude [m]", niveaux=True, n_levels=5, cotes=True, cmap="viridis")
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