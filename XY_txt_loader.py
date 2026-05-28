#from pentes import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualisation as vis
import fonction_theoriques as ft
import pente as pente
# truc à changer 
N_Taille_Grid = 101


# Dimensions des terrains artificiels
x = np.arange(0, N_Taille_Grid)
y = np.arange(0, N_Taille_Grid)
X, Y = np.meshgrid(x, y)


manouvellevariable = "test de branch"
mnt = np.loadtxt('txt/double_sin.txt')
Z = mnt

G, coeffs = pente.gradient_evans(Z)
print("shape de G : ", G.shape)
print(G)

fig, axes = plt.subplots(1, 2)

_, sinusplot, im = vis.afficher_2D(X, Y, Z, ax=axes[0],
                                    title="double sin double monstre",
                                    Zname="Altitude [m]", niveaux=True,
                                    n_levels=5, cotes=True, cmap="viridis")

vis.afficher_gradient(X, Y, G, ax=sinusplot, step=5, color="red", scale=20)

_, sinusplotshade, _ = vis.afficher_2D(X, Y, Z, ax=axes[1],
                                        title="double sin hillshade",
                                        Zname="Altitude [m]", niveaux=True,
                                        n_levels=5, cotes=True, cmap="terrain",
                                        hillshade=True, vert_exag=4, blend_mode='soft')

vis.afficher_gradient(X, Y, G, ax=sinusplotshade, step=5, color="red", scale=20)
plot_axes = fig.axes  # capture avant la colorbar
fig.colorbar(im, ax=plot_axes, label="Altitude [m]", shrink=0.65)  # ← après, pour pas se faire écraser

plt.tight_layout()
fig.subplots_adjust(right=0.75)  # ← après, pour pas se faire écraser

plt.show()