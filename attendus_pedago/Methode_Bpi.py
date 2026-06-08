import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

import fonction_theoriques as ft
import pente as p
import visualisation as vis



x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)
base = os.path.dirname(os.path.dirname(__file__))  # remonte au dossier PROJETHYDRO
Z = np.loadtxt(os.path.join(base, 'txt', 'Sin_card.txt'))
G = p.gradient_fcn(Z)
ASP = p._aspect(G[...,0], G[...,1])

BPI = p.bpi_sector_adaptive(Z,ASP, aspect_convention = 'geo_rad', n_bins= 6, angle_width=np.pi/4)
vis.afficher_2D(X, Y, BPI, title='BPI', Zname='Exposition en radian mais pas randians', cmap='viridis', hillshade=False, niveaux=True, colorbar=True, cotes = True, n_levels=1)

plt.show()



"""x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)
base = os.path.dirname(os.path.dirname(__file__))  # remonte au dossier PROJETHYDRO
Z = np.loadtxt(os.path.join(base, 'txt', 'sin_card.txt'))

Bpi = p.bpi_anneau(Z, r_outer= 30, r_inner=5)

vis.afficher_2D(X, Y, Bpi, title='BPI', Zname='BPI [m]', cmap='seismic', colorbar=True)


# Dans le contexte BPI (garder uniquement les crêtes hautes)
threshold = np.nanpercentile(Bpi, 96)
bpi_cretes = np.where(Bpi >= threshold, Bpi, np.nan)
vis.afficher_2D(X, Y, bpi_cretes, title='BPI - Crêtes Hautes', Zname='BPI [m]', cmap='seismic', colorbar=True)
plt.show()
"""