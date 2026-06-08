import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import XY_txt_loader as xy_loader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

import fonction_theoriques as ft
import pente as p
import visualisation as vis




base = os.path.dirname(os.path.dirname(__file__))  # remonte au dossier PROJETHYDRO

X,Y,Z = xy_loader.load_z(os.path.join(base, 'txt', 'reels', 'Morne_Rouge', 'morneRouge.txt'))


G = p.gradient_fcn(Z)

ASP = p._aspect(G[...,0], G[...,1])

BPI = p.bpi_sector_adaptive(Z,ASP, aspect_convention = 'geo_rad', n_bins= 6, angle_width=2*np.pi)

vis.afficher_2D(X, Y, BPI, title='BPI', Zname='BPI qui regarde au bon endroit', cmap='viridis', hillshade=False, niveaux=True, colorbar=True, cotes = True, n_levels=1)

plt.show()



