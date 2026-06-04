import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import XY_txt_loader as load

import numpy as np
import visualisation as vis
import matplotlib.pyplot as plt

X, Y, Z = load.load_z('txt/reels/bertheaume_z.txt')

# --- Cas 2 : Basse Saint Pierre ---
X_bsp, Y_bsp, Z_bsp = load.load_xyz_separes(
    'txt/reels/Basse_Saint_Pierre/x_basse_St_Pierre.txt',
    'txt/reels/Basse_Saint_Pierre/y_basse_St_Pierre.txt',
    'txt/reels/Basse_Saint_Pierre/z_basse_St_Pierre.txt',
)
X_bsp_rel = X_bsp 
Y_bsp_rel = Y_bsp 

# --- Cas 3 : Lézardrieux ---
X_lez, Y_lez, Z_lez = load.load_z('txt/reels/lezardrieux_z.txt')

# --- Cas 4 : Mzouazia ---
X_mzo, Y_mzo, Z_mzo = load.load_z('txt/reels/mzouazia_z.txt')


fig, ax = plt.subplots(2, 2, figsize=(10, 8),
                       gridspec_kw={'width_ratios': [2, 1]})

terrains = [
    (X,         Y,         Z,        "Bertheaume"),
    (X_bsp_rel, Y_bsp_rel, Z_bsp,    "Basse Saint Pierre : Coorodonées Lamberts"),
]

for i, (Xi, Yi, Zi, nom) in enumerate(terrains):
    vis.afficher_2D(Xi, Yi, Zi, ax=ax[i, 0], title=f"{nom}",
                    Zname="profondeur [m]", niveaux=True, cotes=True,
                    cmap="gist_earth", hillshade=True, n_levels=5, colorbar=True)

    vis.afficher_histogramme(Zi, ax=ax[i, 1],
                             title=f"Histogramme des profondeurs de {nom}", Zname=f"Profondeur [m]",
                             bins=50, density=False, color="steelblue", edgecolor="white", alpha=0.8)

plt.tight_layout()
plt.show()