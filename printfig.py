"""
fig_4_1_visualisation.py
========================
Génère les figures de la section 4.1 (visualisation des MNT + statistiques).
À placer à la RACINE du projet (à côté de pente.py, visualisation.py, etc.)
puis lancer : python fig_4_1_visualisation.py

Produit :
    - Figure 1 : 4 terrains analytiques en 2D (avec courbes de niveau)
    - Figure 2 : 4 terrains analytiques en 3D
    - Figure 3 : histogrammes des 4 terrains analytiques
    - Figure 5 : Morne Rouge en 3D
    + impression console des statistiques (Tableau 1)

NB : la Figure 4 (Morne Rouge 2D ombré + isobathes + histogramme) est déjà
produite par attendus_pedago/4.1.py.
"""

import numpy as np
import matplotlib.pyplot as plt

import fonction_theoriques as ft
import visualisation as vis
import XY_txt_loader as xy_loader

# --- grille analytique 101x101, 0..100 m, pas 1 m -------------------------
x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

terrains = [
    ("plan",          ft.plan(X, Y)),
    ("plateau",       ft.plateau(X, Y)),
    ("sinus cardinal", ft.sinc_card(X, Y)),
    ("double_sin",    ft.double_sin(X, Y)),
]

# --- statistiques (Tableau 1) ---------------------------------------------
print(f"{'Terrain':14s} {'min':>9s} {'max':>9s} {'moyenne':>9s} {'ecart-type':>11s}")
for nom, Z in terrains:
    print(f"{nom:14s} {Z.min():9.3f} {Z.max():9.3f} {Z.mean():9.3f} {Z.std():11.3f}")

# --- Figure 1 : vues 2D ----------------------------------------------------
fig1, ax1 = plt.subplots(2, 2, figsize=(12, 10))
ax1 = ax1.ravel()
for i, (nom, Z) in enumerate(terrains):
    vis.afficher_2D(X, Y, Z, ax=ax1[i], title=nom, Zname="Z",
                    cmap="gist_earth", niveaux=True, n_levels=10,
                    cotes=True, colorbar=True)
fig1.suptitle("MNT analytiques — vue 2D", fontsize=13, fontweight="bold")
fig1.tight_layout()

# --- Figure 2 : vues 3D ----------------------------------------------------
fig2 = plt.figure(figsize=(13, 10))
for i, (nom, Z) in enumerate(terrains):
    ax = fig2.add_subplot(2, 2, i + 1, projection="3d")
    vis.afficher_3D(X, Y, Z, ax=ax, title=nom, Zname="Z",
                    cmap="gist_earth", colorbar=False)
fig2.suptitle("MNT analytiques — vue 3D", fontsize=13, fontweight="bold")
fig2.tight_layout()

# --- Figure 3 : histogrammes ----------------------------------------------
fig3, ax3 = plt.subplots(2, 2, figsize=(12, 9))
ax3 = ax3.ravel()
for i, (nom, Z) in enumerate(terrains):
    vis.afficher_histogramme(Z, ax=ax3[i], title=f"Histogramme — {nom}",
                             Zname="Z", bins=50)
fig3.suptitle("Distribution des valeurs — terrains analytiques",
              fontsize=13, fontweight="bold")
fig3.tight_layout()

# --- Figure 5 : Morne Rouge en 3D -----------------------------------------
X_mr, Y_mr, Z_mr = xy_loader.load_z("txt/reels/Morne_Rouge/morneRouge.txt", pas=1.0)

print("\nMorne Rouge :")
print(f"  min={np.nanmin(Z_mr):.3f}  max={np.nanmax(Z_mr):.3f}  "
      f"moyenne={np.nanmean(Z_mr):.3f}  ecart-type={np.nanstd(Z_mr):.3f}")

fig5 = plt.figure(figsize=(9, 7))
ax5 = fig5.add_subplot(111, projection="3d")
vis.afficher_3D(X_mr, Y_mr, Z_mr, ax=ax5, title="Morne Rouge — relief 3D",
                Zname="profondeur [m]", cmap="gist_earth", colorbar=True)
fig5.tight_layout()

plt.show()