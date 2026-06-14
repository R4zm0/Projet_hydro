"""
methode_bpi_comparaison.py
==========================
Compare les methodes de BPI de pente.py sur Morne Rouge.

- Une FENETRE par methode (rectangle, disque, anneau, secteur adaptatif).
- Echelle / colorbar PARTAGEE entre toutes les fenetres : une seule
  normalisation centree calculee sur l'ensemble des cartes BPI.
- Rectangle / disque : 1 parametre -> une ligne de cartes.
- Anneau            : 2 parametres -> grille r_inner x r_outer.
- Secteur adaptatif : 2 parametres -> grille rayon x ouverture angulaire.

A placer et lancer dans attendus_pedago/ (a cote de Methode_Bpi.py).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

import pente as p
import visualisation as vis
import XY_txt_loader as xy_loader

# ── Chargement Morne Rouge ────────────────────────────────────────────────
base = os.path.dirname(os.path.dirname(__file__))
X, Y, Z = xy_loader.load_z(
    os.path.join(base, 'txt', 'reels', 'Morne_Rouge', 'morneRouge.txt'))

# Aspect local pour le BPI sectoriel adaptatif (convention bearing_rad = sortie _aspect)
G   = p.gradient_fcn(Z)
ASP = p._aspect(G[..., 0], G[..., 1])

CMAP   = 'RdBu_r'   # rouge = au-dessus des voisins (crete), bleu = en-dessous (depression)
N_BINS = 16         # discretisation angulaire du secteur adaptatif (±11°)

# ── Calcul de toutes les cartes BPI ───────────────────────────────────────

# Rectangle (1 parametre : taille)
rect_sizes = [3, 5, 7, 9, 11]
rect_maps  = [(f"{s}x{s}", p.bpi_rectangle(Z, s, s)) for s in rect_sizes]

# Disque (1 parametre : rayon)
disq_radii = [3, 5, 7, 10, 13]
disq_maps  = [(f"r={r}", p.bpi_disque(Z, radius=r)) for r in disq_radii]

# Anneau (2 parametres : r_inner x r_outer)
ann_inner = [1, 2, 3]
ann_outer = [5, 8, 11]
ann_grid  = [[(f"[{ri}, {ro}]", p.bpi_anneau(Z, r_inner=ri, r_outer=ro))
              for ro in ann_outer] for ri in ann_inner]

# Secteur adaptatif (2 parametres : rayon x ouverture)
sec_radii  = [5, 10, 15]
sec_widths = [(np.pi / 4, "pi/4"), (np.pi / 2, "pi/2"), (np.pi, "pi")]
sec_grid   = [[(f"r={r}, {wlbl}",
                p.bpi_sector_adaptive(Z, ASP, radius=r, angle_width=w,
                                      n_bins=N_BINS,
                                      aspect_convention='bearing_rad'))
               for (w, wlbl) in sec_widths] for r in sec_radii]

# ── Normalisation commune a TOUTES les cartes ─────────────────────────────
all_maps  = ([m for _, m in rect_maps]
             + [m for _, m in disq_maps]
             + [m for row in ann_grid for _, m in row]
             + [m for row in sec_grid for _, m in row])

all_vals  = np.concatenate([np.abs(m[np.isfinite(m)]).ravel() for m in all_maps])
HALFRANGE = float(np.percentile(all_vals, 98))   # bornes robustes partagees
NORM      = CenteredNorm(vcenter=0, halfrange=HALFRANGE)

print(f"Echelle commune : BPI dans [-{HALFRANGE:.3f}, +{HALFRANGE:.3f}] m (percentile 98%)")


# ── Helpers d'affichage ───────────────────────────────────────────────────

def fenetre_ligne(titre, maps):
    """Methode a 1 parametre -> une ligne de cartes."""
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (lbl, m) in zip(axes, maps):
        # afficher_2D renvoie le mappable im ; on le reutilise pour la
        # colorbar partagee (toutes les cartes ont la meme NORM/CMAP)
        _, _, im = vis.afficher_2D(X, Y, m, ax=ax, title=lbl, cmap=CMAP,
                                   norm=NORM, niveaux=False, colorbar=False)
    fig.suptitle(titre, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=list(axes),
                 label="BPI [m]  (echelle commune)", shrink=0.85)
    return fig


def fenetre_grille(titre, grid):
    """Methode a 2 parametres -> grille de cartes."""
    nr, nc = len(grid), len(grid[0])
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4 * nr),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for i in range(nr):
        for j in range(nc):
            lbl, m = grid[i][j]
            _, _, im = vis.afficher_2D(X, Y, m, ax=axes[i, j], title=lbl,
                                       cmap=CMAP, norm=NORM,
                                       niveaux=False, colorbar=False)
    fig.suptitle(titre, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=list(axes.ravel()),
                 label="BPI [m]  (echelle commune)", shrink=0.85)
    return fig


# ── Figure de reference : le MNT ──────────────────────────────────────────
fig0, ax0 = plt.subplots(figsize=(7, 6))
vis.afficher_2D(X, Y, Z, ax=ax0, title="MNT Morne Rouge (reference)",
                Zname="profondeur [m]", cmap='gist_earth',
                hillshade=True, vert_exag=4, blend_mode='soft',
                niveaux=True, n_levels=10, cotes=False, colorbar=True)
fig0.tight_layout()

# ── Une fenetre par methode ───────────────────────────────────────────────
fenetre_ligne("BPI Rectangle - Morne Rouge\n"
              "rouge = crete | bleu = depression", rect_maps)

fenetre_ligne("BPI Disque - Morne Rouge\n"
              "rouge = crete | bleu = depression", disq_maps)

fenetre_grille("BPI Anneau - Morne Rouge   (lignes : r_inner, colonnes : r_outer)\n"
               "rouge = crete | bleu = depression", ann_grid)

fenetre_grille("BPI Secteur adaptatif - Morne Rouge   (lignes : rayon, colonnes : ouverture)\n"
               "secteur oriente selon l'aspect local | rouge = crete | bleu = depression",
               sec_grid)

plt.show()