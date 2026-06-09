"""
afficher_dikau.py
=================
Script d'affichage de la classification de Dikau.
Lance directement depuis PyCharm.

Changer : TERRAIN = "plan" | "plateau" | "sinc_card" | "double_sin" | "reel"
"""

import numpy as np
import matplotlib.pyplot as plt
import fonction_theoriques as ft
import pente
from dikau import Dikau, classer_mnt
import visualisation as vis
import visualisation_dikau as vd
import XY_txt_loader as xy_loader
# ─────────────────────────────────────────────
TERRAIN       = "reel"
FICHIER_REEL  = "morneRouge.txt"
RESOL         = 1.0
EPSILON_P = 0.1 # Eon
EPSILON_C = 0.05 # au lieu de 2e-2
# ─────────────────────────────────────────────

# ===========================================================================
# CHARGEMENT
# ===========================================================================

if TERRAIN == "reel":
    X,Y,Z     = xy_loader.load_z(f"txt/reels/Morne_Rouge/{FICHIER_REEL}", pas=RESOL)
    titre = "Morne Rouge"
else:
    N = 101
    x = np.arange(N, dtype=float)
    y = np.arange(N, dtype=float)
    X_tmp, Y_tmp = np.meshgrid(x, y)
    fn_z, _ = {
        "plan":       (ft.plan,       ft.grad_plan),
        "plateau":    (ft.plateau,    ft.grad_plateau),
        "sinc_card":  (ft.sinc_card,  ft.grad_sinc),
        "double_sin": (ft.double_sin, ft.grad_double_sin),
    }[TERRAIN]
    Z     = fn_z(X_tmp, Y_tmp)
    titre = TERRAIN

    N_rows, N_cols = Z.shape
    x = np.arange(N_cols) * RESOL
    y = np.arange(N_rows) * RESOL
    X, Y = np.meshgrid(x, y)

# ===========================================================================
# CALCULS
# ===========================================================================

print("Calcul des courbures...")
kv, kh, kmin, kmax, slope, G = pente.calculer_courbures_evans2(Z, n = 5)

print("Classification Dikau...")
mat_dikau = classer_mnt(kv, kh, kmin, kmax, slope,
                        epsilon_p=EPSILON_P, epsilon_c=EPSILON_C)

# Stats
total = np.sum(~np.isnan(mat_dikau))
print(f"\nDistribution ({total} pixels classes) :")
for x_cls in Dikau:
    cnt = int(np.sum(mat_dikau == x_cls.value))
    if cnt > 0:
        print(f"  {x_cls.name:20s} : {cnt:5d} ({100*cnt/total:.1f}%)")



# --- DIAGNOSTIC SEUILS ---
print(f"slope  : moy={slope.mean():.4f}  std={slope.std():.4f}")
print(f"kV     : moy={kv.mean():.4f}  std={kv.std():.4f}")
print(f"kH     : moy={kh.mean():.4f}  std={kh.std():.4f}")
print(f"kV p10/p90 : {np.percentile(kv,10):.4f} / {np.percentile(kv,90):.4f}")
print(f"kH p10/p90 : {np.percentile(kh,10):.4f} / {np.percentile(kh,90):.4f}")
# -------------------------

print("Classification Dikau...")

# ===========================================================================
# FIGURE 1 — MNT + courbures + classification
# ===========================================================================

fig1, axes1 = plt.subplots(1, 4, figsize=(22, 6))
fig1.suptitle(f"Dikau evans 2 — {titre}", fontsize=13, fontweight='bold')
 
#EVANS 2 PERMET D'EFFACER LES IRREGULARITES ET FAIRE QQL CHOSE DE PLUS SEGMENTE MOINS DANS LE DETAIL
# MNT
_, axes1[0], im0 = vis.afficher_2D(X, Y, Z, ax=axes1[0],
                                    title="MNT",
                                    cmap='gist_earth',
                                    niveaux=True, n_levels=10,
                                    cotes=False, colorbar=True,
                                    Zname="z [m]",
                                    hillshade=(TERRAIN == "reel"),
                                    vert_exag=4, blend_mode='soft')

# Courbure kV
vd.afficher_courbures(X, Y, kv, kh,
                      ax_kv=axes1[1], ax_kh=axes1[2])

# Classification Dikau
vd.afficher_dikau(X, Y, mat_dikau, ax=axes1[3],
                  title="Classification Dikau",
                  colorbar=True)

plt.tight_layout()
plt.savefig(f"dikau_{titre}.png", dpi=130, bbox_inches='tight')
print(f"\n-> dikau_{titre}.png")

# ===========================================================================
# FIGURE 2 — Distribution des classes
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.suptitle(f"Distribution des classes — {titre}", fontweight='bold')
vd.afficher_distribution_dikau(mat_dikau, ax=ax2)

plt.tight_layout()
plt.savefig(f"dikau_distribution_{titre}.png", dpi=130, bbox_inches='tight')
print(f"-> dikau_distribution_{titre}.png")

plt.show()

#En dessous on fait une map pour visualiser dikau, code des profs pour le visuel

import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
kv, kh, kmin, kmax, slope, G = pente.calculer_courbures_evans2(Z, n = 5)

classif = classer_mnt(kv, kh, kmin, kmax, slope,
                        epsilon_p=EPSILON_P, epsilon_c=EPSILON_C)

noms = [
    "NOSE",            #  1 - éperon convexe
    "SHOULDER_SLOPE",  #  2 - replat convexe
    "HOLLOW_SHOULDER", #  3 - épaulement concave
    "SPUR",            #  4 - crête divergente
    "PLANAR_SLOPE",    #  5 - pente plane
    "HOLLOW",          #  6 - creux convergent
    "SPUR_FOOT",       #  7 - pied d'éperon
    "FOOT_SLOPE",      #  8 - pied de pente
    "HOLLOW_FOOT",     #  9 - bas de pente concave
    "PEAK",            # 10 - sommet
    "RIDGE",           # 11 - crête plate
    "PLAIN",           # 12 - plaine
    "SADDLE",          # 13 - col
    "CHANNEL",         # 14 - chenal
    "PIT",             # 15 - fosse
]

vmin = int(np.nanmin(classif))
vmax = int(np.nanmax(classif))
N = vmax - vmin + 1

# Noms uniquement pour les classes présentes
noms_presents = noms[vmin-1 : vmax]

colors = [
    '#FF0000',  #  1 - NOSE            - rouge vif
    '#FF4500',  #  2 - SHOULDER_SLOPE  - rouge-orangé
    '#FF8C00',  #  3 - HOLLOW_SHOULDER - orange foncé
    '#FFA500',  #  4 - SPUR            - orange
    '#FFD700',  #  5 - PLANAR_SLOPE    - or
    '#ADFF2F',  #  6 - HOLLOW          - vert jaune
    '#7FFF00',  #  7 - SPUR_FOOT       - chartreuse
    '#00FF7F',  #  8 - FOOT_SLOPE      - vert printemps
    '#00FFFF',  #  9 - HOLLOW_FOOT     - cyan
    '#00BFFF',  # 10 - PEAK            - bleu ciel
    '#1E90FF',  # 11 - RIDGE           - bleu dodger
    '#4169E1',  # 12 - PLAIN           - bleu royal
    '#0000CD',  # 13 - SADDLE          - bleu moyen
    '#00008B',  # 14 - CHANNEL         - bleu foncé
    '#000080',  # 15 - PIT             - marine
]
cm = matplotlib.colors.ListedColormap(colors[:N])

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 2, 1)
im = ax.imshow(classif, origin='lower', cmap=cm, vmin=vmin-.5, vmax=vmax+.5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, ticks=range(vmin, vmax+1), cax=cax)
cbar.ax.set_yticklabels(noms_presents)

axe = fig.add_subplot(1, 2, 2, projection='3d')
norm = Normalize(vmin=np.nanmin(classif), vmax=np.nanmax(classif))
my_col = cm(norm(classif))
axe.view_init(elev=35., azim=20)
surf = axe.plot_surface(X, Y, Z, facecolors=my_col, linewidth=0, antialiased=False, rstride=1, cstride=1)
m = plt.cm.ScalarMappable(cmap=cm, norm=norm)
cbar2 = plt.colorbar(m, ax=axe, shrink=.7,
                     ticks=np.linspace(vmin+0.5, vmax-0.5, len(noms_presents)))
cbar2.ax.set_yticklabels(noms_presents)

fig.suptitle('Segmentation')
plt.tight_layout()
plt.show()