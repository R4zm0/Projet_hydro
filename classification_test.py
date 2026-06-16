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
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
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

def classer_mnt_bpi(Z, n=1, bpi_radius=5, rug_size=3,
                    PENTE_FAIBLE=4, PENTE_MOYENNE=15,
                    BPI_NEG=-0.2, BPI_POS=0.9,
                    RUG_FAIBLE=0.05, RUG_FORTE=0.11):
    """
    Classification morphologique à partir du MNT brut.
    Calcule en interne la pente, le BPI et la rugosité.
    """

    # --- 1. Pente et aspect via Evans ---
    G, coeffs = pente.gradient_evans_methode2(Z, n=n)
    slope_rad = np.arctan(np.sqrt(G[..., 0]**2 + G[..., 1]**2))
    slope_deg = np.degrees(slope_rad)
    aspect    = np.arctan2(G[..., 1], G[..., 0])  # convention trig

    # --- 2. BPI (secteur adaptatif) ---
    bpi = pente.bpi_sector_adaptive(Z, aspect, radius=bpi_radius)


    # Normalisation du BPI entre -1 et 1
    bpi_std = np.nanstd(bpi)
    if bpi_std > 0:
        bpi = bpi / bpi_std

    # --- 3. Rugosité ---
    rug = pente.rugosite_lisse(Z, size=rug_size)
    print(rug)
    for nom, tab in [("Pente (deg)", slope_deg), ("BPI (normalisé)", bpi), ("Rugosité", rug)]:
        q1, q3 = np.nanpercentile(tab, [25, 75])
        print(f"{nom:18s} Q1={q1:8.4f}  Q3={q3:8.4f}")

    # --- 4. Masques ---
    p_f = slope_deg < PENTE_FAIBLE
    p_m = (slope_deg >= PENTE_FAIBLE) & (slope_deg < PENTE_MOYENNE)
    p_s = slope_deg >= PENTE_MOYENNE

    bpi_n = bpi < BPI_NEG
    bpi_p = bpi > BPI_POS
    bpi_0 = ~bpi_n & ~bpi_p

    rug_f = rug < RUG_FAIBLE
    rug_s = rug >= RUG_FORTE
    rug_m = ~rug_f & ~rug_s

    # --- 5. Classification ---
    classes = np.full(Z.shape, np.nan)

    # Pente faible
    classes[p_f & bpi_0 & rug_f]           = 1   # FLAT_PLAIN
    classes[p_f & bpi_0 & (rug_m | rug_s)] = 2   # FLAT_ROUGH
    classes[p_f & bpi_n]                   = 3   # FLAT_DEPRESSION
    classes[p_f & bpi_p & rug_f]           = 4   # FLAT_MOUND
    classes[p_f & bpi_p & (rug_m | rug_s)] = 5   # FLAT_ROUGH_MOUND

    # Pente moyenne
    classes[p_m & bpi_0 & rug_f]           = 6   # SLOPE_REGULAR
    classes[p_m & bpi_0 & (rug_m | rug_s)] = 7   # SLOPE_ROUGH
    classes[p_m & bpi_p]                   = 8   # SLOPE_CREST
    classes[p_m & bpi_n & rug_f]           = 9   # SLOPE_VALLEY
    classes[p_m & bpi_n & (rug_m | rug_s)] = 10  # SLOPE_GULLY

    # Pente forte
    classes[p_s & bpi_0 & rug_f]           = 11  # STEEP_SLOPE
    classes[p_s & bpi_0 & (rug_m | rug_s)] = 12  # STEEP_ROUGH
    classes[p_s & bpi_p]                   = 13  # STEEP_CREST
    classes[p_s & bpi_n & rug_f]           = 14  # STEEP_VALLEY
    classes[p_s & bpi_n & (rug_m | rug_s)] = 15  # STEEP_GULLY

    return classes


noms = [
    "FLAT_PLAIN",       #  1
    "FLAT_ROUGH",       #  2
    "FLAT_DEPRESSION",  #  3
    "FLAT_MOUND",       #  4
    "FLAT_ROUGH_MOUND", #  5
    "SLOPE_REGULAR",    #  6
    "SLOPE_ROUGH",      #  7
    "SLOPE_CREST",      #  8
    "SLOPE_VALLEY",     #  9
    "SLOPE_GULLY",      # 10
    "STEEP_SLOPE",      # 11
    "STEEP_ROUGH",      # 12
    "STEEP_CREST",      # 13
    "STEEP_VALLEY",     # 14
    "STEEP_GULLY",      # 15
]

classif = classer_mnt_bpi(Z)


vmin = int(np.nanmin(classif))
vmax = int(np.nanmax(classif))
N = vmax - vmin + 1

# Noms uniquement pour les classes présentes
noms_presents = noms[vmin-1 : vmax]

colors = [
    "#02021E",  #  1 - FLAT_PLAIN       - marine
    '#00008B',  #  2 - FLAT_ROUGH       - bleu foncé
    '#0000CD',  #  3 - FLAT_DEPRESSION  - bleu moyen
    '#4169E1',  #  4 - FLAT_MOUND       - bleu royal
    '#1E90FF',  #  5 - FLAT_ROUGH_MOUND - bleu dodger
    '#00BFFF',  #  6 - SLOPE_REGULAR    - bleu ciel
    '#00FFFF',  #  7 - SLOPE_ROUGH      - cyan
    '#00FF7F',  #  8 - SLOPE_CREST      - vert printemps
    '#7FFF00',  #  9 - SLOPE_VALLEY     - chartreuse
    '#ADFF2F',  # 10 - SLOPE_GULLY      - vert jaune
    '#FFD700',  # 11 - STEEP_SLOPE      - or
    '#FFA500',  # 12 - STEEP_ROUGH      - orange
    '#FF8C00',  # 13 - STEEP_CREST      - orange foncé
    '#FF4500',  # 14 - STEEP_VALLEY     - rouge-orangé
    '#FF0000',  # 15 - STEEP_GULLY      - rouge vif
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