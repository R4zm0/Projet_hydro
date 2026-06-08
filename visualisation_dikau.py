"""
visualisation_dikau.py
======================
Fonctions d'affichage pour la classification de Dikau et les courbures.
Suit la meme regle que visualisation.py :
    - uniquement des definitions de fonctions
    - pas de code qui s'execute a l'import
    - seulement des ax.*, jamais de plt.*
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dikau import Dikau


# ===========================================================================
# Palette et noms des classes Dikau (utilises par toutes les fonctions)
# ===========================================================================

DIKAU_COLORS = [
    '#8B0000', '#CD5C5C', '#FF6347',   # NOSE, SHOULDER_SLOPE, HOLLOW_SHOULDER
    '#228B22', '#90EE90', '#006400',   # SPUR, PLANAR_SLOPE, HOLLOW
    '#4169E1', '#87CEEB', '#000080',   # SPUR_FOOT, FOOT_SLOPE, HOLLOW_FOOT
    '#FFD700', '#FFA500', '#FFFFE0',   # PEAK, RIDGE, PLAIN
    '#9370DB', '#00CED1', '#2F4F4F',   # SADDLE, CHANNEL, PIT
]

DIKAU_VALS  = [x.value for x in Dikau]
DIKAU_NOMS  = [x.name  for x in Dikau]
VMIN_D      = min(DIKAU_VALS)
VMAX_D      = max(DIKAU_VALS)
N_CLASSES   = VMAX_D - VMIN_D + 1
CMAP_DIKAU  = mcolors.ListedColormap(DIKAU_COLORS[:N_CLASSES])
NORM_DIKAU  = mcolors.Normalize(vmin=VMIN_D - 0.5, vmax=VMAX_D + 0.5)


# ===========================================================================
# Utilitaire interne
# ===========================================================================

def _colorbar(fig, im, ax, label, zero_line=False):
    """Colorbar avec make_axes_locatable."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb  = fig.colorbar(im, cax=cax, label=label)
    if zero_line:
        cb.ax.axhline(0, color='black', lw=1.2, linestyle='--')
    return cb


# ===========================================================================
# 1. Carte de classification Dikau
# ===========================================================================

def afficher_dikau(X, Y, mat_dikau, ax=None, title="Classification de Dikau",
                   colorbar=True):
    """
    Affiche la carte de classification de Dikau.

    Parametres
    ----------
    X, Y      : np.ndarray  meshgrid (2D)
    mat_dikau : np.ndarray  carte des classes (N, M), valeurs entières Dikau
    ax        : Axes        axe cible (cree si None)
    title     : str         titre du graphique
    colorbar  : bool        afficher la colorbar avec les noms des classes

    Retourne
    --------
    fig, ax, im
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    ext = (X.min(), X.max(), Y.min(), Y.max())

    # Remplacer NaN par une valeur hors palette pour l'affichage
    data = np.where(np.isnan(mat_dikau), VMIN_D - 1, mat_dikau)

    im = ax.imshow(data, origin='lower', extent=ext,
                   cmap=CMAP_DIKAU, norm=NORM_DIKAU)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb  = fig.colorbar(im, cax=cax, ticks=range(VMIN_D, VMAX_D + 1))
        cb.ax.set_yticklabels(DIKAU_NOMS, fontsize=7.5)

    return fig, ax, im


# ===========================================================================
# 2. Cartes des courbures kV et kH
# ===========================================================================

def afficher_courbures(X, Y, kv, kh, ax_kv=None, ax_kh=None,
                       title_kv="Courbure verticale kV\n(+ concave | - convexe)",
                       title_kh="Courbure horizontale kH\n(+ convergent | - divergent)",
                       colorbar=True):
    """
    Affiche kV et kH cote a cote avec palette divergente centree sur 0.

    Parametres
    ----------
    X, Y   : np.ndarray  meshgrid (2D)
    kv     : np.ndarray  courbure verticale   (N, M)
    kh     : np.ndarray  courbure horizontale (N, M)
    ax_kv  : Axes        axe pour kV (cree si None)
    ax_kh  : Axes        axe pour kH (cree si None)
    title_kv, title_kh : str  titres
    colorbar : bool      afficher les colorbars

    Retourne
    --------
    fig, ax_kv, ax_kh, im_kv, im_kh
    """
    if ax_kv is None or ax_kh is None:
        fig, (ax_kv, ax_kh) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax_kv.get_figure()

    ext    = (X.min(), X.max(), Y.min(), Y.max())
    vabs_v = float(np.nanpercentile(np.abs(kv), 98))
    vabs_h = float(np.nanpercentile(np.abs(kh), 98))
    norm_v = TwoSlopeNorm(vmin=-vabs_v, vcenter=0, vmax=vabs_v)
    norm_h = TwoSlopeNorm(vmin=-vabs_h, vcenter=0, vmax=vabs_h)

    im_kv = ax_kv.imshow(kv, origin='lower', extent=ext,
                         cmap='seismic', norm=norm_v)
    im_kh = ax_kh.imshow(kh, origin='lower', extent=ext,
                         cmap='seismic', norm=norm_h)

    for ax, im, titre, label in [
        (ax_kv, im_kv, title_kv, "kV [m⁻¹]"),
        (ax_kh, im_kh, title_kh, "kH [m⁻¹]"),
    ]:
        ax.set_title(titre)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        if colorbar:
            _colorbar(fig, im, ax, label, zero_line=True)

    return fig, ax_kv, ax_kh, im_kv, im_kh


# ===========================================================================
# 3. Distribution des classes (barres horizontales)
# ===========================================================================

def afficher_distribution_dikau(mat_dikau, ax=None,
                                 title="Distribution des classes de Dikau"):
    """
    Affiche un diagramme en barres horizontales du nombre de pixels par classe.

    Parametres
    ----------
    mat_dikau : np.ndarray  carte des classes (N, M)
    ax        : Axes        axe cible (cree si None)
    title     : str         titre

    Retourne
    --------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    total   = np.sum(~np.isnan(mat_dikau))
    counts  = []
    labels  = []
    colors  = []

    for v, n, c in zip(DIKAU_VALS, DIKAU_NOMS, DIKAU_COLORS):
        cnt = int(np.sum(mat_dikau == v))
        if cnt > 0:
            counts.append(cnt)
            labels.append(n)
            colors.append(c)

    y_pos = np.arange(len(counts))
    bars  = ax.barh(y_pos, counts, color=colors,
                    edgecolor='#333333', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Nombre de pixels")
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.4)

    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / total if total > 0 else 0
        ax.text(bar.get_width() + total * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va='center', fontsize=7.5)

    return fig, ax
