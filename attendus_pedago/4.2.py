import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

import fonction_theoriques as ft
import pente as p
import visualisation as vis
import XY_txt_loader as load   # ← loader centralisé, plus de np.loadtxt sauvage

base = os.path.dirname(os.path.dirname(__file__))

# ─── utilitaire ──────────────────────────────────────────────────────────────

def stats_title(nom, e):
    return f"{nom}\nμ={e.mean():.2e}  σ={e.std():.2e}"

# slice pour exclure les bords (effet de bord des gradients numériques)
sl = slice(1, -1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Terrain artificiel : double_sin
# ─────────────────────────────────────────────────────────────────────────────

# load_z construit X, Y + charge Z en une seule ligne, pas la peine de refaire
# np.arange / np.meshgrid à la main
X, Y, mnt = load.load_z(os.path.join(base, 'txt', 'double_sin.txt'))

# gradient analytique (référence)
G_th       = ft.grad_double_sin(X, Y)
pente_th   = p._safe_gradient_norm(G_th[..., 0], G_th[..., 1])

# gradients numériques
G_tpp       = p.gradient_tpp(mnt)
pente_tpp   = p._safe_gradient_norm(G_tpp[..., 0], G_tpp[..., 1])

G_fcn       = p.gradient_fcn(mnt)
pente_fcn   = p._safe_gradient_norm(G_fcn[..., 0], G_fcn[..., 1])

G_evans, _  = p.gradient_evans(mnt)
pente_evans = p._safe_gradient_norm(G_evans[..., 0], G_evans[..., 1])

# erreurs (bords exclus)
erreur_tpp   = (pente_tpp   - pente_th)[sl, sl]
erreur_fcn   = (pente_fcn   - pente_th)[sl, sl]
erreur_evans = (pente_evans - pente_th)[sl, sl]

# ← X et Y rognés cohérents avec les erreurs rognées :
#   afficher_2D utilise imshow(extent=...) + contour(X, Y, Z)
#   donc X/Y et Z DOIVENT avoir la même shape si niveaux=True
Xc, Yc = X[sl, sl], Y[sl, sl]

norm_ds = CenteredNorm(0)   # normalisation partagée pour cette figure
cmap    = 'seismic'

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
vis.afficher_2D(Xc, Yc, erreur_tpp,   ax=ax[0],
                title=stats_title('TPP',   erreur_tpp),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_ds,
                niveaux=False, colorbar=True)
vis.afficher_2D(Xc, Yc, erreur_fcn,   ax=ax[1],
                title=stats_title('FCN',   erreur_fcn),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_ds,
                niveaux=False, colorbar=True)
vis.afficher_2D(Xc, Yc, erreur_evans, ax=ax[2],
                title=stats_title('Evans', erreur_evans),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_ds,
                niveaux=False, colorbar=True)
plt.suptitle('Erreurs (pente numérique − analytique) — double_sin'
             '\nnormalisation partagée')
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Terrain réel : Morne Rouge — comparaison inter-méthodes
# ─────────────────────────────────────────────────────────────────────────────

# load_z gère la construction du meshgrid à partir des dimensions du fichier ;
# plus besoin de len(mnt[0]) / len(mnt) à la main
X_mr, Y_mr, mnt_mr = load.load_z(
    os.path.join(base, 'txt', 'reels', 'Morne_Rouge', 'morneRouge.txt'))

G_tpp_mr        = p.gradient_tpp(mnt_mr)
pente_tpp_mr    = p._safe_gradient_norm(G_tpp_mr[..., 0],   G_tpp_mr[..., 1])

G_fcn_mr        = p.gradient_fcn(mnt_mr)
pente_fcn_mr    = p._safe_gradient_norm(G_fcn_mr[..., 0],   G_fcn_mr[..., 1])

G_evans_mr, _   = p.gradient_evans(mnt_mr)
pente_evans_mr  = p._safe_gradient_norm(G_evans_mr[..., 0], G_evans_mr[..., 1])

erreur_tpp_evans  = (pente_tpp_mr   - pente_evans_mr)[sl, sl]
erreur_fcn_tpp    = (pente_fcn_mr   - pente_tpp_mr  )[sl, sl]
erreur_evans_fcn  = (pente_evans_mr - pente_fcn_mr  )[sl, sl]

Xc_mr, Yc_mr = X_mr[sl, sl], Y_mr[sl, sl]   # grille rognée cohérente

norm_mr = CenteredNorm(0)   # normalisation partagée pour cette figure

series = [
    ('TPP − Evans', erreur_tpp_evans),
    ('FCN − TPP',   erreur_fcn_tpp),
    ('Evans − FCN', erreur_evans_fcn),
]

# cartes d'erreur
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for col, (label, erreur) in enumerate(series):
    vis.afficher_2D(Xc_mr, Yc_mr, erreur, ax=ax[col],
                    title=stats_title(label, erreur),
                    Zname='Erreur [m/m]', cmap=cmap, norm=norm_mr,
                    niveaux=False, colorbar=True)
plt.suptitle('Différences inter-méthodes — Morne Rouge\nnormalisation partagée')
plt.tight_layout()

# histogrammes individuels
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for col, (label, erreur) in enumerate(series):
    vis.afficher_histogramme(erreur, ax=ax[col],
                             title=label, Zname='Erreur [m/m]',
                             bins=50, density=False,
                             color='steelblue', edgecolor='white', alpha=0.8)
plt.suptitle('Histogrammes des différences inter-méthodes — Morne Rouge')
plt.tight_layout()

# histogrammes superposés (comparaison directe des formes)
from matplotlib.patches import Patch
COULEURS = {'TPP − Evans': '#E07030', 'FCN − TPP': '#2E8B57', 'Evans − FCN': '#1A5EA8'}

fig, ax = plt.subplots(figsize=(8, 5))
for label, erreur in series:
    vis.afficher_histogramme(erreur, ax=ax,
                             title='Distributions des erreurs inter-méthodes — Morne Rouge',
                             Zname='Erreur [m/m]',
                             bins=50, density=True,
                             color=COULEURS[label], edgecolor='none', alpha=0.5,
                             show_moyenne=False, show_mediane=False, show_std=False,
                             show_min=False, show_max=False)
ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l)
                   for l, c in COULEURS.items()])
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Comparaison des deux variantes Evans — Morne Rouge
# ─────────────────────────────────────────────────────────────────────────────

# mnt_mr et X_mr/Y_mr sont déjà chargés plus haut, pas besoin de recharger
G_evans2_mr, _  = p.gradient_evans_methode2(mnt_mr)
pente_evans2_mr = p._safe_gradient_norm(G_evans2_mr[..., 0], G_evans2_mr[..., 1])

erreur_evans_meth = (pente_evans_mr - pente_evans2_mr)[sl, sl]

norm_ev = CenteredNorm(0)

fig, ax = plt.subplots(figsize=(7, 6))
vis.afficher_2D(Xc_mr, Yc_mr, erreur_evans_meth, ax=ax,
                title=stats_title('Evans (formules) − Evans (moindres carrés)',
                                  erreur_evans_meth),
                Zname='Erreur [m/m]', norm=norm_ev, cmap=cmap,
                niveaux=False, colorbar=True)
plt.suptitle('Comparaison des deux variantes Evans — Morne Rouge')
plt.tight_layout()

plt.show()