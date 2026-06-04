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
mnt = np.loadtxt(os.path.join(base, 'txt', 'double_sin.txt'))


G_reel       = ft.grad_double_sin(X, Y)
pente_reel   = p._safe_gradient_norm(G_reel[..., 0], G_reel[..., 1])

G_tpp        = p.gradient_tpp(mnt)
pente_tpp    = p._safe_gradient_norm(G_tpp[..., 0], G_tpp[..., 1])

G_fcn        = p.gradient_fcn(mnt)
pente_fcn    = p._safe_gradient_norm(G_fcn[..., 0], G_fcn[..., 1])

G_evans, _   = p.gradient_evans(mnt)
pente_evans  = p._safe_gradient_norm(G_evans[..., 0], G_evans[..., 1])

erreur_tpp   = (pente_tpp   - pente_reel)[1:-1, 1:-1]
erreur_fcn   = (pente_fcn   - pente_reel)[1:-1, 1:-1]
erreur_evans = (pente_evans - pente_reel)[1:-1, 1:-1]

def stats_title(nom, e):
    return f"{nom}\nμ={e.mean():.2e}  σ={e.std():.2e}"

norm_commune = CenteredNorm(0)
cmap = 'seismic'

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

vis.afficher_2D(X, Y, erreur_tpp,   ax=ax[0], title=stats_title('TPP',   erreur_tpp),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)
vis.afficher_2D(X, Y, erreur_fcn,   ax=ax[1], title=stats_title('FCN',   erreur_fcn),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)
vis.afficher_2D(X, Y, erreur_evans, ax=ax[2], title=stats_title('Evans', erreur_evans),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)

plt.suptitle('Erreurs ( Pente Analytique - Pente Réelle ) pour la fonction double sinus  : normalisation partagée')
plt.tight_layout()
plt.show()