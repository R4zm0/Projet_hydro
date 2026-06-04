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

#Terrain réel

base = os.path.dirname(os.path.dirname(__file__))  # remonte au dossier PROJETHYDRO
mnt = np.loadtxt(os.path.join(base, 'txt', 'reels', 'morneRouge.txt'))

x = np.arange(0, len(mnt[0]), 1)        #taille pour x : taille d'une sous liste qui fait une ligne
y = np.arange(0, len(mnt), 1)       # nombre de lignes ou de sous liste 

X, Y = np.meshgrid(x, y)

G_reel_tpp        = p.gradient_tpp(mnt)
pente_tpp    = p._safe_gradient_norm(G_reel_tpp[..., 0], G_reel_tpp[..., 1])

G_reel_fcn        = p.gradient_fcn(mnt)
pente_fcn    = p._safe_gradient_norm(G_reel_fcn[..., 0], G_reel_fcn[..., 1])

G_reel_evans, _   = p.gradient_evans(mnt)
pente_evans  = p._safe_gradient_norm(G_reel_evans[..., 0], G_reel_evans[..., 1])

erreur_tpp_evans   = (pente_tpp   - pente_evans)[1:-1, 1:-1]
erreur_fcn_tpp  = (pente_fcn   - pente_tpp)[1:-1, 1:-1]
erreur_evans_fcn = (pente_evans - pente_fcn)[1:-1, 1:-1]

#Affichage des erreurs entre méthode

def stats_title(nom, e):
    return f"{nom}\nμ={e.mean():.2e}  σ={e.std():.2e}"

norm_commune = CenteredNorm(0)
cmap = 'seismic'

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

vis.afficher_2D(X, Y, erreur_tpp_evans,   ax=ax[0], title=stats_title('TPP, evans',   erreur_tpp_evans),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)
vis.afficher_2D(X, Y, erreur_fcn_tpp,   ax=ax[1], title=stats_title('FCN, TPP',   erreur_fcn_tpp),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)
vis.afficher_2D(X, Y, erreur_evans_fcn, ax=ax[2], title=stats_title('Evans, FCN', erreur_evans_fcn),
                Zname='Erreur [m/m]', cmap=cmap, norm=norm_commune, niveaux=False, colorbar=True)

plt.suptitle('Erreurs des différentes méthodes de pente pour la carte Morne Rouge  : normalisation partagée')
plt.tight_layout()
plt.show()

#Affichage des histogrammes de diff entre méthodes

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

vis.afficher_histogramme(erreur_tpp_evans, ax=ax[0],
                             title=f"Histogramme des différences de méthode de tpp/evans ", Zname=f"Erreur [m]",
                             bins=50, density=False, color="steelblue", edgecolor="white", alpha=0.8)
vis.afficher_histogramme(erreur_fcn_tpp, ax=ax[1],
                             title=f"Histogramme des différences de méthode de fcn/tpp", Zname=f"Erreur [m]",
                             bins=50, density=False, color="steelblue", edgecolor="white", alpha=0.8)
vis.afficher_histogramme(erreur_evans_fcn, ax=ax[2],
                             title=f"Histogramme des différences de méthode de evans/fcn", Zname=f"Erreur [m]",
                             bins=50, density=False, color="steelblue", edgecolor="white", alpha=0.8)

