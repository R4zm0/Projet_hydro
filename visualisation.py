
#cette lib sert à l'affiche.

# les fonctions de la forme Afficher_Qlqchose sont des fonctions d'affichage:
# elles prennent en argument deux mesh grid : X, Y et un np array Z de même dimension que X et Y
# elles returnerons des subplots à afficher, on EVITERA les plt.show() dans ces fonctions, et les plt.Qlqchose() tout court !
# on fera les plt.show () après avoir appeler les fonctions d'affichage pour afficher les subplots côte à côte !
#


# Regle simple :  -----------------------------------------------

#   Dans les fonction :  seulement ax.*, QUE DES ax.Qlqchose() PAS DE PLT.Qlqchose() 

#   Fin du script ailleur dans un autre fichier  : plt.* comme plt.show() ou plt.tight_layout() 
# -----------------------------------------------  ----------------------------------------------- 



# prend exemple sur afficher 2D pour la forme attendu d'une fonction d'affichage

import matplotlib
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def afficher_2D(X, Y, Z, title = "Z", niveaux = True, n_levels = 10, cotes = True, cmap = "viridis", Zname = "Z"):
    """
        Paramètres
        ----------
        X, Y : np.ndarray   meshgrid (2D)
        Z    : np.ndarray   valeurs à afficher (même shape que X, Y)
        
        title   : str       titre du graphique
        Zname : str       nom de la variable Z pour la colorbar
        

        niveaux : Bool      Si oui ou non on veut des courbes de niveaux
        n_levels: int       nombre de lignes de niveau
        cotes :   Bool      Si oui ou non on veut des cotes d'altitudes ")

        cmap    : str       colormap matplotlib
          
        Retourne
        --------
        fig, ax : Figure et Axes matplotlib
        on peut alors tout de même encore modifié le subplot retourné en faisant par exemple : ax.set_title("mon titre") ou ax.set_xlabel("x [m]") ou ax.set_ylabel("y [m]")
    """
    fig, ax = plt.subplots()
    im = ax.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    fig.colorbar(im, ax=ax, label=Zname)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if niveaux:
        contours = ax.contour(X, Y, Z, levels=n_levels, colors="black", linewidths=0.5)
        if cotes:
            ax.clabel(contours, inline=True, fontsize=8)
            
    return fig, ax

def afficher_gradient(X, Y, G, ax=None, step=10, color="red", scale=None, Pointe_vers_max = False):
    
    "afficher les gradient sur une figure déjà existante, G doit être de shape (X.shape[0], X.shape[1], 2) et correspondre au gradient vectoriel de Z, c'est à dire G[:,:,0] = dZ/dx et G[:,:,1] = dZ/dy"
    
    """scale=None   # automatique
    scale=1      # très longues
    scale=10     # moyennes
    scale=100    # courtes

    Pointe_vers_max = True : les flèches pointent vers la direction de la pente la plus forte (max de Z), les maximums locaux
    Pointe_vers_max = False : les flèches pointent vers la direction de la pente la plus faible (min de Z), les minimums locaux
    """
    if not Pointe_vers_max:
        G = -G  # inverser les flèches pour pointer vers les minimums locaux au lieu des maximums locaux

    if ax is None:
        _, ax = plt.subplots()
    
    sl = (slice(None, None, step), slice(None, None, step))
    Gx = G[..., 0]
    Gy = G[..., 1]
    ax.quiver(X[sl], Y[sl], Gx[sl], Gy[sl], color=color, scale=scale)

    return ax
