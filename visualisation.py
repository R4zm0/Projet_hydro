
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