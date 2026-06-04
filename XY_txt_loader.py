#from pentes import *

import cmocean
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualisation as vis
import fonction_theoriques as ft
import pente as pente
# truc à changer 

"""
load_reel.py  –  Loader pour les MNT réels du projet hydro
===========================================================

Deux fonctions publiques :

    X, Y, Z = load_z(filepath, pas=1.0)
        → pour tous les fichiers "Z seul" : *_z.txt, z_*.txt, *.z, *.xyz (grille)
          - construit un meshgrid X, Y avec un pas métrique donné (défaut 1 m)

    X, Y, Z = load_xyz_separes(x_file, y_file, z_file)
        → pour les fichiers à coordonnées réelles séparées :
          x_basse_St_Pierre.txt / y_basse_St_Pierre.txt / z_basse_St_Pierre.txt
          (et leurs versions _zoom)
          - X, Y sont de vraies coordonnées Lambert (métriques)
          - Z peut contenir des NaN (zones non mesurées)
"""

import numpy as np


def load_z(filepath, pas=1.0):
    """
    Charge un MNT depuis un fichier contenant uniquement les valeurs Z
    (grille 2D de flottants séparés par des espaces/tabs).

    Formats supportés : .txt, .z, .xyz (grille déguisée)

    Paramètres
    ----------
    filepath : str   chemin vers le fichier
    pas      : float résolution spatiale en mètres (défaut 1 m)

    Retourne
    --------
    X, Y, Z : np.ndarray 2D  (meshgrid-compatibles avec vis.afficher_2D)
    """
    Z = np.loadtxt(filepath)

    nrows, ncols = Z.shape
    x = np.arange(ncols) * pas
    y = np.arange(nrows) * pas
    X, Y = np.meshgrid(x, y)

    return X, Y, Z


def load_xyz_separes(x_file, y_file, z_file):
    """
    Charge un MNT depuis trois fichiers séparés x / y / z.

    Chaque fichier est une grille 2D de même dimension.
    Les coordonnées X, Y sont métriques (Lambert).
    Z peut contenir des NaN (zones non sondées).

    Paramètres
    ----------
    x_file, y_file, z_file : str  chemins vers les fichiers

    Retourne
    --------
    X, Y, Z : np.ndarray 2D
    """
    X = np.loadtxt(x_file)
    Y = np.loadtxt(y_file)
    Z = np.loadtxt(z_file)

    assert X.shape == Y.shape == Z.shape, (
        f"Dimensions incohérentes : X={X.shape}, Y={Y.shape}, Z={Z.shape}"
    )

    return X, Y, Z

