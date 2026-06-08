"""
courbures.py
============
Calcul des courbures verticale (kV), horizontale (kH)
et principales (kmin, kmax) a partir des coefficients d'Evans.

Fonctions exportables :
    calculer_courbures(Z, dx) -> kv, kh, kmin, kmax, slope, G
"""

import numpy as np
import pente


def calculer_courbures(Z, dx=1.0):
    """
    Calcule les 4 courbures d'Evans en chaque pixel.

    Parametres
    ----------
    Z  : np.ndarray (N, M)   MNT
    dx : float               pas spatial (m/pixel)

    Retourne
    --------
    kv    : courbure verticale   (sujet eq.4)
    kh    : courbure horizontale (sujet eq.5)
    kmin  : courbure principale minimale (sujet eq.6) — zones plates
    kmax  : courbure principale maximale (sujet eq.7) — zones plates
    slope : pente ||grad z||
    G     : gradient Evans (N, M, 2)
    """
    G, coeffs = pente.gradient_evans(Z, s=dx)

    A  = coeffs[..., 0]
    B  = coeffs[..., 1]
    C  = coeffs[..., 2]
    fx = coeffs[..., 3]
    fy = coeffs[..., 4]

    fxx = 2 * A       # d2z/dx2
    fyy = 2 * B       # d2z/dy2
    fxy = C           # d2z/dxdy

    p = np.maximum(fx**2 + fy**2, 1e-10)   # norme2 gradient (protege /0)
    q = p + 1

    # Courbure verticale — sujet eq.4
    kv = -(fxx*fx**2 + 2*fxy*fx*fy + fyy*fy**2) / (p * np.sqrt(q**3))

    # Courbure horizontale — sujet eq.5
    kh = -(fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) / (p * np.sqrt(q))

    # Courbures principales — sujet eq.6 et 7 (utilisees quand pente = 0)
    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    slope, _ = pente._aspect(fx, fy)

    return kv, kh, kmin, kmax, slope, G

def calculer_courbures_evans2(Z, n=1):
    """
    Calcule les 4 courbures d'Evans avec la méthode des moindres carrés en chaque pixel.

    Parametres
    ----------
    Z  : np.ndarray (N, M)   MNT
    dx : float               pas spatial (m/pixel)

    Retourne
    --------
    kv    : courbure verticale   (sujet eq.4)
    kh    : courbure horizontale (sujet eq.5)
    kmin  : courbure principale minimale (sujet eq.6) — zones plates
    kmax  : courbure principale maximale (sujet eq.7) — zones plates
    slope : pente ||grad z||
    G     : gradient Evans (N, M, 2)
    """
    G, coeffs = pente.gradient_evans_methode2(Z, n=n)

    A  = coeffs[..., 0]
    B  = coeffs[..., 1]
    C  = coeffs[..., 2]
    fx = coeffs[..., 3]
    fy = coeffs[..., 4]

    fxx = 2 * A       # d2z/dx2
    fyy = 2 * B       # d2z/dy2
    fxy = C           # d2z/dxdy

    p = np.maximum(fx**2 + fy**2, 1e-10)   # norme2 gradient (protege /0)
    q = p + 1

    # Courbure verticale — sujet eq.4
    kv = -(fxx*fx**2 + 2*fxy*fx*fy + fyy*fy**2) / (p * np.sqrt(q**3))

    # Courbure horizontale — sujet eq.5
    kh = -(fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) / (p * np.sqrt(q))

    # Courbures principales — sujet eq.6 et 7 (utilisees quand pente = 0)
    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    slope, _ = pente._aspect(fx, fy)

    return kv, kh, kmin, kmax, slope, G