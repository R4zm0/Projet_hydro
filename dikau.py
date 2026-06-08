"""
dikau.py
========
Classification morphologique de Dikau pour un pixel.

La fonction classer_pixel() prend les courbures et la pente d'un pixel
et retourne la classe Dikau correspondante.

La fonction classer_mnt() applique classer_pixel() sur tout le MNT.
"""

import numpy as np
from enum import IntEnum, auto


class Dikau(IntEnum):
    NOSE             = auto()   #  1 - eperon convexe
    SHOULDER_SLOPE   = auto()   #  2 - replat convexe
    HOLLOW_SHOULDER  = auto()   #  3 - epaulement concave
    SPUR             = auto()   #  4 - crete divergente
    PLANAR_SLOPE     = auto()   #  5 - pente plane
    HOLLOW           = auto()   #  6 - creux convergent
    SPUR_FOOT        = auto()   #  7 - pied d eperon
    FOOT_SLOPE       = auto()   #  8 - pied de pente
    HOLLOW_FOOT      = auto()   #  9 - bas de pente concave
    PEAK             = auto()   # 10 - sommet
    RIDGE            = auto()   # 11 - crete plate
    PLAIN            = auto()   # 12 - plaine
    SADDLE           = auto()   # 13 - col
    CHANNEL          = auto()   # 14 - chenal
    PIT              = auto()   # 15 - fosse


def _signe(valeur, epsilon):
    """
    Retourne le signe discretise d une courbure :
      +1  si valeur >  epsilon  (convexe ou convergent)
       0  si |valeur| <= epsilon (neutre)
      -1  si valeur < -epsilon  (concave ou divergent)
    """
    if valeur >  epsilon:
        return  1
    if valeur < -epsilon:
        return -1
    return 0


def classer_pixel(kv, kh, kmin, kmax, slope,
                  epsilon_p=2e-3, epsilon_c=2e-2):
    """
    Classifie UN pixel selon la methode de Dikau.

    Parametres
    ----------
    kv, kh         : courbures verticale et horizontale
                     (utilisees si pente >= epsilon_p)
    kmin, kmax     : courbures principales
                     (utilisees si pente <  epsilon_p)
    slope          : norme du gradient (pente scalaire)
    epsilon_p      : seuil pente — en dessous = terrain plat
    epsilon_c      : seuil courbure — en dessous = courbure nulle

    Retourne
    --------
    Dikau (IntEnum) : classe morphologique du pixel

    Principe
    --------
    Zones pentues (slope >= epsilon_p) :
        On croise le signe de kV et le signe de kH -> 9 classes

              kH  convexe(+1)  nul(0)         concave(-1)
        kV
        convexe(-1)  NOSE      SHOULDER_SLOPE  HOLLOW_SHOULDER
        nul(0)       SPUR      PLANAR_SLOPE    HOLLOW
        concave(+1)  SPUR_FOOT FOOT_SLOPE      HOLLOW_FOOT

    Zones plates (slope < epsilon_p) :
        On croise le signe de kmax et le signe de kmin -> 6 classes

              kmin convexe(+1)  nul(0)   concave(-1)
        kmax
        convexe(+1)  PEAK       RIDGE    SADDLE
        nul(0)       ---        PLAIN    CHANNEL
        concave(-1)  ---        ---      PIT
    """

    if slope >= epsilon_p:
        # ── Zone pentue ───────────────────────────────────────────────────
        sv = _signe(kv, epsilon_c)
        sh = _signe(kh, epsilon_c)

        table = {
            (-1, -1): Dikau.NOSE,
            (-1,  0): Dikau.SHOULDER_SLOPE,
            (-1, +1): Dikau.HOLLOW_SHOULDER,
            ( 0, -1): Dikau.SPUR,
            ( 0,  0): Dikau.PLANAR_SLOPE,
            ( 0, +1): Dikau.HOLLOW,
            (+1, -1): Dikau.SPUR_FOOT,
            (+1,  0): Dikau.FOOT_SLOPE,
            (+1, +1): Dikau.HOLLOW_FOOT,
        }
        return table[(sv, sh)]

    else:
        # ── Zone plate ────────────────────────────────────────────────────
        s_max = _signe(kmax, epsilon_c)
        s_min = _signe(kmin, epsilon_c)

        table = {
            (+1, +1): Dikau.PEAK,
            (+1,  0): Dikau.RIDGE,
            ( 0,  0): Dikau.PLAIN,
            (+1, -1): Dikau.SADDLE,
            ( 0, -1): Dikau.CHANNEL,
            (-1, -1): Dikau.PIT,
        }
        # Combinaison non prevue -> PLAIN par defaut
        return table.get((s_max, s_min), Dikau.PLAIN)


def classer_mnt(kv, kh, kmin, kmax, slope,
                epsilon_p=2e-3, epsilon_c=2e-2):
    """
    Applique classer_pixel() sur tout le MNT.

    Parametres
    ----------
    kv, kh, kmin, kmax, slope : np.ndarray (N, M)
        Sorties de courbures.calculer_courbures()

    Retourne
    --------
    mat_dikau : np.ndarray (N, M) de type float
        Valeur entiere Dikau en chaque pixel (nan si non classe)
    """
    mat_dikau = np.full(slope.shape, np.nan)

    flat  = slope < epsilon_p
    steep = ~flat

    sv   = np.zeros_like(kv,   dtype=int)
    sh   = np.zeros_like(kh,   dtype=int)
    smax = np.zeros_like(kmax, dtype=int)
    smin = np.zeros_like(kmin, dtype=int)

    sv  [ kv   >  epsilon_c] =  1;  sv  [ kv   < -epsilon_c] = -1
    sh  [ kh   >  epsilon_c] =  1;  sh  [ kh   < -epsilon_c] = -1
    smax[ kmax >  epsilon_c] =  1;  smax[ kmax < -epsilon_c] = -1
    smin[ kmin >  epsilon_c] =  1;  smin[ kmin < -epsilon_c] = -1

    # Zones pentues
    mat_dikau[steep & (sv==-1) & (sh==-1)] = Dikau.NOSE.value
    mat_dikau[steep & (sv==-1) & (sh== 0)] = Dikau.SHOULDER_SLOPE.value
    mat_dikau[steep & (sv==-1) & (sh==+1)] = Dikau.HOLLOW_SHOULDER.value
    mat_dikau[steep & (sv== 0) & (sh==-1)] = Dikau.SPUR.value
    mat_dikau[steep & (sv== 0) & (sh== 0)] = Dikau.PLANAR_SLOPE.value
    mat_dikau[steep & (sv== 0) & (sh==+1)] = Dikau.HOLLOW.value
    mat_dikau[steep & (sv==+1) & (sh==-1)] = Dikau.SPUR_FOOT.value
    mat_dikau[steep & (sv==+1) & (sh== 0)] = Dikau.FOOT_SLOPE.value
    mat_dikau[steep & (sv==+1) & (sh==+1)] = Dikau.HOLLOW_FOOT.value

    # Zones plates
    mat_dikau[flat & (smax==+1) & (smin==+1)] = Dikau.PEAK.value
    mat_dikau[flat & (smax==+1) & (smin== 0)] = Dikau.RIDGE.value
    mat_dikau[flat & (smax== 0) & (smin== 0)] = Dikau.PLAIN.value
    mat_dikau[flat & (smax==+1) & (smin==-1)] = Dikau.SADDLE.value
    mat_dikau[flat & (smax== 0) & (smin==-1)] = Dikau.CHANNEL.value
    mat_dikau[flat & (smax==-1) & (smin==-1)] = Dikau.PIT.value

    return mat_dikau
