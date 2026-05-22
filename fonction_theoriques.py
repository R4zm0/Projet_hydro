import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ici seront stocker l'ensemble des fonctiones théoriques pour les terrains artificiels
# et leurs gradients respectifs

def plan(x, y):
    return 0.07*(x - 50) + 0.1*(y - 50)

def plateau(x, y):
    return 5 * np.tanh((x - 40) / 5)

def sinc_card(x, y):
    d = np.sqrt((x - 40)**2 + (y - 50)**2)
    if d == 0:
        return 10.0
    return 10 * np.sin(0.1*d) / (0.1*d)

def double_sin(x, y):
    return 5*np.sin(x/10 + 3*np.sin(y/20)) + 2*np.sin(y/5)

def grad_plan(x, y):
    return np.array([0.07, 0.1])

def grad_plateau(x, y):
    u = (x - 40) / 5
    return np.array([1 - np.tanh(u)**2, 0.0])

def grad_sinc(x, y):
    d = np.sqrt((x - 40)**2 + (y - 50)**2)
    if d == 0:
        return np.array([0.0, 0.0])
    coeff = 10 / d * (np.cos(0.1*d) / d - np.sin(0.1*d) / (0.1 * d**2))
    return coeff * np.array([x - 40, y - 50])

def grad_double_sin(x, y):
    u = x/10 + 3*np.sin(y/20)
    dx = 0.5 * np.cos(u)
    dy = 0.75 * np.cos(u) * np.cos(y/20) + 0.4 * np.cos(y/5)
    return np.array([dx, dy])

