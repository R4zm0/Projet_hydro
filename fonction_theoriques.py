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
    return np.where(d == 0, 10.0, 10 * np.sin(0.1*d) / (0.1*d))  

def double_sin(x, y):
    return 5*np.sin(x/10 + 3*np.sin(y/20)) + 2*np.sin(y/5)

#gradiants vectorisés/numpy compatibles, renvoi G = [dZ/dx, dZ/dy] de même shape que X et Y, le gradient vectoriel en chaque point du maillage quoi

def grad_plan(X, Y):
    return np.stack([np.full_like(X, 0.07), np.full_like(Y, 0.1)], axis=-1)

def grad_plateau(X, Y):
    u = (X - 40) / 5
    return np.stack([1 - np.tanh(u)**2, np.zeros_like(Y)], axis=-1)

def grad_sinc(X, Y):
    d = np.sqrt((X - 40)**2 + (Y - 50)**2)
    coeff = np.where(d == 0, 0.0, 10/d * (np.cos(0.1*d)/d - np.sin(0.1*d)/(0.1*d**2)))
    return np.stack([coeff * (X - 40), coeff * (Y - 50)], axis=-1)

def grad_double_sin(X, Y):
    u = X/10 + 3*np.sin(Y/20)
    dx = 0.5 * np.cos(u)
    dy = 0.75 * np.cos(u) * np.cos(Y/20) + 0.4 * np.cos(Y/5)
    return np.stack([dx, dy], axis=-1)
