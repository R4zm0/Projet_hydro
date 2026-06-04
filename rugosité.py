import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generic_filter, gaussian_filter
import XY_txt_loader as loader
taille_fenetre = [3, 5, 7, 9, 11, 13, 15]


#X,Y,z = loader.load_z('txt/reels/bertheaume_z.txt')
z = np.loadtxt('txt/reels/bertheaume_z.txt')
#sigma² = E[z²] - E[z]²
print(z.shape)


def rugosite_std(z, size):
    ecarttype = generic_filter(z, np.nanstd, size=size) 
    
    #np.reshape(ecarttype, z.shape)
    return ecarttype
 
for size in taille_fenetre:
    rugosite = rugosite_std(z, size=size)
    plt.figure(size)
    plt.imshow(rugosite, cmap='viridis')
    plt.legend()
    plt.colorbar(label='Rugosité (écart-type local)')
    plt.title(f'Rugosité (écart-type local) avec une fenêtre de {size}x{size}')



#dispersion des vecteurs normaux 

#calculs des vecteurs normaux

def rugosite_normale(pente, exposition, size):
    theta = np.arctan(pente)
    phi = exposition
    #ramener ma matrice sous forme de vecteur flatten ou stack 
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sum_x = convolve(x, mode='constant', cval=0.0)
    sum_y = convolve(y, mode='constant', cval=0.0)
    sum_z = convolve(z, mode='constant', cval=0.0)
    r = np.sqrt(np.sum(sum_x)**2 + np.sum(sum_y)**2 + np.sum(sum_z)**2)
    return 1 - r / n

for size in taille_fenetre:
        rugosite = rugosite_normale(pente, exposition, size=size)
        plt.figure(size+10)
        plt.imshow(rugosite, cmap='viridis')
        plt.legend()
        plt.colorbar(label='Rugosité (dispersion des normales locales)')
        plt.title(f'Rugosité (dispersion des normales locales) avec une fenêtre de {size}x{size}')      