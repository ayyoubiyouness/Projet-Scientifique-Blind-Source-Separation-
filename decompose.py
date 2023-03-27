from statistics import mean
import numpy as np

import itertools



def decomposer(y):
    
    image_vecteur=list(itertools.chain.from_iterable(y))
    #taille de l'image
    nbligne=len(y)
    nbcolonne=len(y[0])
    L=[image_vecteur,nbligne,nbcolonne]
    return L

