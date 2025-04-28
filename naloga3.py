import cv2 as cv
import numpy as np

def manhattan_razdalja(a, b):
    return np.sum(np.abs(a-b), axis=-1)

def gaussovo_jedro(d, h):
    return np.exp(-d**2 / (2 * h**2))


def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.1'''
    pass

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.2'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.3'''
    pass

if __name__ == "__main__":
    pass