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

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    '''Izračuna centre za metodo kmeans.3'''
    h, w, _ = slika.shape
    if dimenzija_centra == 5:
        X, Y = np.meshgrid(np.arrange(w), np.arrange(h))
        slika_features = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)  # Združi barve in koordinate
    else:
        slika_features = slika

    if izbira == "nakljucno":
        centri = []
        while len(centri) < k:
            idx = np.random.randint(0, h)  # Naključno izberi indeks vrstice
            idy = np.random.randint(0, w)  # Naključno izberi indeks stolpca
            kandidat = slika_features[idx, idy]  # Kandidat za center
            if all(np.linalg.norm(kandidat - c) > T for c in centri):  # Preveri, da je kandidat dovolj oddaljen od vseh obstoječih centrov
                centri.append(kandidat)  # Če je pogoj izpolnjen, shrani kandidata kot center
        return np.array(centri)  # Vrni centre kot numpy array
    elif izbira == "rocno":
        raise NotImplementedError("Ročna izbira še ni implementirana!")
    else:
        raise ValueError("Napaka: izbira mora biti 'nakljucno' ali 'rocno'.")
    pass

if __name__ == "__main__":
    pass