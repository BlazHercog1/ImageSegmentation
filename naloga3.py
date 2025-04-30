import cv2 as cv
import numpy as np

def manhattan_razdalja(a, b):
    return np.sum(np.abs(a-b), axis=-1)

def gaussovo_jedro(d, h):
    return np.exp(-d**2 / (2 * h**2))


def kmeans(slika, k=3, iteracije=10, izbira = "nakljucno", dimenzija_centra = 3, T=30):
    '''Izvede segmentacijo slike z uporabo metode k-means.1'''
    h, w, c = slika.shape  # Dobi višino, širino in kanale slike
    if dimenzija_centra == 5:
        X, Y = np.meshgrid(np.arange(w), np.arange(h))  # Mreža koordinat
        podatki = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)  # Združi barve + koordinate
    else:
        podatki = slika  # Če samo barva, obdrži sliko

    podatki = podatki.reshape(-1, dimenzija_centra)
    centri = izracunaj_centre(slika, izbira, dimenzija_centra, T, k)

    for _ in range(iteracije):
        dists = np.array([manhattan_razdalja(podatki, center) for center in centri])
        labels = np.argmin(dists, axis=0)

        for i in range(k):
            if np.any(labels == i):
                centri[i] = np.mean(podatki[labels == i], axis=0)

    nova_slika = np.zeros_like(podatki)  # Inicializira novo sliko (prazno)
    for i in range(k):
        nova_slika[labels == i] = centri[i]
    if dimenzija_centra == 5:
        nova_slika = nova_slika[:, :3]

    return nova_slika.reshape(h, w, 3).astype(np.uint8)
    pass

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.2'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    h, w, _ = slika.shape  # Dobi višino (h), širino (w) in globino (_) slike
    if dimenzija_centra == 5:
        X, Y = np.meshgrid(np.arange(w), np.arange(h))  # Ustvari mrežo koordinat X in Y
        slika_features = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)  # Združi barve in koordinate
    else:
        slika_features = slika  # Če dimenzija=3, obdrži samo barve

    if izbira == "nakljucno":
        centri = []  # Inicializira seznam za centre
        while len(centri) < k:
            idx = np.random.randint(0, h)  # Naključno izberi indeks vrstice
            idy = np.random.randint(0, w)  # Naključno izberi indeks stolpca
            kandidat = slika_features[idx, idy]  # Kandidat za center
            if all(np.linalg.norm(kandidat - c) > T for c in centri):  # Preveri, da je kandidat dovolj oddaljen od vseh obstoječih centrov
                centri.append(kandidat)  # Če je pogoj izpolnjen, shrani kandidata kot center
        return np.array(centri)  # Vrni centre kot numpy array
    elif izbira == "rocno":
        raise NotImplementedError("Ročna izbira še ni implementirana!")  # Če bi hoteli ročno izbirati centre
    else:
        raise ValueError("Napaka: izbira mora biti 'nakljucno' ali 'rocno'.")  # Če izbira ni veljavna, sproži napako


if __name__ == "__main__":
     # -- Naloži sliko --
    slika = cv.imread('.utils\zelenjava.jpg')  # Preberi sliko iz datoteke
    if slika is None:
        raise FileNotFoundError("Slika 'zelenjava.jpg' ni bila najdena.")
    slika = cv.cvtColor(slika, cv.COLOR_BGR2RGB)

    # -- K-means segmentacija --
    k = 4  # Število centrov
    iteracije = 10
    dim = 5
    T = 20  # Toleranca med centri

    segmentirana_kmeans = kmeans(slika, k=k, iteracije=iteracije, izbira="nakljucno", dimenzija_centra=dim, T=T)
    cv.imshow("rezultat_kmeans.png", cv.cvtColor(segmentirana_kmeans, cv.COLOR_RGB2BGR))

    print("Obdelava končana. Slike shranjene.")
    pass