import cv2 as cv
import numpy as np

#Izracun Manhattan razdalje
def manhattan_razdalja(a, b):
    return np.sum(np.abs(a-b), axis=-1)
#Izracun Gaussovega jedra
def gaussovo_jedro(d, h):
    return np.exp(-d**2 / (2 * h**2))


def kmeans(slika, k=3, iteracije=10, izbira = "nakljucno", dimenzija_centra = 3, T=30):
    '''Izvede segmentacijo slike z uporabo metode k-means.1'''
    h, w, c = slika.shape  # Dobi višino, širino in kanale slike
    #ce uporabimo dimenzijo 5, dodamo tudi lokacijske koordinate
    if dimenzija_centra == 5:
        X, Y = np.meshgrid(np.arange(w), np.arange(h))  # Ustvari mrežo koordinat X in Y
        X = X / w #Normaliziramo x koordinate 
        Y = Y / h #normaliziramo y koordinate
        podatki = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)  # Združi barve + koordinate
    else:
        podatki = slika  # uporabimo samo barvne vrednosti

    podatki = podatki.reshape(-1, dimenzija_centra) #pretvorba v seznam pikslov
    centri = izracunaj_centre(slika, izbira, dimenzija_centra, T, k) #inicializacija centrov

    for _ in range(iteracije):
        dists = np.array([manhattan_razdalja(podatki, center) for center in centri])#izracun razdalje do centrov
        labels = np.argmin(dists, axis=0) #vsak piksel dobi oznako centra, ki je najbližji

        #Pridobimo vsak center
        for i in range(k):
            if np.any(labels == i): #Preverimo pripadajoče piksele
                centri[i] = np.mean(podatki[labels == i], axis=0) #Novi center je povprečje vseh pikslov, ki pripadajo centru

    nova_slika = np.zeros_like(podatki)  # Inicializira novo sliko (prazno)
    for i in range(k):
        nova_slika[labels == i] = centri[i] #Vsaka točka dobi barvo njenega centra
    if dimenzija_centra == 5:
        nova_slika = nova_slika[:, :3] # Če imamo dimenzijo 5, obdržimo samo barvne vrednosti

    return nova_slika.reshape(h, w, 3).astype(np.uint8) #Preoblikujemo v 3D sliko
    pass

#min_cd bomo uporabljali za združevanje centrov, ki so blizu drug drugemu, z max_iter pa limitiramo število iteracij, da se izognemo neskončnim zankam
def meanshift(slika, h, dimenzija=3, min_cd=20, max_iter=10):
    """Segmentacija slike z uporabo metode Mean-Shift."""
    slika = slika.astype(np.float32)
    h_img, w_img, _ = slika.shape

    if dimenzija == 5:
        X, Y = np.meshgrid(np.arange(w_img), np.arange(h_img))
        #normaliziramo, ker imajo barve(0-255) in koordinate okolja(0-102) drugačne vrednosti/razmerja
        X = X / w_img  # Normaliziramo na [0, 1]
        Y = Y / h_img  # Normaliziramo na [0, 1]
        #slika_norm = slika / 255.0  # Normalize colors
        points = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)
    else:
        points = slika

    points = points.reshape(-1, dimenzija)
    premiki = np.copy(points) # Ustvari kopijo točk za premike

    for i in range(len(points)):
        tocka = points[i] # Izberi začetno točko
        for _ in range(max_iter): #Iteriramo premik tocke
            razdalje = manhattan_razdalja(premiki, tocka)
            mask = razdalje < 3 * h #Upostevamo samo lokalno okolico
            if np.sum(mask) == 0:  # Izognemo se deljenju z 0, saj je to program pohitrilo za nekaj sekund
                break
            utezi = gaussovo_jedro(razdalje[mask], h) # Uporabimo Gaussovo jedro za uteži
            nova_tocka = np.sum(utezi[:, None] * premiki[mask], axis=0) / np.sum(utezi) #Nova pozicija tocke
            if manhattan_razdalja(nova_tocka, tocka) < 1: # Ce se tocka skoraj ne premika - konvergenca
                break
            tocka = nova_tocka
        premiki[i] = tocka #Shranimo konvergirano pozicijo točke
        if i % 3000 == 0:
            print(f"{i}/{len(points)} obdelanih...")#napredek obdelovanja

    # Združevanje končnih točk v centre (glede na min_cd)
    centri = []
    labels = np.zeros(len(points), dtype=np.int32)
    for i in range(len(points)):
        found = False
        for j, c in enumerate(centri):
            if manhattan_razdalja(premiki[i], c) < min_cd: #Če je blizu že obstoječemu centru jo pripnemo obstoječemu centru
                labels[i] = j
                found = True
                break
        if not found:
            centri.append(premiki[i]) #Dodamo nov center
            labels[i] = len(centri) - 1
    print(f'Najdenih {len(centri)} skupin točk.')

    nova_slika = np.array([centri[l] for l in labels]) #Barve po centrih
    if dimenzija == 5:
        nova_slika = nova_slika[:, :3]#odstranimo lokacije
    return nova_slika.reshape(h_img, w_img, 3).astype(np.uint8)
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    h, w, _ = slika.shape  # Dobi višino (h), širino (w) in globino (_) slike
    if dimenzija_centra == 5:
        X, Y = np.meshgrid(np.arange(w), np.arange(h))  # Ustvari mrežo koordinat X in Y
        X = X / w
        Y = Y / h
        slika_norm = slika / 255.0
        slika_features = np.concatenate((slika, X[..., np.newaxis], Y[..., np.newaxis]), axis=-1)  # Združi barve in koordinate
    else:
        slika_features = slika  # Če dimenzija=3, obdrži samo barve

    if izbira == "nakljucno":
        slika_features = slika_features.reshape(-1, dimenzija_centra)
        centri = []
        while len(centri) < k:
            idx = np.random.randint(0, len(slika_features)) # Naključno izberi indeks
            kandidat = slika_features[idx]
            if not centri or all(manhattan_razdalja(kandidat, c) > T for c in centri):
                centri.append(kandidat) # Dodaj kandidat, če je dovolj daleč od obstoječih centrov
        return np.array(centri)  # Vrni centre kot numpy array
    elif izbira == "rocno":
        raise NotImplementedError("Ročna izbira še ni implementirana!")  # Če bi hoteli ročno izbirati centre
    else:
        raise ValueError("Napaka: izbira mora biti 'nakljucno' ali 'rocno'.")  # Če izbira ni veljavna, sproži napako


if __name__ == "__main__":
    print("Začetek")
    slika = cv.imread(".utils\zelenjava.jpg")
    shape = slika.shape
    if slika is None:
        raise FileNotFoundError("Slika 'zelenjava.jpg' ni bila najdena.")
    slika = cv.cvtColor(slika, cv.COLOR_BGR2RGB)#da bo slika v izvornih barvah
    slika = cv.resize(slika, (90, 90))  # Pomanjšaj sliko

    #segment_kmeans = kmeans(slika, k=3, iteracije=5, dimenzija_centra=5, T=0.2)
    #cv.imwrite("rezultat_kmeans.png", cv.cvtColor(segment_kmeans, cv.COLOR_RGB2BGR))

    segment_meanshift = meanshift(slika, h=10, dimenzija=5, min_cd=5, max_iter=10)
    cv.imwrite("rezultat_meanshift.png", cv.cvtColor(segment_meanshift, cv.COLOR_RGB2BGR))

    print("Segmentacija končana.")