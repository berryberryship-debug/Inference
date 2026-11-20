cd ~
cp -p experience_orientation.py experience_orientation.py.avant_bande_commune.bak 2>/dev/null || true

cat > experience_orientation.py <<'PYTHON'
#!/usr/bin/env python3
"""Effet d'un couplage oriente par alignement dans le modele E-I."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, hilbert, periodogram, sosfiltfilt

import stabilite_neuronale as modele

W_EE = 8.0
DUREE = 240.0
DT = 0.05
AMPLITUDE = 1e-4
SEED_PERTURBATION = 123
SEED_DIRECTIONS = 7
KAPPAS = [0.0, 1.0, 2.0, 4.0]
FENETRE_ANALYSE = (160.0, 240.0)

DOSSIER = Path("resultats/experience_orientation")


def matrice_couplage_orientee(points, adjacency, directions, kappa):
    """A[i, j] = poids de la connexion de j vers i."""
    points = np.asarray(points, dtype=float)
    A0 = np.asarray(adjacency, dtype=float).copy()
    n = points.shape[0]

    if A0.shape != (n, n):
        raise ValueError("adjacency doit avoir la forme (N, N).")

    np.fill_diagonal(A0, 0.0)

    r = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(r, axis=2)
    direction_geometrique = np.zeros_like(r)

    masque = (distance > 0.0) & (A0 > 0.0)
    direction_geometrique[masque] = (
        r[masque] / distance[masque, None]
    )

    directions = np.asarray(directions, dtype=float)
    if directions.shape != (n, 3):
        raise ValueError("directions doit avoir la forme (N, 3).")

    normes = np.linalg.norm(directions, axis=1, keepdims=True)
    directions_unitaires = np.divide(
        directions,
        normes,
        out=np.zeros_like(directions),
        where=normes > 0.0,
    )

    alignement = np.einsum(
        "jd,ijd->ij",
        directions_unitaires,
        direction_geometrique,
    )
    alignement = np.clip(alignement, -1.0, 1.0)

    A = A0 * np.exp(kappa * alignement)
    A[A0 == 0.0] = 0.0
    np.fill_diagonal(A, 0.0)

    force_initiale = A0.sum(axis=1, keepdims=True)
    force_orientee = A.sum(axis=1, keepdims=True)

    A = np.divide(
        A * force_initiale,
        force_orientee,
        out=np.zeros_like(A),
        where=force_orientee > 0.0,
    )

    return A


def laplacien_entree(A):
    """L_in = diag(somme des poids entrants) - A ; L_in @ 1 = 0."""
    return np.diag(A.sum(axis=1)) - A


def construire_rhs(L, w_EE):
    P, Q = modele.entrees(w_EE)

    def rhs(y):
        E, I = y
        dE = (
            -E
            + modele.sigmoid(w_EE * E - modele.W_EI * I + P)
            - modele.D_E * (L @ E)
        ) / modele.TAU_E
        dI = (
            -I
            + modele.sigmoid(modele.W_IE * E - modele.W_II * I + Q)
            - modele.D_I * (L @ I)
        ) / modele.TAU_I
        return np.array([dE, dI])

    return rhs


def simuler(L, etat_initial):
    rhs = construire_rhs(L, W_EE)
    nombre_pas = int(round(DUREE / DT))
    y = etat_initial.copy()

    temps = [0.0]
    E = [y[0].copy()]
    I = [y[1].copy()]

    for numero_pas in range(nombre_pas):
        a = rhs(y)
        b = rhs(y + DT * a / 2.0)
        c = rhs(y + DT * b / 2.0)
        d = rhs(y + DT * c)
        y = y + DT * (a + 2.0 * b + 2.0 * c + d) / 6.0

        if not np.all(np.isfinite(y)):
            raise RuntimeError("Etat non fini.")

        if np.any(y < -1e-9) or np.any(y > 1.0 + 1e-9):
            raise RuntimeError("Etat hors de [0, 1].")

        if (numero_pas + 1) % 10 == 0:
            temps.append((numero_pas + 1) * DT)
            E.append(y[0].copy())
            I.append(y[1].copy())

    return np.asarray(temps), np.asarray(E), np.asarray(I)


def temps_pics_paraboliques(temps, signal):
    if signal.size < 3:
        return np.array([], dtype=float)

    indices = np.flatnonzero(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
    ) + 1

    if indices.size == 0:
        return np.array([], dtype=float)

    instants = []
    pas = float(temps[1] - temps[0])

    for indice in indices:
        if indice == 0 or indice == signal.size - 1:
            instants.append(float(temps[indice]))
            continue

        y_gauche = signal[indice - 1]
        y_centre = signal[indice]
        y_droite = signal[indice + 1]
        denominateur = y_gauche - 2.0 * y_centre + y_droite

        if abs(denominateur) < 1e-30:
            decalage = 0.0
        else:
            decalage = 0.5 * (y_gauche - y_droite) / denominateur
            decalage = float(np.clip(decalage, -0.5, 0.5))

        instants.append(float(temps[indice] + decalage * pas))

    return np.asarray(instants, dtype=float)


def parametres_bande_commune(signal_E, signal_I, fs):
    """Frequence dominante et bande communes a E et I."""
    x = detrend(signal_E, type="linear")
    y = detrend(signal_I, type="linear")

    frequences, puissance_E = periodogram(
        x,
        fs=fs,
        window="hann",
        detrend=False,
    )
    _, puissance_I = periodogram(
        y,
        fs=fs,
        window="hann",
        detrend=False,
    )

    puissance_commune = puissance_E + puissance_I
    valide = (frequences > 0.0) & (frequences < 0.9 * fs / 2.0)

    if not np.any(valide):
        raise ValueError("Aucune frequence dominante exploitable.")

    f0 = float(
        frequences[valide][
            np.argmax(puissance_commune[valide])
        ]
    )

    df = float(frequences[1] - frequences[0])
    largeur = max(2.0 * df, 0.25 * f0)
    fbas = max(df, f0 - largeur)
    fhaut = min(0.9 * fs / 2.0, f0 + largeur)

    if not (0.0 < fbas < fhaut < fs / 2.0):
        raise ValueError("Bande de filtrage incompatible.")

    return f0, fbas, fhaut


def filtrer_dans_bande(signal, fs, fbas, fhaut):
    """Filtrage passe-bande zero-phase."""
    x = detrend(signal, type="linear")

    sos = butter(
        4,
        [fbas, fhaut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, x)


def periode_par_phase_depliee(temps, signal_filtre):
    """Periode par regression lineaire de la phase de Hilbert depliee."""
    temps = np.asarray(temps, dtype=float)
    phase = np.unwrap(np.angle(hilbert(signal_filtre)))

    temps_relatif = temps - temps[0]
    pente, _ = np.polyfit(temps_relatif, phase, 1)
    frequence = float(pente / (2.0 * np.pi))

    if frequence <= 0.0:
        raise ValueError("Frequence estimee non positive.")

    return float(1.0 / frequence), frequence


def analyser_fenetre(temps, E, I, debut, fin):
    masque = (temps >= debut) & (temps <= fin)
    t = temps[masque]

    moyenne_E = E[masque].mean(axis=1)
    moyenne_I = I[masque].mean(axis=1)

    instants_E = temps_pics_paraboliques(t, moyenne_E)
    instants_I = temps_pics_paraboliques(t, moyenne_I)

    if instants_E.size < 3 or instants_I.size < 3:
        raise RuntimeError("Pas assez de pics pour estimer une periode.")

    intervalles_E = np.diff(instants_E)
    intervalles_I = np.diff(instants_I)

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    f0, fbas, fhaut = parametres_bande_commune(
        moyenne_E,
        moyenne_I,
        fs,
    )

    E_filtre = filtrer_dans_bande(moyenne_E, fs, fbas, fhaut)
    I_filtre = filtrer_dans_bande(moyenne_I, fs, fbas, fhaut)

    z_E_complet = hilbert(E_filtre)
    z_I_complet = hilbert(I_filtre)

    nombre_bord = int(np.ceil(2.0 * fs / f0))

    if len(t) <= 2 * nombre_bord + 10:
        raise RuntimeError("Fenetre trop courte apres exclusion des bords.")

    t_phase = t[nombre_bord:-nombre_bord]
    z_E = z_E_complet[nombre_bord:-nombre_bord]
    z_I = z_I_complet[nombre_bord:-nombre_bord]

    E_phase = np.real(z_E)
    I_phase = np.real(z_I)

    periode_E, frequence_E = periode_par_phase_depliee(
        t_phase,
        E_phase,
    )
    periode_I, frequence_I = periode_par_phase_depliee(
        t_phase,
        I_phase,
    )

    delta_phi = np.angle(z_E * np.conj(z_I))
    vecteur = np.mean(np.exp(1j * delta_phi))

    return {
        "amplitude_E": float(np.ptp(moyenne_E)),
        "amplitude_I": float(np.ptp(moyenne_I)),
        "periode_pics_E": float(intervalles_E.mean()),
        "periode_pics_I": float(intervalles_I.mean()),
        "periode_phase_E": float(periode_E),
        "periode_phase_I": float(periode_I),
        "frequence_phase_E": float(frequence_E),
        "frequence_phase_I": float(frequence_I),
        "frequence_dominante_commune": float(f0),
        "bande_phase_fbas": float(fbas),
        "bande_phase_fhaut": float(fhaut),
        "phase_relative_deg": float(np.degrees(np.angle(vecteur))),
        "concentration_R": float(abs(vecteur)),
        "variance_spatiale_E_moyenne": float(
            E[masque].var(axis=1).mean()
        ),
        "variance_spatiale_I_moyenne": float(
            I[masque].var(axis=1).mean()
        ),
        "nombre_bord_exclu": int(nombre_bord),
    }


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Effet d'un couplage oriente par alignement ===")
    print(f"w_EE = {W_EE} ; kappas = {KAPPAS}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)

    rng_directions = np.random.default_rng(SEED_DIRECTIONS)
    directions = rng_directions.normal(size=(modele.N, 3))

    rng_perturbation = np.random.default_rng(SEED_PERTURBATION)
    etat_initial = 0.5 + AMPLITUDE * rng_perturbation.normal(
        size=(2, modele.N)
    )

    resultats = []

    for kappa in KAPPAS:
        print(f"\nKappa = {kappa:g}")

        A_oriente = matrice_couplage_orientee(
            points,
            adjacency,
            directions,
            kappa,
        )

        if kappa == 0.0:
            difference = np.max(
                np.abs(
                    A_oriente
                    - np.asarray(adjacency, dtype=float)
                )
            )

            if difference > 1e-12:
                raise RuntimeError(
                    "Kappa = 0 ne reproduit pas l'adjacence initiale."
                )

        L_oriente = laplacien_entree(A_oriente)

        controle_conservation = float(
            np.max(np.abs(L_oriente @ np.ones(modele.N)))
        )

        if controle_conservation > 1e-10:
            raise RuntimeError(
                "Le laplacien n'annule pas le vecteur constant."
            )

        temps, E, I = simuler(L_oriente, etat_initial)

        suffixe = f"kappa_{kappa:.1f}"
        np.savez_compressed(
            DOSSIER / f"trajectoire_{suffixe}.npz",
            temps=temps,
            E=E,
            I=I,
        )

        mesures = analyser_fenetre(
            temps,
            E,
            I,
            FENETRE_ANALYSE[0],
            FENETRE_ANALYSE[1],
        )
        mesures["kappa"] = kappa
        mesures["controle_conservation"] = controle_conservation
        resultats.append(mesures)

        print(
            f"  Amplitude E / I : "
            f"{mesures['amplitude_E']:.6g} / "
            f"{mesures['amplitude_I']:.6g}"
        )
        print(
            f"  Periode pics E / I : "
            f"{mesures['periode_pics_E']:.6g} / "
            f"{mesures['periode_pics_I']:.6g}"
        )
        print(
            f"  Periode phase E / I : "
            f"{mesures['periode_phase_E']:.6g} / "
            f"{mesures['periode_phase_I']:.6g}"
        )
        print(
            f"  Phase E-I : {mesures['phase_relative_deg']:.2f} deg ; "
            f"R = {mesures['concentration_R']:.3f}"
        )
        print(
            f"  Variance spatiale E / I : "
            f"{mesures['variance_spatiale_E_moyenne']:.3e} / "
            f"{mesures['variance_spatiale_I_moyenne']:.3e}"
        )

    resume = {
        "statut": "Exploration d'un couplage oriente, non calibree",
        "w_EE": W_EE,
        "duree": DUREE,
        "dt": DT,
        "amplitude_perturbation": AMPLITUDE,
        "seed_graphe": modele.SEED,
        "seed_directions": SEED_DIRECTIONS,
        "seed_perturbation": SEED_PERTURBATION,
        "kappas": KAPPAS,
        "fenetre_analyse": list(FENETRE_ANALYSE),
        "convention": "A[i, j] = connexion de j vers i",
        "laplacien": "L_in = diag(somme des poids entrants) - A",
        "filtrage": "Bande commune a E et I ; bords exclus",
        "resultats": resultats,
    }

    with (DOSSIER / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, allow_nan=False)

    print("\n=== Comparaison ===")
    for mesures in resultats:
        print(
            f"kappa={mesures['kappa']:g} : "
            f"A_E={mesures['amplitude_E']:.5f}, "
            f"A_I={mesures['amplitude_I']:.5f}, "
            f"T_E={mesures['periode_phase_E']:.5f}, "
            f"T_I={mesures['periode_phase_I']:.5f}, "
            f"phi={mesures['phase_relative_deg']:.2f} deg"
        )

    print(f"\nResultats : {DOSSIER}")
    print(
        "Kappa = 0 doit reproduire le couplage symetrique initial."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
