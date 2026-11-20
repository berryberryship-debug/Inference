cd ~
cp -p experience_orientation_sensibilite.py experience_orientation_sensibilite.py.avant_correction_json.bak

cat > experience_orientation_sensibilite.py <<'PYTHON'
#!/usr/bin/env python3
"""Sensibilite de la propagation orientee a l'amplitude et au seuil."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

import stabilite_neuronale as modele
from experience_orientation import (
    laplacien_entree,
    matrice_couplage_orientee,
)
from experience_orientation_localisee import (
    construire_profil_local,
    construire_rhs_forcage,
    simuler_segment,
    temps_arrivee,
)

W_EE = 8.0
DT = 0.05

T_DEBUT_PERTURBATION = 160.0
T_FIN = 240.0

AMPLITUDE_INITIALE = 1e-4
SEED_PERTURBATION = 123

KAPPAS = [0.0, 1.0, 2.0, 4.0]
AMPLITUDES_IMPULSION = [0.01, 0.1, 1.0]
FRACTIONS_SEUIL = [0.10, 0.20, 0.30]

NOEUD_SOURCE = 0
SIGMA_SPATIAL = 0.10
DUREE_IMPULSION = 1.0
DUREE_SOUTIEN = 0.50

INTERVALLE_ENREGISTREMENT = 10

DOSSIER = Path(
    "resultats/experience_orientation_sensibilite"
)


def float_json(valeur):
    """Convertit en float JSON ; les valeurs non finies deviennent None."""
    if valeur is None:
        return None

    valeur = float(valeur)

    if not np.isfinite(valeur):
        return None

    return valeur


def nettoyer_json(objet):
    """Remplace recursivement NaN et inf par None."""
    if isinstance(objet, dict):
        return {
            cle: nettoyer_json(valeur)
            for cle, valeur in objet.items()
        }

    if isinstance(objet, (list, tuple)):
        return [nettoyer_json(valeur) for valeur in objet]

    if isinstance(objet, np.integer):
        return int(objet)

    if isinstance(objet, (np.floating, float)):
        return float_json(objet)

    if isinstance(objet, np.ndarray):
        return nettoyer_json(objet.tolist())

    return objet


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


def filtrer_autour_dominante(signal, fs):
    from scipy.signal import (
        butter,
        detrend,
        periodogram,
        sosfiltfilt,
    )

    x = detrend(signal, type="linear")
    frequences, puissance = periodogram(x, fs=fs, window="hann")

    valide = (frequences > 0.0) & (frequences < 0.9 * fs / 2.0)
    f0 = float(frequences[valide][np.argmax(puissance[valide])])

    df = float(frequences[1] - frequences[0])
    largeur = max(2.0 * df, 0.25 * f0)
    fbas = max(df, f0 - largeur)
    fhaut = min(0.9 * fs / 2.0, f0 + largeur)

    if not (0.0 < fbas < fhaut < fs / 2.0):
        raise ValueError("Bande de filtrage incompatible.")

    sos = butter(
        4,
        [fbas, fhaut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, x), f0


def periode_par_phase_depliee(temps, signal_filtre):
    from scipy.signal import hilbert

    phase = np.unwrap(np.angle(hilbert(signal_filtre)))
    temps_relatif = temps - temps[0]
    pente, _ = np.polyfit(temps_relatif, phase, 1)
    frequence = float(pente / (2.0 * np.pi))

    if frequence <= 0.0:
        raise ValueError("Frequence estimee non positive.")

    return 1.0 / frequence, frequence


def analyser_fenetre(temps, E, I, debut, fin):
    from scipy.signal import hilbert

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

    E_filtre, f0 = filtrer_autour_dominante(moyenne_E, fs)
    I_filtre, _ = filtrer_autour_dominante(moyenne_I, fs)

    periode_E, _ = periode_par_phase_depliee(t, E_filtre)
    periode_I, _ = periode_par_phase_depliee(t, I_filtre)

    z_E = hilbert(E_filtre)
    z_I = hilbert(I_filtre)

    delta_phi = np.angle(z_E * np.conj(z_I))
    vecteur = np.mean(np.exp(1j * delta_phi))

    return {
        "amplitude_E": float(np.ptp(moyenne_E)),
        "amplitude_I": float(np.ptp(moyenne_I)),
        "periode_pics_E": float(intervalles_E.mean()),
        "periode_pics_I": float(intervalles_I.mean()),
        "periode_phase_E": float(periode_E),
        "periode_phase_I": float(periode_I),
        "frequence_dominante": f0,
        "phase_relative_deg": float(
            np.degrees(np.angle(vecteur))
        ),
        "concentration_R": float(abs(vecteur)),
    }


def analyser_direction(
    points,
    directions,
    noeud_source,
    arrivees,
    temps_source,
):
    centre = points[noeud_source]

    direction_source = np.asarray(
        directions[noeud_source],
        dtype=float,
    )

    norme = np.linalg.norm(direction_source)

    if norme <= 0.0:
        raise ValueError("Direction nulle au noeud source.")

    direction_source /= norme

    deplacements = points - centre
    distance_signee = deplacements @ direction_source

    resultat = {}

    for nom, masque_direction in (
        ("avant", distance_signee > 0.0),
        ("arriere", distance_signee < 0.0),
    ):
        masque_valide = (
            masque_direction
            & np.isfinite(arrivees)
        )

        distances = np.abs(
            distance_signee[masque_valide]
        ).astype(float)

        temps_relatifs = (
            np.asarray(arrivees[masque_valide])
            - float(temps_source)
        ).astype(float)

        entree = {
            "nombre_noeuds": int(len(distances)),
            "temps_median": None,
            "pente_temps_distance": None,
            "vitesse_apparente": None,
            "R2": None,
        }

        if len(distances) > 0:
            entree["temps_median"] = float_json(
                np.median(temps_relatifs)
            )

        distances_uniques = np.unique(
            np.round(distances, decimals=12)
        )

        if (
            len(distances) >= 3
            and len(distances_uniques) >= 2
            and np.ptp(distances) > 1e-12
        ):
            ajustement = linregress(
                distances,
                temps_relatifs,
            )

            pente = float_json(ajustement.slope)
            rvalue = float_json(ajustement.rvalue)

            entree["pente_temps_distance"] = pente

            if rvalue is not None:
                entree["R2"] = float_json(rvalue ** 2)

            if pente is not None and pente > 0.0:
                entree["vitesse_apparente"] = float_json(
                    1.0 / pente
                )

        resultat[nom] = entree

    return resultat


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Sensibilite de la propagation orientee ===")
    print(f"Kappas : {KAPPAS}")
    print(f"Amplitudes : {AMPLITUDES_IMPULSION}")
    print(f"Seuils : {FRACTIONS_SEUIL}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)

    axe = np.array([1.0, 0.0, 0.0])
    directions = np.repeat(
        axe[None, :],
        modele.N,
        axis=0,
    )

    profil_spatial = construire_profil_local(
        points,
        NOEUD_SOURCE,
        SIGMA_SPATIAL,
    )

    rng_initial = np.random.default_rng(SEED_PERTURBATION)
    etat_initial = 0.5 + AMPLITUDE_INITIALE * rng_initial.normal(
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
        L_oriente = laplacien_entree(A_oriente)

        conservation = np.max(
            np.abs(L_oriente @ np.ones(modele.N))
        )

        if conservation > 1e-10:
            raise RuntimeError(
                "Le laplacien ne conserve pas le mode uniforme."
            )

        rhs_repos = construire_rhs_forcage(
            L_oriente,
            W_EE,
            profil_spatial,
        )

        _, E_repos, I_repos = simuler_segment(
            rhs_repos,
            0.0,
            T_DEBUT_PERTURBATION,
            etat_initial,
        )

        etat_regime = np.array([
            E_repos[-1].copy(),
            I_repos[-1].copy(),
        ])

        rhs_controle = construire_rhs_forcage(
            L_oriente,
            W_EE,
            profil_spatial,
            amplitude_impulsion=0.0,
        )

        temps, E_controle, I_controle = simuler_segment(
            rhs_controle,
            T_DEBUT_PERTURBATION,
            T_FIN,
            etat_regime,
        )

        dt_enregistre = float(np.median(np.diff(temps)))
        largeur_lissage = max(
            1,
            int(round(0.50 / dt_enregistre)),
        )

        for amplitude in AMPLITUDES_IMPULSION:
            rhs_impulsion = construire_rhs_forcage(
                L_oriente,
                W_EE,
                profil_spatial,
                temps_debut_impulsion=T_DEBUT_PERTURBATION,
                duree_impulsion=DUREE_IMPULSION,
                amplitude_impulsion=amplitude,
            )

            _, E_impulsion, I_impulsion = simuler_segment(
                rhs_impulsion,
                T_DEBUT_PERTURBATION,
                T_FIN,
                etat_regime,
            )

            delta_E = E_impulsion - E_controle
            delta_I = I_impulsion - I_controle

            enveloppe_E = uniform_filter1d(
                np.abs(delta_E),
                size=largeur_lissage,
                axis=0,
                mode="nearest",
            )
            enveloppe_I = uniform_filter1d(
                np.abs(delta_I),
                size=largeur_lissage,
                axis=0,
                mode="nearest",
            )

            for fraction in FRACTIONS_SEUIL:
                arrivees_E, seuil_E = temps_arrivee(
                    temps,
                    enveloppe_E,
                    NOEUD_SOURCE,
                    fraction,
                    DUREE_SOUTIEN,
                    T_DEBUT_PERTURBATION,
                )

                arrivees_I, seuil_I = temps_arrivee(
                    temps,
                    enveloppe_I,
                    NOEUD_SOURCE,
                    fraction,
                    DUREE_SOUTIEN,
                    T_DEBUT_PERTURBATION,
                )

                direction_E = analyser_direction(
                    points,
                    directions,
                    NOEUD_SOURCE,
                    arrivees_E,
                    T_DEBUT_PERTURBATION,
                )

                direction_I = analyser_direction(
                    points,
                    directions,
                    NOEUD_SOURCE,
                    arrivees_I,
                    T_DEBUT_PERTURBATION,
                )

                resultat = {
                    "kappa": kappa,
                    "amplitude_impulsion": amplitude,
                    "fraction_seuil": fraction,
                    "seuil_E": seuil_E,
                    "seuil_I": seuil_I,
                    "nombre_arrivees_E": int(
                        np.isfinite(arrivees_E).sum()
                    ),
                    "nombre_arrivees_I": int(
                        np.isfinite(arrivees_I).sum()
                    ),
                    "amplitude_max_delta_E": float(
                        np.max(np.abs(delta_E))
                    ),
                    "amplitude_max_delta_I": float(
                        np.max(np.abs(delta_I))
                    ),
                    "direction_E": direction_E,
                    "direction_I": direction_I,
                }
                resultats.append(resultat)

                print(
                    f"  A={amplitude:g}, seuil={fraction:g} : "
                    f"E={resultat['nombre_arrivees_E']}/128, "
                    f"I={resultat['nombre_arrivees_I']}/128, "
                    f"max_dE={resultat['amplitude_max_delta_E']:.3g}, "
                    f"max_dI={resultat['amplitude_max_delta_I']:.3g}"
                )

    rapport = {
        "statut": "Sensibilite de la propagation orientee",
        "w_EE": W_EE,
        "dt": DT,
        "fenetre": [
            T_DEBUT_PERTURBATION,
            T_FIN,
        ],
        "noeud_source": NOEUD_SOURCE,
        "sigma_spatial": SIGMA_SPATIAL,
        "duree_impulsion": DUREE_IMPULSION,
        "duree_soutien": DUREE_SOUTIEN,
        "champ_directions": "coherent selon l'axe x",
        "resultats": resultats,
    }

    rapport = nettoyer_json(rapport)

    with (DOSSIER / "resume.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"\nResultats : {DOSSIER}")
    print(
        "Une reponse globale a faible amplitude indiquerait une "
        "sensibilisation du regime, pas necessairement une propagation."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
