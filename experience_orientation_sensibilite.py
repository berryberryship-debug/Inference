cd ~

cat > experience_orientation_sensibilite.py <<'PYTHON'
#!/usr/bin/env python3
"""Sensibilite de la propagation orientee a l'amplitude et au seuil."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

import stabilite_neuronale as modele
from experience_orientation import (
    laplacien_entree,
    matrice_couplage_orientee,
)
from experience_orientation_localisee import (
    analyser_direction,
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


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Sensibilite de la propagation orientee ===")
    print(f"Kappas : {KAPPAS}")
    print(f"Amplitudes : {AMPLITUDES_IMPULSION}")
    print(f"Seuils : {FRACTIONS_SEUIL}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)

    # Champ coherent selon l'axe x.
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

    with (DOSSIER / "resume.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "statut": "Sensibilite de la propagation orientee",
                "w_EE": W_EE,
                "dt": DT,
                "fenetre": [T_DEBUT_PERTURBATION, T_FIN],
                "noeud_source": NOEUD_SOURCE,
                "sigma_spatial": SIGMA_SPATIAL,
                "duree_impulsion": DUREE_IMPULSION,
                "duree_soutien": DUREE_SOUTIEN,
                "champ_directions": "coherent selon l'axe x",
                "resultats": resultats,
            },
            f,
            indent=2,
            allow_nan=False,
        )

    print(f"\nResultats : {DOSSIER}")
    print(
        "Une reponse globale a faible amplitude indiquerait une "
        "sensibilisation du regime, pas necessairement une propagation."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
