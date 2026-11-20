cd ~

cat > experience_orientation_multisource.py <<'PYTHON'
#!/usr/bin/env python3
"""Moyenne de l'effet directionnel sur plusieurs noeuds sources."""

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
    construire_profil_local,
    construire_rhs_forcage,
    simuler_segment,
    temps_arrivee,
)
from experience_orientation_sensibilite import (
    analyser_direction,
    nettoyer_json,
)

W_EE = 8.0
DT = 0.05

T_DEBUT_PERTURBATION = 160.0
T_FIN = 240.0

AMPLITUDE_INITIALE = 1e-4
SEED_PERTURBATION = 123

KAPPAS = [0.0, 1.0, 2.0, 4.0]
AMPLITUDE_IMPULSION = 0.01
FRACTION_SEUIL = 0.10

SOURCES = [0, 16, 32, 48, 64, 80, 96, 112]
SIGMA_SPATIAL = 0.10
DUREE_IMPULSION = 1.0
DUREE_SOUTIEN = 0.50

INTERVALLE_ENREGISTREMENT = 10

DOSSIER = Path(
    "resultats/experience_orientation_multisource"
)


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Effet directionnel moyen sur plusieurs sources ===")
    print(f"Kappas : {KAPPAS}")
    print(f"Sources : {SOURCES}")
    print(f"Amplitude : {AMPLITUDE_IMPULSION}")
    print(f"Seuil : {FRACTION_SEUIL}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)

    axe = np.array([1.0, 0.0, 0.0])
    directions = np.repeat(
        axe[None, :],
        modele.N,
        axis=0,
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

        totaux = {
            "avant_E": 0,
            "arriere_E": 0,
            "avant_I": 0,
            "arriere_I": 0,
            "total_E": 0,
            "total_I": 0,
        }

        for source in SOURCES:
            profil_spatial = construire_profil_local(
                points,
                source,
                SIGMA_SPATIAL,
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

            rhs_impulsion = construire_rhs_forcage(
                L_oriente,
                W_EE,
                profil_spatial,
                temps_debut_impulsion=T_DEBUT_PERTURBATION,
                duree_impulsion=DUREE_IMPULSION,
                amplitude_impulsion=AMPLITUDE_IMPULSION,
            )

            _, E_impulsion, I_impulsion = simuler_segment(
                rhs_impulsion,
                T_DEBUT_PERTURBATION,
                T_FIN,
                etat_regime,
            )

            delta_E = E_impulsion - E_controle
            delta_I = I_impulsion - I_controle

            dt_enregistre = float(np.median(np.diff(temps)))
            largeur_lissage = max(
                1,
                int(round(0.50 / dt_enregistre)),
            )

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

            arrivees_E, _ = temps_arrivee(
                temps,
                enveloppe_E,
                source,
                FRACTION_SEUIL,
                DUREE_SOUTIEN,
                T_DEBUT_PERTURBATION,
            )

            arrivees_I, _ = temps_arrivee(
                temps,
                enveloppe_I,
                source,
                FRACTION_SEUIL,
                DUREE_SOUTIEN,
                T_DEBUT_PERTURBATION,
            )

            direction_E = analyser_direction(
                points,
                directions,
                source,
                arrivees_E,
                T_DEBUT_PERTURBATION,
            )

            direction_I = analyser_direction(
                points,
                directions,
                source,
                arrivees_I,
                T_DEBUT_PERTURBATION,
            )

            totaux["avant_E"] += direction_E["avant"]["nombre_noeuds"]
            totaux["arriere_E"] += direction_E["arriere"]["nombre_noeuds"]
            totaux["avant_I"] += direction_I["avant"]["nombre_noeuds"]
            totaux["arriere_I"] += direction_I["arriere"]["nombre_noeuds"]
            totaux["total_E"] += int(np.isfinite(arrivees_E).sum())
            totaux["total_I"] += int(np.isfinite(arrivees_I).sum())

        resultat = {
            "kappa": kappa,
            "nombre_sources": len(SOURCES),
            "amplitude_impulsion": AMPLITUDE_IMPULSION,
            "fraction_seuil": FRACTION_SEUIL,
            **totaux,
            "moyenne_avant_E": totaux["avant_E"] / len(SOURCES),
            "moyenne_arriere_E": totaux["arriere_E"] / len(SOURCES),
            "moyenne_avant_I": totaux["avant_I"] / len(SOURCES),
            "moyenne_arriere_I": totaux["arriere_I"] / len(SOURCES),
            "moyenne_total_E": totaux["total_E"] / len(SOURCES),
            "moyenne_total_I": totaux["total_I"] / len(SOURCES),
        }
        resultats.append(resultat)

        print(
            f"  E : avant={resultat['moyenne_avant_E']:.2f}, "
            f"arriere={resultat['moyenne_arriere_E']:.2f}, "
            f"total={resultat['moyenne_total_E']:.2f}"
        )
        print(
            f"  I : avant={resultat['moyenne_avant_I']:.2f}, "
            f"arriere={resultat['moyenne_arriere_I']:.2f}, "
            f"total={resultat['moyenne_total_I']:.2f}"
        )

    rapport = {
        "statut": "Effet directionnel moyen sur plusieurs sources",
        "w_EE": W_EE,
        "dt": DT,
        "fenetre": [T_DEBUT_PERTURBATION, T_FIN],
        "amplitude_impulsion": AMPLITUDE_IMPULSION,
        "fraction_seuil": FRACTION_SEUIL,
        "sources": SOURCES,
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
        "Une moyenne sur plusieurs sources distingue l'effet du champ "
        "de celui de la position du noeud source."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
