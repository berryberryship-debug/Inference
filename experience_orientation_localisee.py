cd ~

cat > experience_orientation_localisee.py <<'PYTHON'
#!/usr/bin/env python3
"""Propagation d'une perturbation localisee selon l'orientation du couplage."""

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

W_EE = 8.0
DT = 0.05

T_DEBUT_PERTURBATION = 160.0
T_FIN = 240.0

AMPLITUDE_INITIALE = 1e-4
SEED_PERTURBATION = 123
SEED_DIRECTIONS = 7

KAPPAS = [0.0, 1.0, 2.0, 4.0]

NOEUD_SOURCE = 0
SIGMA_SPATIAL = 0.10
AMPLITUDE_IMPULSION = 1.0
DUREE_IMPULSION = 1.0

INTERVALLE_ENREGISTREMENT = 10

FRACTION_SEUIL = 0.20
DUREE_SOUTIEN = 0.50

DOSSIER = Path(
    "resultats/experience_orientation_localisee"
)


def construire_rhs_forcage(
    L,
    w_EE,
    profil_spatial,
    temps_debut_impulsion=None,
    duree_impulsion=0.0,
    amplitude_impulsion=0.0,
):
    P, Q = modele.entrees(w_EE)

    def porte_temporelle(t):
        if (
            temps_debut_impulsion is None
            or amplitude_impulsion == 0.0
            or t < temps_debut_impulsion
            or t > temps_debut_impulsion + duree_impulsion
        ):
            return 0.0

        phase = (
            t - temps_debut_impulsion
        ) / duree_impulsion

        return float(np.sin(np.pi * phase) ** 2)

    def rhs(t, y):
        E, I = y

        forcage = (
            amplitude_impulsion
            * porte_temporelle(t)
            * profil_spatial
        )

        entree_E = (
            w_EE * E
            - modele.W_EI * I
            + P
            + forcage
        )

        entree_I = (
            modele.W_IE * E
            - modele.W_II * I
            + Q
        )

        dE = (
            -E
            + modele.sigmoid(entree_E)
            - modele.D_E * (L @ E)
        ) / modele.TAU_E

        dI = (
            -I
            + modele.sigmoid(entree_I)
            - modele.D_I * (L @ I)
        ) / modele.TAU_I

        return np.array([dE, dI])

    return rhs


def simuler_segment(rhs, temps_debut, temps_fin, etat_initial):
    nombre_pas = int(
        round((temps_fin - temps_debut) / DT)
    )

    y = etat_initial.copy()

    temps = [temps_debut]
    E = [y[0].copy()]
    I = [y[1].copy()]

    for numero_pas in range(nombre_pas):
        t = temps_debut + numero_pas * DT

        a = rhs(t, y)
        b = rhs(t + DT / 2.0, y + DT * a / 2.0)
        c = rhs(t + DT / 2.0, y + DT * b / 2.0)
        d = rhs(t + DT, y + DT * c)

        y = y + DT * (
            a + 2.0 * b + 2.0 * c + d
        ) / 6.0

        if not np.all(np.isfinite(y)):
            raise RuntimeError("Etat non fini.")

        if np.any(y < -1e-9) or np.any(y > 1.0 + 1e-9):
            raise RuntimeError("Etat hors de [0, 1].")

        if (
            (numero_pas + 1)
            % INTERVALLE_ENREGISTREMENT
            == 0
            or numero_pas == nombre_pas - 1
        ):
            temps.append(t + DT)
            E.append(y[0].copy())
            I.append(y[1].copy())

    return (
        np.asarray(temps),
        np.asarray(E),
        np.asarray(I),
    )


def construire_profil_local(points, noeud_source, sigma):
    centre = points[noeud_source]
    distances2 = np.sum(
        (points - centre) ** 2,
        axis=1,
    )

    profil = np.exp(
        -distances2 / (2.0 * sigma ** 2)
    )

    maximum = profil.max()

    if maximum <= 0.0:
        raise ValueError("Profil spatial local nul.")

    return profil / maximum


def temps_arrivee(
    temps,
    enveloppe,
    noeud_source,
    fraction_seuil,
    duree_soutien,
    temps_min,
):
    """Premier franchissement soutenu d'un seuil relatif a la source."""
    dt_enregistre = float(
        np.median(np.diff(temps))
    )

    nombre_soutien = max(
        1,
        int(round(duree_soutien / dt_enregistre)),
    )

    maximum_source = np.max(
        enveloppe[:, noeud_source]
    )

    if maximum_source <= 0.0:
        raise ValueError(
            "Enveloppe nulle au noeud source."
        )

    seuil = fraction_seuil * maximum_source

    debut = int(
        np.searchsorted(temps, temps_min)
    )

    nombre_temps, nombre_noeuds = enveloppe.shape
    arrivees = np.full(
        nombre_noeuds,
        np.nan,
        dtype=float,
    )

    for noeud in range(nombre_noeuds):
        for index in range(
            debut,
            nombre_temps - nombre_soutien + 1,
        ):
            bloc = enveloppe[
                index:index + nombre_soutien,
                noeud,
            ]

            if np.all(bloc >= seuil):
                arrivees[noeud] = temps[index]
                break

    return arrivees, float(seuil)


def analyser_direction(
    points,
    directions,
    noeud_source,
    arrivees,
    temps_source,
):
    centre = points[noeud_source]

    direction_source = directions[noeud_source]
    norme = np.linalg.norm(direction_source)

    if norme <= 0.0:
        raise ValueError(
            "Direction nulle au noeud source."
        )

    direction_source = direction_source / norme

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
        )

        temps_relatifs = (
            arrivees[masque_valide]
            - temps_source
        )

        entree = {
            "nombre_noeuds": int(len(distances)),
            "temps_median": None,
            "pente_temps_distance": None,
            "vitesse_apparente": None,
            "R2": None,
        }

        if len(distances) >= 3:
            entree["temps_median"] = float(
                np.median(temps_relatifs)
            )

            ajustement = linregress(
                distances,
                temps_relatifs,
            )

            pente = float(ajustement.slope)

            entree["pente_temps_distance"] = pente
            entree["R2"] = float(
                ajustement.rvalue ** 2
            )

            if pente > 0.0:
                entree["vitesse_apparente"] = float(
                    1.0 / pente
                )

        resultat[nom] = entree

    return resultat


def main():
    DOSSIER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Perturbation localisee et propagation orientee ==="
    )
    print(f"Kappas : {KAPPAS}")
    print(
        f"Source : noeud {NOEUD_SOURCE} ; "
        f"impulsion : {DUREE_IMPULSION}"
    )

    rng = np.random.default_rng(
        modele.SEED
    )

    points = rng.uniform(
        0.0,
        1.0,
        size=(modele.N, 3),
    )

    adjacency = modele.construire_graphe(points)

    rng_directions = np.random.default_rng(
        SEED_DIRECTIONS
    )

    directions = rng_directions.normal(
        size=(modele.N, 3)
    )

    profil_spatial = construire_profil_local(
        points,
        NOEUD_SOURCE,
        SIGMA_SPATIAL,
    )

    rng_initial = np.random.default_rng(
        SEED_PERTURBATION
    )

    etat_initial = 0.5 + AMPLITUDE_INITIALE * (
        rng_initial.normal(
            size=(2, modele.N)
        )
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

        L_oriente = laplacien_entree(
            A_oriente
        )

        conservation = np.max(
            np.abs(
                L_oriente @ np.ones(modele.N)
            )
        )

        if conservation > 1e-10:
            raise RuntimeError(
                "Le laplacien ne conserve pas "
                "le mode uniforme."
            )

        # 1. Etablissement du regime avant perturbation
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

        # 2. Controle sans impulsion
        rhs_controle = construire_rhs_forcage(
            L_oriente,
            W_EE,
            profil_spatial,
            amplitude_impulsion=0.0,
        )

        temps, E_controle, I_controle = (
            simuler_segment(
                rhs_controle,
                T_DEBUT_PERTURBATION,
                T_FIN,
                etat_regime,
            )
        )

        # 3. Simulation avec impulsion localisee
        rhs_impulsion = construire_rhs_forcage(
            L_oriente,
            W_EE,
            profil_spatial,
            temps_debut_impulsion=(
                T_DEBUT_PERTURBATION
            ),
            duree_impulsion=DUREE_IMPULSION,
            amplitude_impulsion=(
                AMPLITUDE_IMPULSION
            ),
        )

        temps_impulsion, E_impulsion, I_impulsion = (
            simuler_segment(
                rhs_impulsion,
                T_DEBUT_PERTURBATION,
                T_FIN,
                etat_regime,
            )
        )

        if not np.allclose(temps, temps_impulsion):
            raise RuntimeError(
                "Les grilles temporelles different."
            )

        # Reponse differentielle a l'impulsion
        delta_E = E_impulsion - E_controle
        delta_I = I_impulsion - I_controle

        dt_enregistre = float(
            np.median(np.diff(temps))
        )

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

        arrivees_E, seuil_E = temps_arrivee(
            temps,
            enveloppe_E,
            NOEUD_SOURCE,
            FRACTION_SEUIL,
            DUREE_SOUTIEN,
            T_DEBUT_PERTURBATION,
        )

        arrivees_I, seuil_I = temps_arrivee(
            temps,
            enveloppe_I,
            NOEUD_SOURCE,
            FRACTION_SEUIL,
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

        suffixe = f"kappa_{kappa:.1f}"

        np.savez_compressed(
            DOSSIER / f"propagation_{suffixe}.npz",
            temps=temps,
            E_controle=E_controle,
            I_controle=I_controle,
            E_impulsion=E_impulsion,
            I_impulsion=I_impulsion,
            delta_E=delta_E,
            delta_I=delta_I,
            enveloppe_E=enveloppe_E,
            enveloppe_I=enveloppe_I,
            arrivees_E=arrivees_E,
            arrivees_I=arrivees_I,
            points=points,
            directions=directions,
        )

        resultat = {
            "kappa": kappa,
            "seuil_E": seuil_E,
            "seuil_I": seuil_I,
            "nombre_arrivees_E": int(
                np.isfinite(arrivees_E).sum()
            ),
            "nombre_arrivees_I": int(
                np.isfinite(arrivees_I).sum()
            ),
            "direction_E": direction_E,
            "direction_I": direction_I,
        }

        resultats.append(resultat)

        print(
            f"  Arrivees E : "
            f"{resultat['nombre_arrivees_E']}"
            f" / {modele.N}"
        )
        print(
            f"  Arrivees I : "
            f"{resultat['nombre_arrivees_I']}"
            f" / {modele.N}"
        )

        for nom, direction in (
            ("E", direction_E),
            ("I", direction_I),
        ):
            for sens in ("avant", "arriere"):
                mesures = direction[sens]
                print(
                    f"  {nom}, {sens} : "
                    f"n={mesures['nombre_noeuds']}, "
                    f"temps_median="
                    f"{mesures['temps_median']}, "
                    f"vitesse="
                    f"{mesures['vitesse_apparente']}"
                )

    with (
        DOSSIER / "resume.json"
    ).open("w", encoding="utf-8") as fichier:
        json.dump(
            {
                "w_EE": W_EE,
                "dt": DT,
                "fenetre": [
                    T_DEBUT_PERTURBATION,
                    T_FIN,
                ],
                "noeud_source": NOEUD_SOURCE,
                "sigma_spatial": SIGMA_SPATIAL,
                "amplitude_impulsion": (
                    AMPLITUDE_IMPULSION
                ),
                "duree_impulsion": (
                    DUREE_IMPULSION
                ),
                "fraction_seuil": FRACTION_SEUIL,
                "duree_soutien": DUREE_SOUTIEN,
                "resultats": resultats,
            },
            fichier,
            indent=2,
            allow_nan=False,
        )

    print(f"\nResultats : {DOSSIER}")
    print(
        "Les directions aleatoires definissent avant/arriere "
        "uniquement par rapport a la direction du noeud source."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
