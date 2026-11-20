#!/usr/bin/env python3
"""
Prototype hippocampo-cingulaire du "futur passe".

Module attendu par joindre_futur_passe_metric.py.

Modele conceptuel exploratoire, non valide biologiquement ou cliniquement.
Aucune demonstration de conscience ou de retrocausalite.

Les futurs simules ne sont pas reinjectes dans la dynamique presente.
Les tests d'invariance sont des controles logiciels.
"""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Parametres d'origine
# ---------------------------------------------------------------------

SEED = 1234

N = 80
DIM_H = 12
DIM_C = 4
DIM_M = 16

DT = 0.02
DUREE = 300.0
N_PAS = int(round(DUREE / DT))

TAU_E = 1.0
TAU_I = 0.65
TAU_C = 4.0
TAU_H = 2.0
TAU_M = 12.0

D_E = 0.08
D_I = 0.05

SIGMA_SPATIALE = 0.20
BRUIT_HIPPOCAMPIQUE = 0.025
BRUIT_FUTUR = 0.025

Q_REVE = 0.80

T0_CAUSALITE = 150.0
PAS_T0 = int(round(T0_CAUSALITE / DT))

GRAINE_FUTUR_A = 1
GRAINE_FUTUR_B = 2

DECALAGE_CONTROLE = 25.0

DOSSIER_SCRIPT = Path(__file__).resolve().parent

DOSSIER_SORTIE = (
    DOSSIER_SCRIPT
    / "resultats"
    / "prototype_futur_passe"
)


# ---------------------------------------------------------------------
# Fonctions numeriques
# ---------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def normaliser_lignes(matrice):
    sommes = matrice.sum(axis=1, keepdims=True)

    return np.divide(
        matrice,
        sommes,
        out=np.zeros_like(matrice),
        where=sommes > 0.0,
    )


def cosinus(a, b):
    norme_a = np.linalg.norm(a)
    norme_b = np.linalg.norm(b)

    if norme_a <= 1e-12 or norme_b <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / (norme_a * norme_b))


def construire_graphe(points):
    differences = points[:, None, :] - points[None, :, :]
    distances2 = np.sum(differences ** 2, axis=2)

    adjacency = np.exp(
        -distances2 / (2.0 * SIGMA_SPATIALE ** 2)
    )

    np.fill_diagonal(adjacency, 0.0)

    adjacency = normaliser_lignes(adjacency)
    laplacien = np.eye(len(points)) - adjacency

    return adjacency, laplacien


def initialiser_parametres(rng, points):
    n = len(points)

    adjacency, laplacien = construire_graphe(points)

    W_EE = 1.55 * adjacency
    W_IE = 1.25 * adjacency
    W_EI = 1.85 * np.eye(n)
    W_II = 0.45 * np.eye(n)

    W_HH = rng.normal(
        0.0,
        0.22 / np.sqrt(DIM_H),
        size=(DIM_H, DIM_H),
    )
    W_HH += 0.75 * np.eye(DIM_H)

    W_HC = rng.normal(
        0.0,
        0.35 / np.sqrt(DIM_C),
        size=(DIM_H, DIM_C),
    )

    W_HG = rng.normal(
        0.0,
        0.25,
        size=(DIM_H, 2),
    )

    W_CC = rng.normal(
        0.0,
        0.25 / np.sqrt(DIM_C),
        size=(DIM_C, DIM_C),
    )
    W_CC += 0.80 * np.eye(DIM_C)

    W_CG = rng.normal(
        0.0,
        0.30,
        size=(DIM_C, 2),
    )

    W_CH = rng.normal(
        0.0,
        0.18 / np.sqrt(DIM_H),
        size=(DIM_C, DIM_H),
    )

    decodeur_scene = rng.normal(
        0.0,
        0.90 / np.sqrt(DIM_H),
        size=(n, DIM_H),
    )

    W_MEM_G = rng.normal(
        0.0,
        0.35,
        size=(DIM_M, 2),
    )

    W_MEM_C = rng.normal(
        0.0,
        0.30 / np.sqrt(DIM_C),
        size=(DIM_M, DIM_C),
    )

    W_MEM_H = rng.normal(
        0.0,
        0.20 / np.sqrt(DIM_H),
        size=(DIM_M, DIM_H),
    )

    return {
        "adjacency": adjacency,
        "laplacien": laplacien,
        "W_EE": W_EE,
        "W_IE": W_IE,
        "W_EI": W_EI,
        "W_II": W_II,
        "W_HH": W_HH,
        "W_HC": W_HC,
        "W_HG": W_HG,
        "W_CC": W_CC,
        "W_CG": W_CG,
        "W_CH": W_CH,
        "decodeur_scene": decodeur_scene,
        "W_MEM_G": W_MEM_G,
        "W_MEM_C": W_MEM_C,
        "W_MEM_H": W_MEM_H,
    }


def encoder_memoire(
    etat_global,
    contexte,
    etat_hippocampique,
    parametres,
):
    entree = (
        parametres["W_MEM_G"] @ etat_global
        + parametres["W_MEM_C"] @ contexte
        + parametres["W_MEM_H"] @ etat_hippocampique
    )

    return np.tanh(entree)


def generer_futur(
    etat_hippocampique,
    contexte,
    etat_global,
    parametres,
    rng_futur,
):
    bruit_futur = BRUIT_FUTUR * rng_futur.normal(size=DIM_H)

    return np.tanh(
        parametres["W_HH"] @ etat_hippocampique
        + parametres["W_HC"] @ contexte
        + parametres["W_HG"] @ etat_global
        + bruit_futur
    )


# ---------------------------------------------------------------------
# Dynamique du modele
# ---------------------------------------------------------------------

def simuler(
    rng,
    points,
    parametres,
    graine_futur=GRAINE_FUTUR_A,
):
    n = len(points)

    L = parametres["laplacien"]

    W_EE = parametres["W_EE"]
    W_IE = parametres["W_IE"]
    W_EI = parametres["W_EI"]
    W_II = parametres["W_II"]

    W_HH = parametres["W_HH"]
    W_HC = parametres["W_HC"]
    W_HG = parametres["W_HG"]

    W_CC = parametres["W_CC"]
    W_CG = parametres["W_CG"]
    W_CH = parametres["W_CH"]

    decodeur_scene = parametres["decodeur_scene"]

    rng_futur = np.random.default_rng(graine_futur)

    E = 0.45 + 0.01 * rng.normal(size=n)
    I = 0.35 + 0.01 * rng.normal(size=n)

    C = np.array([0.8, 0.4, 0.2, 0.6])
    H = 0.10 * rng.normal(size=DIM_H)

    M_passe = np.zeros(DIM_M)

    contexte_externe = np.array([0.60, 0.20, 0.10, 0.50])

    temps = np.arange(N_PAS) * DT

    historique_E = np.zeros((N_PAS, n))
    historique_I = np.zeros((N_PAS, n))
    historique_C = np.zeros((N_PAS, DIM_C))
    historique_H = np.zeros((N_PAS, DIM_H))
    historique_scene = np.zeros((N_PAS, n))
    historique_M_passe = np.zeros((N_PAS, DIM_M))
    historique_M_futur = np.zeros((N_PAS, DIM_M))
    historique_continuite = np.zeros(N_PAS)
    historique_stabilite = np.zeros(N_PAS)

    test_causalite = None

    centre = np.array([0.50, 0.50])
    distance2 = np.sum((points - centre) ** 2, axis=1)

    entree_spatiale = np.exp(
        -distance2 / (2.0 * 0.12 ** 2)
    )

    for pas in range(N_PAS):
        etat_global = np.array([E.mean(), I.mean()])

        # 1. Hippocampe.
        bruit_H = BRUIT_HIPPOCAMPIQUE * rng.normal(size=DIM_H)

        entree_H = (
            W_HH @ H
            + W_HC @ C
            + W_HG @ etat_global
            + bruit_H
        )

        dH = (-H + np.tanh(entree_H)) / TAU_H
        H = H + DT * dH

        # 2. Scene spatiale.
        scene = sigmoid(decodeur_scene @ H)
        scene_top_down = Q_REVE * scene

        # 3. Contexte cingulaire.
        entree_C = (
            W_CC @ C
            + W_CG @ etat_global
            + W_CH @ H
            + contexte_externe
        )

        dC = (-C + np.tanh(entree_C)) / TAU_C
        C = C + DT * dC

        # 4. Reseau spatial E/I.
        entree_sensorielle = (
            (1.0 - Q_REVE) * entree_spatiale
        )

        entree_E = (
            W_EE @ E
            - W_EI @ I
            + 0.75 * scene_top_down
            + 0.25 * entree_sensorielle
            - D_E * (L @ E)
        )

        entree_I = (
            W_IE @ E
            - W_II @ I
            + 0.20 * scene_top_down
            - D_I * (L @ I)
        )

        dE = (-E + sigmoid(entree_E)) / TAU_E
        dI = (-I + sigmoid(entree_I)) / TAU_I

        E = np.clip(E + DT * dE, 0.0, 1.0)
        I = np.clip(I + DT * dI, 0.0, 1.0)

        # 5. Memoire courante et proposition future.
        etat_global_actualise = np.array([
            E.mean(),
            I.mean(),
        ])

        memoire_cible = encoder_memoire(
            etat_global_actualise,
            C,
            H,
            parametres,
        )

        dM = (-M_passe + memoire_cible) / TAU_M
        M_passe = M_passe + DT * dM

        H_futur = generer_futur(
            H,
            C,
            etat_global_actualise,
            parametres,
            rng_futur,
        )

        M_futur_passe = encoder_memoire(
            etat_global_actualise,
            C,
            H_futur,
            parametres,
        )

        continuite = cosinus(M_passe, M_futur_passe)

        # 6. Stabilite locale.
        perturbations = (
            H[None, :]
            + 0.05 * rng.normal(size=(8, DIM_H))
        )

        scenes_perturbees = sigmoid(
            perturbations @ decodeur_scene.T
        )

        stabilite_scene = float(
            np.mean([
                cosinus(scene, scene_perturbee)
                for scene_perturbee in scenes_perturbees
            ])
        )

        # 7. Controle local d'absence de mutation.
        if pas == PAS_T0:
            etat_t0 = {
                "E": E.copy(),
                "I": I.copy(),
                "C": C.copy(),
                "H": H.copy(),
                "M_passe": M_passe.copy(),
            }

            H_futur_A = generer_futur(
                etat_t0["H"],
                etat_t0["C"],
                etat_global_actualise,
                parametres,
                np.random.default_rng(GRAINE_FUTUR_A),
            )

            H_futur_B = generer_futur(
                etat_t0["H"],
                etat_t0["C"],
                etat_global_actualise,
                parametres,
                np.random.default_rng(GRAINE_FUTUR_B),
            )

            M_futur_A = encoder_memoire(
                etat_global_actualise,
                etat_t0["C"],
                H_futur_A,
                parametres,
            )

            M_futur_B = encoder_memoire(
                etat_global_actualise,
                etat_t0["C"],
                H_futur_B,
                parametres,
            )

            difference_passes = max(
                np.max(np.abs(etat_t0["E"] - E)),
                np.max(np.abs(etat_t0["I"] - I)),
                np.max(np.abs(etat_t0["C"] - C)),
                np.max(np.abs(etat_t0["M_passe"] - M_passe)),
            )

            difference_futures = np.linalg.norm(
                M_futur_A - M_futur_B
            )

            if difference_passes > 1e-12:
                raise RuntimeError(
                    "Le futur simule a modifie un etat passe."
                )

            test_causalite = {
                "temps_t0": T0_CAUSALITE,
                "difference_etats_passes": float(difference_passes),
                "difference_memories_futures": float(difference_futures),
            }

        historique_E[pas] = E
        historique_I[pas] = I
        historique_C[pas] = C
        historique_H[pas] = H
        historique_scene[pas] = scene
        historique_M_passe[pas] = M_passe
        historique_M_futur[pas] = M_futur_passe
        historique_continuite[pas] = continuite
        historique_stabilite[pas] = stabilite_scene

    return {
        "temps": temps,
        "E": historique_E,
        "I": historique_I,
        "C": historique_C,
        "H": historique_H,
        "scene": historique_scene,
        "M_passe": historique_M_passe,
        "M_futur_passe": historique_M_futur,
        "continuite": historique_continuite,
        "stabilite_scene": historique_stabilite,
        "test_causalite": test_causalite,
        "points": points,
    }


# ---------------------------------------------------------------------
# Controles et mesures
# ---------------------------------------------------------------------

def ajouter_controle_temporel(resultats):
    m_passe = resultats["M_passe"]
    m_futur = resultats["M_futur_passe"]

    decalage_controle = int(round(DECALAGE_CONTROLE / DT))
    continuite_controle = np.zeros(N_PAS)

    for t in range(N_PAS):
        t_controle = (t + decalage_controle) % N_PAS

        continuite_controle[t] = cosinus(
            m_passe[t],
            m_futur[t_controle],
        )

    resultats["continuite_controle"] = continuite_controle

    return resultats


def mesurer_dynamique_tardive(resultats):
    periode_analyse = int(round(50.0 / DT))

    scene_tardive = resultats["scene"][-periode_analyse:]
    H_tardif = resultats["H"][-periode_analyse:]

    variation_scene = float(
        np.mean(
            np.linalg.norm(
                np.diff(scene_tardive, axis=0),
                axis=1,
            )
        )
    )

    variation_H = float(
        np.mean(
            np.linalg.norm(
                np.diff(H_tardif, axis=0),
                axis=1,
            )
        )
    )

    continuite_tardive = float(
        np.mean(
            resultats["continuite"][-periode_analyse:]
        )
    )

    continuite_controle_tardive = float(
        np.mean(
            resultats["continuite_controle"][-periode_analyse:]
        )
    )

    return {
        "variation_scene": variation_scene,
        "variation_H": variation_H,
        "continuite_tardive": continuite_tardive,
        "continuite_controle_tardive": continuite_controle_tardive,
    }


def tester_non_signalisation(points, parametres):
    """
    Controle logiciel : changer la graine des propositions futures
    ne doit pas changer les etats presentes et passes.

    Ce controle n'est pas une experience physique.
    """
    rng_A = np.random.default_rng(SEED)
    rng_B = np.random.default_rng(SEED)

    trajectoire_A = simuler(
        rng_A,
        points,
        parametres,
        graine_futur=GRAINE_FUTUR_A,
    )

    trajectoire_B = simuler(
        rng_B,
        points,
        parametres,
        graine_futur=GRAINE_FUTUR_B,
    )

    difference_avant_t0 = max(
        np.max(
            np.abs(
                trajectoire_A[champ][:PAS_T0 + 1]
                - trajectoire_B[champ][:PAS_T0 + 1]
            )
        )
        for champ in ("E", "I", "C", "M_passe")
    )

    if difference_avant_t0 > 1e-12:
        raise RuntimeError(
            "Le choix du futur simule modifie le passe : "
            "fuite d'information future."
        )

    return {
        "difference_avant_t0": float(difference_avant_t0),
        "temps_t0": T0_CAUSALITE,
    }


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------

def tracer_resultats(resultats):
    temps = resultats["temps"]

    E_moyen = resultats["E"].mean(axis=1)
    I_moyen = resultats["I"].mean(axis=1)
    scene_moyenne = resultats["scene"].mean(axis=1)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        sharex=True,
    )

    axes[0].plot(temps, E_moyen, label="E moyen")
    axes[0].plot(temps, I_moyen, label="I moyen")
    axes[0].set_ylabel("Activite")
    axes[0].set_title("Reseau neuronal excitateur-inhibiteur")
    axes[0].legend()

    axes[1].plot(temps, resultats["C"])
    axes[1].set_ylabel("Contexte C")
    axes[1].set_title("Etat contextuel cingulaire")

    axes[2].plot(
        temps,
        scene_moyenne,
        label="Intensite moyenne de la scene",
    )

    axes[2].plot(
        temps,
        resultats["continuite"],
        label="Continuite futur-passe",
    )

    axes[2].plot(
        temps,
        resultats["continuite_controle"],
        label="Controle decale",
        linestyle="--",
    )

    axes[2].set_ylabel("Indice")
    axes[2].set_title(
        "Scene hippocampique et continuite temporelle"
    )
    axes[2].legend()

    axes[3].plot(
        temps,
        resultats["stabilite_scene"],
        color="tab:green",
        label="Stabilite locale de la scene",
    )

    axes[3].set_ylabel("Similarite")
    axes[3].set_xlabel("Temps du modele")
    axes[3].set_title(
        "Stabilite de l'attracteur spatio-temporel"
    )
    axes[3].legend()

    fig.tight_layout()

    fig.savefig(
        DOSSIER_SORTIE / "diagnostics.png",
        dpi=150,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------

def main():
    DOSSIER_SORTIE.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Prototype du futur passe ===", flush=True)
    print(f"Noeuds spatiaux : {N}", flush=True)
    print(f"Dimension H : {DIM_H}", flush=True)
    print(f"Dimension C : {DIM_C}", flush=True)
    print(f"Duree : {DUREE}", flush=True)
    print("Simulation principale...", flush=True)

    rng = np.random.default_rng(SEED)

    points = rng.uniform(
        0.0,
        1.0,
        size=(N, 2),
    )

    parametres = initialiser_parametres(rng, points)

    resultats = simuler(
        rng,
        points,
        parametres,
        graine_futur=GRAINE_FUTUR_A,
    )

    resultats = ajouter_controle_temporel(resultats)
    mesures = mesurer_dynamique_tardive(resultats)

    print(
        "Controle logiciel de non-signalisation...",
        flush=True,
    )

    test_non_signalisation = tester_non_signalisation(
        points,
        parametres,
    )

    np.savez_compressed(
        DOSSIER_SORTIE / "trajectoire.npz",
        **{
            cle: valeur
            for cle, valeur in resultats.items()
            if cle != "test_causalite"
        },
    )

    tracer_resultats(resultats)

    nombre_final = max(
        1,
        int(round(0.10 * N_PAS)),
    )

    continuite_finale = np.mean(
        resultats["continuite"][-nombre_final:]
    )

    continuite_controle_finale = np.mean(
        resultats["continuite_controle"][-nombre_final:]
    )

    stabilite_finale = np.mean(
        resultats["stabilite_scene"][-nombre_final:]
    )

    print("\n=== Resultats ===")

    print(
        "Continuite futur-passe finale : "
        f"{continuite_finale:.6f}"
    )

    print(
        "Continuite controle decalee finale : "
        f"{continuite_controle_finale:.6f}"
    )

    print(
        "Stabilite finale de la scene : "
        f"{stabilite_finale:.6f}"
    )

    print("\n=== Dynamique tardive ===")

    print(
        "Variation temporelle de la scene : "
        f"{mesures['variation_scene']:.6e}"
    )

    print(
        "Variation temporelle de H : "
        f"{mesures['variation_H']:.6e}"
    )

    print(
        "Continuite futur-passe tardive : "
        f"{mesures['continuite_tardive']:.6f}"
    )

    print(
        "Continuite controle decalee tardive : "
        f"{mesures['continuite_controle_tardive']:.6f}"
    )

    print("\n=== Controle local d'invariance ===")

    print(
        "Difference des etats passes apres generation "
        "de deux futurs : "
        f"{resultats['test_causalite']['difference_etats_passes']:.6e}"
    )

    print(
        "Difference des memoires futures : "
        f"{resultats['test_causalite']['difference_memories_futures']:.6e}"
    )

    print("\n=== Controle logiciel de non-signalisation ===")

    print(
        "Difference avant t0 entre deux futurs simules : "
        f"{test_non_signalisation['difference_avant_t0']:.6e}"
    )

    print(f"\nResultats : {DOSSIER_SORTIE}")

    print(
        "Le futur simule est une hypothese calculee depuis "
        "le present ; il ne modifie aucun etat passe."
    )

    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
