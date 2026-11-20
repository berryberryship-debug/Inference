cat > stabilite_neuronale.py <<'PYTHON'
#!/usr/bin/env python3
"""Audit spectral d'un modele E-I abstrait sur un graphe synthetique."""

import csv
import json
from pathlib import Path

import numpy as np

SEED = 42
N = 128
K = 10

TAU_E = 1.0
TAU_I = 2.0

W_EI = 12.0
W_IE = 10.0
W_II = 2.0

D_E = 0.05
D_I = 0.2

W_MIN = 0.0
W_MAX = 10.0
PAS = 0.1

TOLERANCE_JACOBIEN = 1e-6
TOLERANCE_FRONTIERE = 1e-6

DOSSIER = Path("resultats/stabilite_neuronale")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def construire_graphe(points):
    """Graphe k-plus-proches-voisins symetrise, sans boucles."""
    if points.shape != (N, 3):
        raise ValueError("points doit avoir la forme (N, 3).")

    differences = points[:, None, :] - points[None, :, :]
    distances_carrees = np.sum(differences**2, axis=2)
    np.fill_diagonal(distances_carrees, np.inf)

    indices = np.argsort(distances_carrees, axis=1)[:, :K]
    adjacency = np.zeros((N, N), dtype=bool)

    for i, voisins in enumerate(indices):
        adjacency[i, voisins] = True

    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float64)


def composantes_connexes(adjacency):
    """Nombre de composantes connexes par parcours en profondeur."""
    n = adjacency.shape[0]
    vus = np.zeros(n, dtype=bool)
    nombre = 0

    for depart in range(n):
        if vus[depart]:
            continue

        nombre += 1
        pile = [depart]
        vus[depart] = True

        while pile:
            noeud = pile.pop()
            voisins = np.flatnonzero(adjacency[noeud] > 0)
            for voisin in voisins:
                if not vus[voisin]:
                    vus[voisin] = True
                    pile.append(int(voisin))

    return nombre


def entrees(w_EE):
    """Entrees maintenant E = I = 0.5 a l'equilibre."""
    E = 0.5
    I = 0.5
    P = E - w_EE * E + W_EI * I
    Q = I - W_IE * E + W_II * I
    return P, Q


def jacobien_homogene(w_EE):
    """Jacobien du mode homogene autour de E = I = 0.5."""
    return np.array(
        [
            [-1.0 + w_EE / 4.0, -W_EI],
            [W_IE / TAU_I, -1.0 / TAU_I - W_II / TAU_I],
        ]
    )


def valeurs_propres(jacobien):
    valeurs = np.linalg.eigvals(jacobien)
    return np.sort_complex(valeurs)


def verifier_jacobien(adjacency, w_EE, epsilon=1e-6):
    """Compare le jacobien analytique au jacobien par differences finies."""
    P, Q = entrees(w_EE)
    E = 0.5 + epsilon * np.ones(N)
    I = 0.5 + epsilon * np.ones(N)

    def dynamique(E, I):
        dE = (
            -E
            + sigmoid(w_EE * E - W_EI * I + P)
            - D_E * (adjacency @ E)
        ) / TAU_E
        dI = (
            -I
            + sigmoid(W_IE * E - W_II * I + Q)
            - D_I * (adjacency @ I)
        ) / TAU_I
        return dE, dI

    E_plus, I_plus = dynamique(E, I)
    E_moins, I_moins = dynamique(0.5 - epsilon * np.ones(N), I)
    I_2_plus, I_2_moins = dynamique(E, 0.5 + epsilon * np.ones(N)), dynamique(
        E, 0.5 - epsilon * np.ones(N)
    )

    jacobien_numerique = np.array(
        [
            [(E_plus - E_moins) / (2 * epsilon), (I_plus - I_moins) / (2 * epsilon)],
            [
                (I_2_plus[0] - I_2_moins[0]) / (2 * epsilon),
                (I_2_plus[1] - I_2_moins[1]) / (2 * epsilon),
            ],
        ]
    )

    return float(np.max(np.abs(jacobien_numerique - jacobien_homogene(w_EE))))


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Audit spectral excitation-inhibition ===")
    print("Graphe synthetique ; parametres non calibres.")

    rng = np.random.default_rng(SEED)
    points = rng.uniform(0.0, 1.0, size=(N, 3))
    adjacency = construire_graphe(points)
    arretes = int(adjacency.sum() // 2)

    print(f"Noeuds : {N}")
    print(f"Aretes : {arretes}")
    print(f"Composantes estimees : {composantes_connexes(adjacency)}")

    erreur_jacobien = verifier_jacobien(adjacency, W_MIN)
    print(f"Erreur maximale du jacobien : {erreur_jacobien:.3e}")

    if erreur_jacobien > TOLERANCE_JACOBIEN:
        raise RuntimeError("Le jacobien numerique ne correspond pas.")

    lignes = []
    dernier_stable = None
    premier_instable = None
    mode_dominant = None
    croissance_maximale = -np.inf

    valeurs_w = np.arange(W_MIN, W_MAX + PAS / 2, PAS)

    for w in valeurs_w:
        jacobien = jacobien_homogene(w)
        valeurs = valeurs_propres(jacobien)
        partie_reelle_max = float(np.max(valeurs.real))

        if partie_reelle_max < -TOLERANCE_FRONTIERE:
            statut = "stable"
            if dernier_stable is None or w > dernier_stable:
                dernier_stable = float(w)
        elif partie_reelle_max > TOLERANCE_FRONTIERE:
            statut = "instable"
            if premier_instable is None:
                premier_instable = float(w)
                mode_dominant = int(np.argmax(valeurs.real))
        else:
            statut = "proche_frontiere"

        if partie_reelle_max > croissance_maximale:
            croissance_maximale = partie_reelle_max

        lignes.append(
            {
                "w_EE": float(w),
                "partie_reelle_max": partie_reelle_max,
                "valeur_propre_1": complex(valeurs[0]),
                "valeur_propre_2": complex(valeurs[1]),
                "statut": statut,
            }
        )

    with (DOSSIER / "audit_spectral.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "w_EE",
                "partie_reelle_max",
                "valeur_propre_1_re",
                "valeur_propre_1_im",
                "valeur_propre_2_re",
                "valeur_propre_2_im",
                "statut",
            ]
        )
        for ligne in lignes:
            v1, v2 = ligne["valeur_propre_1"], ligne["valeur_propre_2"]
            writer.writerow(
                [
                    ligne["w_EE"],
                    ligne["partie_reelle_max"],
                    v1.real,
                    v1.imag,
                    v2.real,
                    v2.imag,
                    ligne["statut"],
                ]
            )

    resume = {
        "statut": "Audit spectral synthetique, non calibre",
        "seed": SEED,
        "N": N,
        "K": K,
        "aretes": arretes,
        "tau_E": TAU_E,
        "tau_I": TAU_I,
        "w_EI": W_EI,
        "w_IE": W_IE,
        "w_II": W_II,
        "D_E": D_E,
        "D_I": D_I,
        "erreur_max_jacobien": erreur_jacobien,
        "dernier_stable": dernier_stable,
        "premier_instable": premier_instable,
        "mode_dominant_premier_instable": mode_dominant,
        "croissance_maximale": croissance_maximale,
    }

    with (DOSSIER / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, allow_nan=False)

    if dernier_stable is not None and premier_instable is not None:
        print(
            f"Passage stable -> instable encadre par : "
            f"{dernier_stable:.3f} et {premier_instable:.3f}"
        )
        print(f"Premier point instable : w_EE={premier_instable:.3f}")
        print(f"Mode dominant a ce point : {mode_dominant}")

    print(f"Croissance maximale : {croissance_maximale:+.6f}")
    print(f"Resultats : {DOSSIER}")
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
