cd ~

cat > simulation_tres_longue.py <<'PYTHON'
#!/usr/bin/env python3
"""Simulation tres longue a w_EE = 8 pour tester la stationnarite tardive."""

import json
from pathlib import Path

import numpy as np
import stabilite_neuronale as modele

W_EE = 8.0
DUREE = 1000.0
DT = 0.05
AMPLITUDE = 1e-4
SEED_PERTURBATION = 123
INTERVALLE_ENREGISTREMENT = 10

DOSSIER = Path("resultats/simulation_tres_longue")


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


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Simulation tres longue, w_EE = 8 ===")
    print(f"Duree : {DUREE} ; pas : {DT}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)
    L = modele.laplacien(adjacency)

    rng_perturbation = np.random.default_rng(SEED_PERTURBATION)
    etat_initial = 0.5 + AMPLITUDE * rng_perturbation.normal(
        size=(2, modele.N)
    )

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

        if (numero_pas + 1) % INTERVALLE_ENREGISTREMENT == 0:
            temps.append((numero_pas + 1) * DT)
            E.append(y[0].copy())
            I.append(y[1].copy())

        if (numero_pas + 1) % 4000 == 0:
            print(
                f"Temps : {(numero_pas + 1) * DT:.1f} / {DUREE:.1f}",
                flush=True,
            )

    temps = np.asarray(temps)
    E = np.asarray(E)
    I = np.asarray(I)

    np.savez_compressed(
        DOSSIER / "trajectoire_tres_longue.npz",
        temps=temps,
        E=E,
        I=I,
    )

    resume = {
        "statut": "Simulation tres longue synthetique, non calibree",
        "w_EE": W_EE,
        "duree": DUREE,
        "dt": DT,
        "amplitude_perturbation": AMPLITUDE,
        "seed_graphe": modele.SEED,
        "seed_perturbation": SEED_PERTURBATION,
        "intervalles_enregistrement": INTERVALLE_ENREGISTREMENT,
        "N": modele.N,
        "K": modele.K,
    }

    with (DOSSIER / "configuration.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(resume, f, indent=2, allow_nan=False)

    print(f"Termine. Resultats : {DOSSIER}")
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
