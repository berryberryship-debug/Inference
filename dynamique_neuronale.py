cd ~

cat > dynamique_neuronale.py <<'PYTHON'
#!/usr/bin/env python3
"""Dynamique E-I non lineaire sur le graphe de l'audit spectral."""

import csv
import json
from pathlib import Path

import numpy as np
import stabilite_neuronale as modele

OUT = Path("resultats/dynamique_neuronale")
DUREE = 80.0
DT = 0.05
AMPLITUDE = 1e-4


def construire_rhs(L, w):
    P, Q = modele.entrees(w)

    def rhs(y):
        E, I = y
        dE = (
            -E
            + modele.sigmoid(w * E - modele.W_EI * I + P)
            - modele.D_E * (L @ E)
        ) / modele.TAU_E
        dI = (
            -I
            + modele.sigmoid(modele.W_IE * E - modele.W_II * I + Q)
            - modele.D_I * (L @ I)
        ) / modele.TAU_I
        return np.array([dE, dI])

    return rhs


def simuler(L, initial, w, dt, enregistrer_tous):
    rhs = construire_rhs(L, w)
    n = int(round(DUREE / dt))
    y = initial.copy()
    temps = [0.0]
    etats = [y.copy()]

    for k in range(n):
        a = rhs(y)
        b = rhs(y + dt * a / 2)
        c = rhs(y + dt * b / 2)
        d = rhs(y + dt * c)
        y = y + dt * (a + 2*b + 2*c + d) / 6

        if not np.all(np.isfinite(y)):
            raise RuntimeError("Etat non fini.")

        if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
            raise RuntimeError("Etat hors de [0, 1] : verifier le pas.")

        if (k + 1) % enregistrer_tous == 0 or k + 1 == n:
            temps.append((k + 1) * dt)
            etats.append(y.copy())

    return np.array(temps), np.array(etats)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Dynamique neuronale non lineaire ===", flush=True)

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0, 1, size=(modele.N, 3))
    A = modele.construire_graphe(points)
    L = np.diag(A.sum(axis=1)) - A

    # Perturbation commune aux deux conditions.
    # Une partie homogene excite aussi le mode spatial constant.
    rng_perturbation = np.random.default_rng(123)
    initial = 0.5 + AMPLITUDE * (
        np.ones((2, modele.N))
        + rng_perturbation.normal(size=(2, modele.N))
    )

    rapport = {
        "statut": "Simulation synthetique non calibree",
        "numpy_version": np.__version__,
        "duree": DUREE,
        "dt": DT,
        "seed_graphe": modele.SEED,
        "seed_perturbation": 123,
        "amplitude_perturbation": AMPLITUDE,
        "N": modele.N,
        "K": modele.K,
        "tau_E": modele.TAU_E,
        "tau_I": modele.TAU_I,
        "w_EI": modele.W_EI,
        "w_IE": modele.W_IE,
        "w_II": modele.W_II,
        "D_E": modele.D_E,
        "D_I": modele.D_I,
        "conditions": {},
    }

    for nom, w in [("sous_seuil", 6.0), ("au_dessus", 8.0)]:
        print(f"Simulation {nom}, w_EE={w}...", flush=True)

        temps, etats = simuler(L, initial, w, DT, 10)
        temps_fins, fins = simuler(L, initial, w, DT / 2, 20)

        if not np.allclose(temps, temps_fins):
            raise RuntimeError("Instants de comparaison differents.")

        rms_equilibre = np.sqrt(
            np.mean((etats - 0.5)**2, axis=(1, 2))
        )
        moyenne_E = etats[:, 0].mean(axis=1)
        moyenne_I = etats[:, 1].mean(axis=1)
        debut_queue = 3 * len(temps) // 4
        P, Q = modele.entrees(w)

        mesures = {
            "w_EE": w,
            "P": float(P),
            "Q": float(Q),
            "rms_equilibre_initial": float(rms_equilibre[0]),
            "rms_equilibre_final": float(rms_equilibre[-1]),
            "rms_equilibre_max": float(rms_equilibre.max()),
            "amplitude_moyenne_E_dernier_quart": float(
                np.ptp(moyenne_E[debut_queue:])
            ),
            "amplitude_moyenne_I_dernier_quart": float(
                np.ptp(moyenne_I[debut_queue:])
            ),
            "ecart_max_demi_pas_aux_instants_enregistres": float(
                np.max(np.abs(etats - fins))
            ),
        }
        rapport["conditions"][nom] = mesures

        np.savez_compressed(
            OUT / f"trajectoire_{nom}.npz",
            temps=temps,
            E=etats[:, 0],
            I=etats[:, 1],
        )

        with (OUT / f"mesures_{nom}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "temps", "moyenne_E", "moyenne_I",
                "variance_spatiale_E", "variance_spatiale_I",
                "rms_equilibre",
            ])
            writer.writerows(zip(
                temps, moyenne_E, moyenne_I,
                etats[:, 0].var(axis=1),
                etats[:, 1].var(axis=1),
                rms_equilibre,
            ))

        print(
            "  RMS initial/final : "
            f"{rms_equilibre[0]:.6g} / {rms_equilibre[-1]:.6g}"
        )
        print(
            "  Ecart maximal au demi-pas : "
            f"{mesures['ecart_max_demi_pas_aux_instants_enregistres']:.3e}"
        )

    np.savez_compressed(
        OUT / "configuration.npz",
        points=points,
        adjacency=A,
        etat_initial=initial,
    )

    with (OUT / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"Termine. Resultats : {OUT}")
    print("Les amplitudes sont mesurees sur une fenetre finie.")
    print("Elles ne prouvent pas a elles seules un cycle limite.")
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
