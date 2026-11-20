cd ~

cat > experience_longue.py <<'PYTHON'
#!/usr/bin/env python3
"""Simulation longue a w_EE = 8 pour caracteriser le regime tardif."""

import csv
import json
from pathlib import Path

import numpy as np
import stabilite_neuronale as modele

W_EE = 8.0
DUREE = 400.0
DT = 0.05
AMPLITUDE = 1e-4
SEED_PERTURBATION = 123

INTERVALLE_ENREGISTREMENT = 10
DOSSIER = Path("resultats/experience_longue")


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
            raise RuntimeError("Etat non fini pendant la simulation.")

        if np.any(y < -1e-9) or np.any(y > 1.0 + 1e-9):
            raise RuntimeError("Etat hors de [0, 1] : verifier le pas.")

        if (numero_pas + 1) % INTERVALLE_ENREGISTREMENT == 0:
            temps.append((numero_pas + 1) * DT)
            E.append(y[0].copy())
            I.append(y[1].copy())

    return (
        np.asarray(temps),
        np.asarray(E),
        np.asarray(I),
    )


def analyser_fenetre(temps, E, I, debut, fin):
    masque = (temps >= debut) & (temps <= fin)
    t = temps[masque]
    moyenne_E = E[masque].mean(axis=1)
    moyenne_I = I[masque].mean(axis=1)

    def pics(signal):
        if len(signal) < 3:
            return np.array([], dtype=int)

        indices = np.flatnonzero(
            (signal[1:-1] > signal[:-2])
            & (signal[1:-1] >= signal[2:])
        ) + 1
        return indices

    indices_E = pics(moyenne_E)
    indices_I = pics(moyenne_I)

    def mesures(signal, indices):
        resultat = {
            "amplitude_crete_a_crete": float(np.ptp(signal)),
            "nombre_pics": int(indices.size),
        }

        if indices.size >= 3:
            intervalles = np.diff(t[indices])
            moyenne = float(intervalles.mean())
            resultat["intervalle_moyen_entre_pics"] = moyenne
            resultat["ecart_type_intervalles"] = float(intervalles.std())
            resultat["coefficient_variation_intervalles"] = float(
                intervalles.std() / moyenne
            )

        return resultat

    return {
        "debut": float(debut),
        "fin": float(fin),
        "E": mesures(moyenne_E, indices_E),
        "I": mesures(moyenne_I, indices_I),
        "variance_spatiale_E_moyenne": float(E[masque].var(axis=1).mean()),
        "variance_spatiale_I_moyenne": float(I[masque].var(axis=1).mean()),
        "rms_equilibre_final": float(
            np.sqrt(np.mean((np.vstack([E[masque][-1], I[masque][-1]]) - 0.5)**2))
        ),
    }


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Simulation longue, w_EE = 8 ===")
    print(f"Duree : {DUREE} ; pas : {DT}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)
    L = modele.laplacien(adjacency)

    rng_perturbation = np.random.default_rng(SEED_PERTURBATION)
    etat_initial = 0.5 + AMPLITUDE * (
        np.ones((2, modele.N))
        + rng_perturbation.normal(size=(2, modele.N))
    )

    temps, E, I = simuler(L, etat_initial)

    np.savez_compressed(
        DOSSIER / "trajectoire_longue.npz",
        temps=temps,
        E=E,
        I=I,
    )

    fenetres = []
    bornes = [(240.0, 280.0), (280.0, 320.0), (320.0, 360.0), (360.0, 400.0)]

    print("\nFenetres tardives :")
    for debut, fin in bornes:
        bilan = analyser_fenetre(temps, E, I, debut, fin)
        fenetres.append(bilan)

        print(f"[{debut:g}, {fin:g}]")
        for population in ("E", "I"):
            mesures = bilan[population]
            print(
                f"  {population} : amplitude = "
                f"{mesures['amplitude_crete_a_crete']:.6g}"
                f" ; pics = {mesures['nombre_pics']}"
            )

            if "intervalle_moyen_entre_pics" in mesures:
                print(
                    "    Intervalle moyen = "
                    f"{mesures['intervalle_moyen_entre_pics']:.6g}"
                    " ; CV = "
                    f"{mesures['coefficient_variation_intervalles']:.3g}"
                )

        print(
            "  Variances spatiales moyennes E / I : "
            f"{bilan['variance_spatiale_E_moyenne']:.3e} / "
            f"{bilan['variance_spatiale_I_moyenne']:.3e}"
        )

    resume = {
        "statut": "Simulation longue synthetique, non calibree",
        "w_EE": W_EE,
        "duree": DUREE,
        "dt": DT,
        "amplitude_perturbation": AMPLITUDE,
        "seed_graphe": modele.SEED,
        "seed_perturbation": SEED_PERTURBATION,
        "N": modele.N,
        "K": modele.K,
        "intervalles_enregistrement": INTERVALLE_ENREGISTREMENT,
        "fenetres": fenetres,
    }

    with (DOSSIER / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, allow_nan=False)

    print(f"\nResultats enregistres : {DOSSIER}")
    print(
        "Un regime periodique n'est etabli que si amplitudes et intervalles "
        "cessent de varier entre fenetres tardives."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
