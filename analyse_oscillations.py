cd ~

cat > analyse_oscillations.py <<'PYTHON'
#!/usr/bin/env python3
"""Analyse descriptive des trajectoires deja enregistrees."""

import json
from pathlib import Path

import numpy as np

DOSSIER = Path("resultats/dynamique_neuronale")


def analyser_signal(t, signal):
    amplitude = float(np.ptp(signal))
    resultat = {
        "minimum": float(signal.min()),
        "maximum": float(signal.max()),
        "amplitude_crete_a_crete": amplitude,
    }

    # Seuil descriptif : eviter d'interpreter un signal quasi constant.
    if amplitude < 1e-6:
        resultat["analyse_pics"] = "Amplitude inferieure a 1e-6."
        return resultat

    indices = np.flatnonzero(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
    ) + 1

    resultat["nombre_pics"] = int(indices.size)
    resultat["temps_pics"] = t[indices].tolist()

    if indices.size >= 3:
        intervalles = np.diff(t[indices])
        moyenne = float(intervalles.mean())
        resultat["intervalle_moyen_entre_pics"] = moyenne
        resultat["ecart_type_intervalles"] = float(intervalles.std())
        resultat["coefficient_variation_intervalles"] = float(
            intervalles.std() / moyenne
        )
    else:
        resultat["analyse_pics"] = (
            "Moins de trois pics : intervalles insuffisants."
        )

    return resultat


def analyser_condition(nom):
    fichier = DOSSIER / f"trajectoire_{nom}.npz"
    with np.load(fichier, allow_pickle=False) as donnees:
        t = donnees["temps"]
        E = donnees["E"]
        I = donnees["I"]

    if E.shape != I.shape or E.shape[0] != len(t):
        raise ValueError(f"Dimensions incoherentes : {fichier}")

    if not all(np.all(np.isfinite(x)) for x in (t, E, I)):
        raise ValueError(f"Valeurs non finies : {fichier}")

    if not np.all(np.diff(t) > 0):
        raise ValueError("Les temps doivent etre strictement croissants.")

    moyennes = {"E": E.mean(axis=1), "I": I.mean(axis=1)}
    debut, fin = float(t[0]), float(t[-1])
    duree = fin - debut
    fenetres = []

    print(f"\n=== {nom} ===")

    for numero in range(4):
        borne_gauche = debut + numero * duree / 4
        borne_droite = debut + (numero + 1) * duree / 4
        masque = (t >= borne_gauche) & (
            (t <= borne_droite) if numero == 3 else (t < borne_droite)
        )

        if masque.sum() < 3:
            raise ValueError("Fenetre trop courte pour l'analyse.")

        bilan = {
            "debut": borne_gauche,
            "fin": borne_droite,
            "E": analyser_signal(t[masque], moyennes["E"][masque]),
            "I": analyser_signal(t[masque], moyennes["I"][masque]),
            "variance_spatiale_E_moyenne": float(
                E[masque].var(axis=1).mean()
            ),
            "variance_spatiale_I_moyenne": float(
                I[masque].var(axis=1).mean()
            ),
        }
        fenetres.append(bilan)

        print(f"Fenetre {borne_gauche:g} a {borne_droite:g}")
        for population in ("E", "I"):
            mesures = bilan[population]
            print(
                f"  {population} : amplitude = "
                f"{mesures['amplitude_crete_a_crete']:.6g}"
            )
            if "intervalle_moyen_entre_pics" in mesures:
                print(
                    "    Intervalle moyen entre pics = "
                    f"{mesures['intervalle_moyen_entre_pics']:.6g}"
                    " ; CV = "
                    f"{mesures['coefficient_variation_intervalles']:.3g}"
                )
            else:
                print(
                    "    " + mesures.get(
                        "analyse_pics",
                        "Intervalles insuffisants."
                    )
                )

        print(
            "  Variances spatiales moyennes E / I : "
            f"{bilan['variance_spatiale_E_moyenne']:.3e} / "
            f"{bilan['variance_spatiale_I_moyenne']:.3e}"
        )

    return {"fenetres": fenetres}


def main():
    rapport = {
        "description": "Analyse descriptive sur quatre fenetres.",
        "temps": "Unites du modele, sans calibration biologique.",
        "limites": [
            "Pics detectes sur les seuls instants sauvegardes.",
            "Aucune interpolation des instants des pics.",
            "Le seuil 1e-6 est un choix descriptif.",
            "Cette analyse ne prouve pas un cycle limite.",
        ],
        "conditions": {},
    }

    for nom in ("sous_seuil", "au_dessus"):
        rapport["conditions"][nom] = analyser_condition(nom)

    sortie = DOSSIER / "analyse_oscillations.json"
    with sortie.open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"\nRapport enregistre : {sortie}")
    print("Analyse descriptive uniquement ; aucun diagnostic biologique.")


if __name__ == "__main__":
    main()
PYTHON
