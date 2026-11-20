cd ~

cat > analyse_derives_longue.py <<'PYTHON'
#!/usr/bin/env python3
"""Diagnostics de derive sur huit fenetres tardives d'une trajectoire longue."""

import json
from pathlib import Path

import numpy as np

from analyse_derives import (
    analyser_derives,
    diagnostics_fenetres,
)

FICHIER_TRAJECTOIRE = Path(
    "resultats/simulation_tres_longue/trajectoire_tres_longue.npz"
)
DOSSIER = Path("resultats/analyse_derives_longue")

BORNES_TARDIVES = [
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 500.0),
    (500.0, 600.0),
    (600.0, 700.0),
    (700.0, 800.0),
    (800.0, 900.0),
    (900.0, 1000.0),
]


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    if not FICHIER_TRAJECTOIRE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_TRAJECTOIRE}. "
            "Lance d'abord simulation_tres_longue.py."
        )

    with np.load(FICHIER_TRAJECTOIRE, allow_pickle=False) as donnees:
        temps = donnees["temps"]
        E = donnees["E"]
        I = donnees["I"]

    if E.shape != I.shape or E.shape[0] != len(temps):
        raise ValueError("Dimensions incoherentes dans la trajectoire.")

    print("=== Diagnostics de derive, huit fenetres tardives ===")
    print(f"Duree totale : {temps[-1]:.1f}")

    diagnostics = diagnostics_fenetres(
        temps,
        E,
        I,
        BORNES_TARDIVES,
    )

    for diagnostic in diagnostics:
        print(
            f"\nFenetre [{diagnostic['debut']:g}, {diagnostic['fin']:g}]"
        )
        print(
            f"  Amplitude E / I : "
            f"{diagnostic['amplitude_E']:.8g} / "
            f"{diagnostic['amplitude_I']:.8g}"
        )
        print(
            f"  Periode E / I : "
            f"{diagnostic['periode_E']:.8g} / "
            f"{diagnostic['periode_I']:.8g}"
        )
        print(
            f"  Phase relative : {diagnostic['phase_deg']:.4f} deg ; "
            f"R = {diagnostic['R']:.5f}"
        )

    derives = analyser_derives(diagnostics)

    print("\n=== Tendances lineaires ===")

    for nom, resultat in derives.items():
        print(f"\n{nom}")
        print(f"  moyenne : {resultat['moyenne']:.8g}")
        print(
            f"  pente : {resultat['pente']:.8g} ; "
            f"IC95% : [{resultat['ic_bas']:.8g}, "
            f"{resultat['ic_haut']:.8g}]"
        )
        print(f"  p : {resultat['p_value']:.5g}")
        print(
            f"  excursion relative : "
            f"{100.0 * resultat['excursion_relative']:.5g}%"
        )

    rapport = {
        "statut": "Diagnostics de derive sur une trajectoire longue unique",
        "fichier_trajectoire": str(FICHIER_TRAJECTOIRE),
        "fenetres": diagnostics,
        "tendances": derives,
        "limites": [
            "Huit fenetres de 100 unites de temps.",
            "Environ neuf points par cycle a l'enregistrement.",
            "L'absence de significativite ne prouve pas une absence de derive.",
            "Aucune marge d'equivalence predefinie n'est imposee ici.",
            "La phase depend du filtrage et de la frequence dominante.",
        ],
    }

    with (DOSSIER / "derives.json").open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"\nRapport enregistre : {DOSSIER / 'derives.json'}")
    print(
        "Une phase stable doit montrer une pente de phase depliee "
        "proche de zero."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
