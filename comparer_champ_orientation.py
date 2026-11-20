cd ~

cat > comparer_champs_orientation.py <<'PYTHON'
#!/usr/bin/env python3
"""Compare les effets directionnels des champs +x et -x."""

import json
from pathlib import Path

DOSSIER_PLUS = Path(
    "resultats/experience_orientation_sensibilite"
)
DOSSIER_MOINS = Path(
    "resultats/experience_orientation_sensibilite_inverse"
)


def charger(dossier):
    fichier = dossier / "resume.json"

    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier}")

    with fichier.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    plus = charger(DOSSIER_PLUS)
    moins = charger(DOSSIER_MOINS)

    resultats_plus = {
        (
            r["kappa"],
            r["amplitude_impulsion"],
            r["fraction_seuil"],
        ): r
        for r in plus["resultats"]
    }

    resultats_moins = {
        (
            r["kappa"],
            r["amplitude_impulsion"],
            r["fraction_seuil"],
        ): r
        for r in moins["resultats"]
    }

    cles = sorted(
        set(resultats_plus) & set(resultats_moins)
    )

    print("=== Comparaison des champs +x et -x ===")
    print(
        "kappa | amplitude | seuil | "
        "E+ / E- | I+ / I- | "
        "avant+ / arriere+ | avant- / arriere-"
    )
    print("-" * 100)

    for cle in cles:
        r_plus = resultats_plus[cle]
        r_moins = resultats_moins[cle]

        kappa, amplitude, seuil = cle

        avant_plus = r_plus["direction_E"]["avant"]["nombre_noeuds"]
        arriere_plus = r_plus["direction_E"]["arriere"]["nombre_noeuds"]
        avant_moins = r_moins["direction_E"]["avant"]["nombre_noeuds"]
        arriere_moins = r_moins["direction_E"]["arriere"]["nombre_noeuds"]

        print(
            f"{kappa:g} | {amplitude:g} | {seuil:g} | "
            f"{r_plus['nombre_arrivees_E']} / "
            f"{r_moins['nombre_arrivees_E']} | "
            f"{r_plus['nombre_arrivees_I']} / "
            f"{r_moins['nombre_arrivees_I']} | "
            f"{avant_plus} / {arriere_plus} | "
            f"{avant_moins} / {arriere_moins}"
        )

    sortie = Path(
        "resultats/comparaison_champs_orientation.json"
    )
    sortie.parent.mkdir(parents=True, exist_ok=True)

    rapport = {
        "statut": "Comparaison des champs +x et -x",
        "nombre_combinaisons": len(cles),
        "resultats_plus": plus["resultats"],
        "resultats_moins": moins["resultats"],
    }

    with sortie.open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"\nRapport : {sortie}")
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
