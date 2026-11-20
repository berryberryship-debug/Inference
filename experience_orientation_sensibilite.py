cd ~

python - <<'PYTHON'
from pathlib import Path
import re

fichier = Path("experience_orientation_sensibilite.py")
texte = fichier.read_text(encoding="utf-8")

# 1. Ajouter les convertisseurs s'ils sont absents.
if "def float_json(" not in texte:
    marqueur = "W_EE = 8.0"
    index = texte.index(marqueur)

    aide = '''def float_json(valeur):
    """Convertit en float JSON ; les valeurs non finies deviennent None."""
    if valeur is None:
        return None

    valeur = float(valeur)

    if not np.isfinite(valeur):
        return None

    return valeur


def nettoyer_json(objet):
    """Nettoie recursivement NaN et inf avant json.dump."""
    if isinstance(objet, dict):
        return {
            cle: nettoyer_json(valeur)
            for cle, valeur in objet.items()
        }

    if isinstance(objet, list):
        return [nettoyer_json(valeur) for valeur in objet]

    if isinstance(objet, tuple):
        return [nettoyer_json(valeur) for valeur in objet]

    if isinstance(objet, np.integer):
        return int(objet)

    if isinstance(objet, (np.floating, float)):
        return float_json(objet)

    if isinstance(objet, np.ndarray):
        return nettoyer_json(objet.tolist())

    return objet


'''
    texte = texte[:index] + aide + texte[index:]

# 2. Remplacer analyser_direction par une version sans NaN.
motif_fonction = (
    r"def analyser_direction\([\s\S]*?\n\ndef main\(\):"
)

nouvelle_fonction = '''def analyser_direction(
    points,
    directions,
    noeud_source,
    arrivees,
    temps_source,
):
    centre = points[noeud_source]

    direction_source = np.asarray(
        directions[noeud_source],
        dtype=float,
    )

    norme = np.linalg.norm(direction_source)

    if norme <= 0.0:
        raise ValueError("Direction nulle au noeud source.")

    direction_source /= norme

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
        ).astype(float)

        temps_relatifs = (
            np.asarray(arrivees[masque_valide])
            - float(temps_source)
        ).astype(float)

        entree = {
            "nombre_noeuds": int(len(distances)),
            "temps_median": None,
            "pente_temps_distance": None,
            "vitesse_apparente": None,
            "R2": None,
        }

        if len(distances) > 0:
            entree["temps_median"] = float_json(
                np.median(temps_relatifs)
            )

        distances_uniques = np.unique(
            np.round(distances, decimals=12)
        )

        if (
            len(distances) >= 3
            and len(distances_uniques) >= 2
            and np.ptp(distances) > 1e-12
        ):
            ajustement = linregress(
                distances,
                temps_relatifs,
            )

            pente = float_json(ajustement.slope)
            rvalue = float_json(ajustement.rvalue)

            entree["pente_temps_distance"] = pente

            if rvalue is not None:
                entree["R2"] = float_json(rvalue ** 2)

            if pente is not None and pente > 0.0:
                entree["vitesse_apparente"] = float_json(
                    1.0 / pente
                )

        resultat[nom] = entree

    return resultats


def main():'''

texte, nombre_fonction = re.subn(
    motif_fonction,
    nouvelle_fonction,
    texte,
    count=1,
)

if nombre_fonction != 1:
    raise SystemExit(
        "La fonction analyser_direction n'a pas pu etre remplacee."
    )

# 3. Remplacer le bloc d'ecriture JSON par une version nettoyee.
motif_json = (
    r'    with \(DOSSIER / "resume\.json"\)\.open\('
    r"[\s\S]*?allow_nan=False,\s*\n\s*\)"
)

nouveau_bloc = '''    rapport = {
        "statut": "Sensibilite de la propagation orientee",
        "w_EE": float(W_EE),
        "dt": float(DT),
        "fenetre": [
            float(T_DEBUT_PERTURBATION),
            float(T_FIN),
        ],
        "noeud_source": int(NOEUD_SOURCE),
        "sigma_spatial": float(SIGMA_SPATIAL),
        "duree_impulsion": float(DUREE_IMPULSION),
        "duree_soutien": float(DUREE_SOUTIEN),
        "champ_directions": "coherent selon l'axe x",
        "resultats": resultats,
    }

    rapport = nettoyer_json(rapport)

    with (DOSSIER / "resume.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(rapport, f, indent=2, allow_nan=False)'''

texte, nombre_json = re.subn(
    motif_json,
    nouveau_bloc,
    texte,
    count=1,
)

if nombre_json != 1:
    raise SystemExit(
        "Le bloc d'ecriture JSON n'a pas pu etre remplace."
    )

fichier.write_text(texte, encoding="utf-8")
print("Correctif applique : NaN convertis en null dans le JSON.")
PYTHON
