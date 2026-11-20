cd ~

python - <<'PYTHON'
from pathlib import Path

fichier = Path("stabilite_neuronale.py")
texte = fichier.read_text(encoding="utf-8")

ancien = '''    derivee_analytique = jacobien_homogene(w_EE) @ np.array([1.0, 1.0])
    derivee_analytique = np.concatenate(
        [derivee_analytique[:0] + derivee_analytique[0] * np.ones(N),
         derivee_analytique[1] * np.ones(N)]
    )
'''

nouveau = '''    derivee_analytique = jacobien_homogene(w_EE) @ np.array([1.0, 1.0])
    derivee_analytique = np.concatenate([
        np.full(N, derivee_analytique[0]),
        np.full(N, derivee_analytique[1]),
    ])
'''

if ancien not in texte:
    raise SystemExit("Bloc a remplacer introuvable : ne rien modifier.")

fichier.write_text(texte.replace(ancien, nouveau), encoding="utf-8")
print("Correctif applique.")
PYTHON
