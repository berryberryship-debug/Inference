cd ~

python - <<'PYTHON'
from pathlib import Path

fichier = Path("stabilite_neuronale.py")
texte = fichier.read_text(encoding="utf-8")

debut = texte.index("def verifier_jacobien(")
fin = texte.index("def main():")

nouvelle_fonction = '''def verifier_jacobien(adjacency, w_EE, epsilon=1e-6):
    """Verifie la derivee directionnelle du mode homogene."""

    P, Q = entrees(w_EE)

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
        return np.concatenate([dE, dI])

    E_homogene = 0.5 * np.ones(N)
    I_homogene = 0.5 * np.ones(N)
    direction = np.concatenate([np.ones(N), np.ones(N)])

    etat_plus = np.concatenate(
        [E_homogene + epsilon * np.ones(N), I_homogene + epsilon * np.ones(N)]
    )
    etat_moins = np.concatenate(
        [E_homogene - epsilon * np.ones(N), I_homogene - epsilon * np.ones(N)]
    )

    derivee_numerique = (
        dynamique(etat_plus[:N], etat_plus[N:])
        - dynamique(etat_moins[:N], etat_moins[N:])
    ) / (2.0 * epsilon)

    derivee_analytique = jacobien_homogene(w_EE) @ np.array([1.0, 1.0])
    derivee_analytique = np.concatenate(
        [derivee_analytique[:0] + derivee_analytique[0] * np.ones(N),
         derivee_analytique[1] * np.ones(N)]
    )

    return float(np.max(np.abs(derivee_numerique - derivee_analytique)))


'''

texte = texte[:debut] + nouvelle_fonction + texte[fin:]
fichier.write_text(texte, encoding="utf-8")
print("Fonction verifier_jacobien remplacee.")
PYTHON
