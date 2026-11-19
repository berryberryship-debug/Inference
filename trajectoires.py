#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enregistrement des trajectoires F-I-B sur un graphe fixe.

Ce module :
- utilise les équations de dynamique_double_courant.step ;
- conserve les états à intervalles réguliers ;
- enregistre toujours l'état initial et l'état final ;
- vérifie les dimensions et les valeurs finies ;
- exporte les trajectoires dans un fichier NumPy compressé.

Les indices temporels sont des numéros d'itération.
Aucune unité de temps physique ou biologique n'est supposée.

Dépendances :
    numpy
    dynamique_double_courant.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


if __package__:
    from .dynamique_double_courant import laplacien_graphe, step
else:
    from dynamique_double_courant import laplacien_graphe, step


@dataclass(frozen=True)
class Trajectoire:
    """États enregistrés, de forme (nombre_enregistrements, nombre_noeuds)."""

    iterations: np.ndarray
    F: np.ndarray
    I: np.ndarray
    B: np.ndarray

    def sauvegarder(self, chemin: str | Path) -> Path:
        """Exporte les tableaux dans une archive .npz."""
        destination = Path(chemin)

        if destination.suffix != ".npz":
            destination = Path(f"{destination}.npz")

        destination.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            destination,
            iterations=self.iterations,
            F=self.F,
            I=self.I,
            B=self.B,
        )

        return destination


def _entier_valide(
    valeur: int,
    nom: str,
    minimum: int,
) -> int:
    """Refuse les booléens, les flottants et les entiers hors limites."""
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(
        valeur, (int, np.integer)
    ):
        raise TypeError(f"{nom} doit être un entier.")

    resultat = int(valeur)

    if resultat < minimum:
        raise ValueError(f"{nom} doit être supérieur ou égal à {minimum}.")

    return resultat


def _champ_valide(
    valeurs: np.ndarray,
    nom: str,
    n_noeuds: int,
) -> np.ndarray:
    """Valide et copie un champ sans modifier le tableau fourni."""
    champ = np.asarray(valeurs, dtype=float)

    if champ.shape != (n_noeuds,):
        raise ValueError(
            f"{nom} doit avoir la forme ({n_noeuds},), "
            f"forme reçue : {champ.shape}."
        )

    if not np.all(np.isfinite(champ)):
        raise ValueError(f"{nom} contient des valeurs non finies.")

    return champ.copy()


def simuler_trajectoire(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    adjacency: np.ndarray,
    n_steps: int,
    parametres_step: Mapping[str, Any],
    enregistrer_tous: int = 1,
) -> Trajectoire:
    """
    Fait évoluer les champs et conserve leur trajectoire.

    Parameters
    ----------
    F, I, B :
        Champs initiaux, chacun de forme (N,).
    adjacency :
        Matrice d'adjacence de forme (N, N).
        Aucun changement de graphe n'est effectué pendant cette simulation.
    n_steps :
        Nombre de mises à jour.
    parametres_step :
        Arguments nommés transmis sans modification à step.
    enregistrer_tous :
        Intervalle entre deux enregistrements.
        L'état initial et l'état final sont toujours conservés.

    Returns
    -------
    Trajectoire
        Copies des états enregistrés et indices d'itération associés.

    Notes
    -----
    B est vérifié comme champ numérique fini, sans imposer qu'il reste
    binaire : cette propriété dépend des équations de step.
    """
    n_steps = _entier_valide(n_steps, "n_steps", minimum=0)
    enregistrer_tous = _entier_valide(
        enregistrer_tous,
        "enregistrer_tous",
        minimum=1,
    )

    A = np.asarray(adjacency, dtype=float).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency doit être une matrice carrée.")

    n_noeuds = A.shape[0]

    if n_noeuds == 0:
        raise ValueError("Le graphe doit contenir au moins un nœud.")

    if not np.all(np.isfinite(A)):
        raise ValueError("adjacency contient des valeurs non finies.")

    f = _champ_valide(F, "F", n_noeuds)
    i = _champ_valide(I, "I", n_noeuds)
    b = _champ_valide(B, "B", n_noeuds)

    L = np.asarray(laplacien_graphe(A), dtype=float)

    if L.shape != A.shape:
        raise ValueError("Le laplacien n'a pas la forme attendue.")

    if not np.all(np.isfinite(L)):
        raise ValueError("Le laplacien contient des valeurs non finies.")

    parametres = dict(parametres_step)

    iterations = [0]
    historique_f = [f.copy()]
    historique_i = [i.copy()]
    historique_b = [b.copy()]

    for iteration in range(1, n_steps + 1):
        f_suivant, i_suivant, b_suivant = step(
            f,
            i,
            b,
            L,
            **parametres,
        )

        try:
            f = _champ_valide(f_suivant, "F", n_noeuds)
            i = _champ_valide(i_suivant, "I", n_noeuds)
            b = _champ_valide(b_suivant, "B", n_noeuds)
        except ValueError as erreur:
            raise ValueError(
                f"État invalide après l'itération {iteration} : {erreur}"
            ) from erreur

        if iteration % enregistrer_tous == 0 or iteration == n_steps:
            iterations.append(iteration)
            historique_f.append(f.copy())
            historique_i.append(i.copy())
            historique_b.append(b.copy())

    return Trajectoire(
        iterations=np.asarray(iterations, dtype=np.int64),
        F=np.stack(historique_f),
        I=np.stack(historique_i),
        B=np.stack(historique_b),
    )
