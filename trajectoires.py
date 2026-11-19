#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enregistrement générique de trajectoires numériques.

Dépendance :
    numpy

Aucun autre fichier Python du projet n'est nécessaire.

Ce module :
- reçoit un état initial et une fonction d'évolution ;
- enregistre les états à intervalles réguliers ;
- conserve toujours l'état initial et l'état final ;
- vérifie les noms, les dimensions et les valeurs finies ;
- sauvegarde les résultats dans une archive NumPy compressée.

Les indices enregistrés sont des numéros d'itération.
Aucune durée physique ou biologique n'est supposée.

Exécution directe :
    python trajectoires.py

L'exécution directe lance uniquement un test technique utilisant
un état constant. Ce test ne définit pas un modèle scientifique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Tuple

import numpy as np


Etat = Mapping[str, np.ndarray]
Evolution = Callable[[Etat, int], Etat]


def verifier_entier(
    valeur: int,
    nom: str,
    minimum: int,
) -> int:
    """Valide un entier sans convertir silencieusement un flottant."""
    if isinstance(valeur, (bool, np.bool_)):
        raise TypeError(f"{nom} doit être un entier, pas un booléen.")

    if not isinstance(valeur, (int, np.integer)):
        raise TypeError(f"{nom} doit être un entier.")

    resultat = int(valeur)

    if resultat < minimum:
        raise ValueError(
            f"{nom} doit être supérieur ou égal à {minimum}."
        )

    return resultat


def copier_etat(
    etat: Etat,
) -> Dict[str, np.ndarray]:
    """Valide et copie tous les champs d'un état."""
    if not isinstance(etat, Mapping):
        raise TypeError(
            "L'état doit être un dictionnaire ou un mapping."
        )

    if len(etat) == 0:
        raise ValueError(
            "L'état doit contenir au moins un champ."
        )

    resultat: Dict[str, np.ndarray] = {}

    for nom, valeurs in etat.items():
        if not isinstance(nom, str) or not nom.strip():
            raise ValueError(
                "Chaque champ doit avoir un nom non vide."
            )

        tableau = np.asarray(valeurs, dtype=float)

        if tableau.size == 0:
            raise ValueError(
                f"Le champ {nom!r} est vide."
            )

        if not np.all(np.isfinite(tableau)):
            raise ValueError(
                f"Le champ {nom!r} contient des valeurs non finies."
            )

        resultat[nom] = tableau.copy()

    return resultat


def simuler(
    etat_initial: Etat,
    evolution: Evolution,
    n_steps: int,
    enregistrer_tous: int = 1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Exécute une évolution et enregistre sa trajectoire.

    Parameters
    ----------
    etat_initial :
        Dictionnaire de champs numériques.
        Exemple : {"F": F, "I": I, "B": B}.

    evolution :
        Fonction appelée sous la forme :

            nouvel_etat = evolution(etat, transition)

        Elle reçoit une copie de l'état courant et l'indice
        de la transition, compris entre 0 et n_steps - 1.

        Elle doit retourner les mêmes noms de champs,
        avec les mêmes formes.

    n_steps :
        Nombre de transitions à effectuer.

    enregistrer_tous :
        Nombre de transitions entre deux enregistrements.
        L'état initial et l'état final sont toujours conservés.

    Returns
    -------
    iterations :
        Tableau des numéros d'itération enregistrés.

    champs :
        Dictionnaire de tableaux.
        Le premier axe de chaque tableau correspond aux
        itérations enregistrées.

    Notes
    -----
    Ce module ne définit pas les équations d'évolution.
    Les variables peuvent être continues ou discrètes,
    mais doivent rester numériques et finies.
    """
    n_steps = verifier_entier(
        n_steps,
        "n_steps",
        minimum=0,
    )

    enregistrer_tous = verifier_entier(
        enregistrer_tous,
        "enregistrer_tous",
        minimum=1,
    )

    if not callable(evolution):
        raise TypeError(
            "evolution doit être une fonction appelable."
        )

    etat = copier_etat(etat_initial)

    formes = {
        nom: valeurs.shape
        for nom, valeurs in etat.items()
    }

    iterations = [0]

    historique = {
        nom: [valeurs.copy()]
        for nom, valeurs in etat.items()
    }

    for transition in range(n_steps):
        prochain_etat = evolution(
            copier_etat(etat),
            transition,
        )

        try:
            etat = copier_etat(prochain_etat)
        except (TypeError, ValueError) as erreur:
            raise ValueError(
                "État invalide après la transition "
                f"{transition} : {erreur}"
            ) from erreur

        if set(etat) != set(formes):
            raise ValueError(
                "Les noms des champs ont changé après "
                f"la transition {transition}."
            )

        for nom, forme_attendue in formes.items():
            if etat[nom].shape != forme_attendue:
                raise ValueError(
                    f"La forme du champ {nom!r} a changé après "
                    f"la transition {transition} : "
                    f"{etat[nom].shape} au lieu de {forme_attendue}."
                )

        iteration = transition + 1

        if (
            iteration % enregistrer_tous == 0
            or iteration == n_steps
        ):
            iterations.append(iteration)

            for nom, valeurs in etat.items():
                historique[nom].append(valeurs.copy())

    return (
        np.asarray(iterations, dtype=np.int64),
        {
            nom: np.stack(valeurs, axis=0)
            for nom, valeurs in historique.items()
        },
    )


def sauvegarder(
    chemin: str | Path,
    iterations: np.ndarray,
    champs: Mapping[str, np.ndarray],
) -> Path:
    """
    Sauvegarde une trajectoire dans une archive .npz.

    L'archive contient :
    - iterations ;
    - champ_F, champ_I, etc., selon les noms des champs.

    Un fichier déjà présent au même emplacement est remplacé.
    """
    indices = np.asarray(iterations)

    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            "iterations doit être un tableau 1D non vide."
        )

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(
            "iterations doit contenir des entiers."
        )

    if np.any(indices < 0):
        raise ValueError(
            "Les indices d'itération doivent être non négatifs."
        )

    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError(
            "Les indices d'itération doivent être "
            "strictement croissants."
        )

    champs_valides = copier_etat(champs)

    for nom, valeurs in champs_valides.items():
        if valeurs.ndim == 0:
            raise ValueError(
                f"Le champ {nom!r} doit avoir un axe temporel."
            )

        if valeurs.shape[0] != len(indices):
            raise ValueError(
                f"Le champ {nom!r} contient "
                f"{valeurs.shape[0]} états, mais "
                f"{len(indices)} itérations sont fournies."
            )

    destination = Path(chemin)

    if destination.suffix != ".npz":
        destination = Path(f"{destination}.npz")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    donnees = {
        "iterations": indices,
        **{
            f"champ_{nom}": valeurs
            for nom, valeurs in champs_valides.items()
        },
    }

    np.savez_compressed(
        destination,
        **donnees,
    )

    return destination


def test_enregistrement() -> None:
    """
    Vérifie l'enregistrement et la sauvegarde d'un état constant.

    Il s'agit d'un test technique, pas d'une simulation
    des mécanismes F-I-B.
    """
    etat_initial = {
        "F": np.array([0.1, 0.2, 0.3]),
        "I": np.array([-0.1, 0.0, 0.1]),
        "B": np.array([0.0, 1.0, 1.0]),
    }

    def evolution_identite(
        etat: Etat,
        transition: int,
    ) -> Etat:
        return etat

    iterations, champs = simuler(
        etat_initial=etat_initial,
        evolution=evolution_identite,
        n_steps=10,
        enregistrer_tous=3,
    )

    np.testing.assert_array_equal(
        iterations,
        np.array([0, 3, 6, 9, 10]),
    )

    for nom, initial in etat_initial.items():
        attendu = np.repeat(
            initial[np.newaxis, :],
            repeats=len(iterations),
            axis=0,
        )

        np.testing.assert_array_equal(
            champs[nom],
            attendu,
        )

    # Vérifier également le cas sans transition.
    iterations_zero, champs_zero = simuler(
        etat_initial=etat_initial,
        evolution=evolution_identite,
        n_steps=0,
    )

    np.testing.assert_array_equal(
        iterations_zero,
        np.array([0]),
    )

    for nom, initial in etat_initial.items():
        np.testing.assert_array_equal(
            champs_zero[nom],
            initial[np.newaxis, :],
        )

    destination = sauvegarder(
        "resultats/test_enregistrement.npz",
        iterations,
        champs,
    )

    with np.load(
        destination,
        allow_pickle=False,
    ) as archive:
        np.testing.assert_array_equal(
            archive["iterations"],
            iterations,
        )

        for nom, valeurs in champs.items():
            np.testing.assert_array_equal(
                archive[f"champ_{nom}"],
                valeurs,
            )

    print("Test d'enregistrement réussi.")
    print("Test sans transition réussi.")
    print("Test de sauvegarde et de relecture réussi.")
    print(f"Archive créée : {destination}")
    print(
        "Ce test utilise un état constant : "
        "aucune dynamique scientifique n'est encore définie."
    )


if __name__ == "__main__":
    test_enregistrement()
