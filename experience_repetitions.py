#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Réplications de l'expérience de diffusion sur graphe.

Pour chaque graine :
- création d'un graphe spatial ;
- création d'un état initial commun ;
- simulation du témoin intact ;
- simulation avec différentes fractions d'arêtes conservées ;
- mesure des différences au témoin ;
- mesure de la connectivité.

Les sous-graphes d'une même réplication sont emboîtés :
ils utilisent une même permutation des arêtes.

Les intervalles bootstrap portent sur les réplications numériques.
Ils ne représentent aucune variabilité biologique.

Dépendances :
    numpy
    simulation_graphe.py
    trajectoires.py

Exécution :
    python experience_repetitions.py

Options :
    python experience_repetitions.py --repetitions 100
    python experience_repetitions.py --repetitions 3 --sortie resultats/essai
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import simulation_graphe as modele
from trajectoires import simuler


FRACTIONS = (0.25, 0.50, 0.75, 1.00)
N_BOOTSTRAP = 2000

COLONNES = [
    "seed",
    "fraction_demandee",
    "fraction_effective",
    "n_aretes",
    "n_composantes",
    "taille_plus_grande_composante",
    "n_noeuds_isoles",
    "dt",
    "n_steps",
    "variance_F_initiale",
    "variance_I_initiale",
    "variance_F_finale",
    "variance_I_finale",
    "difference_variance_F",
    "difference_variance_I",
    "rms_F_au_temoin",
    "rms_I_au_temoin",
    "fraction_B_differents",
    "derive_somme_F_finale",
    "derive_somme_I_finale",
]

METRIQUES_RESUME = [
    "n_composantes",
    "taille_plus_grande_composante",
    "n_noeuds_isoles",
    "variance_F_finale",
    "variance_I_finale",
    "difference_variance_F",
    "difference_variance_I",
    "rms_F_au_temoin",
    "rms_I_au_temoin",
    "fraction_B_differents",
]


def analyser_connectivite(A):
    """Calcule les composantes d'un graphe non orienté."""
    voisins = np.asarray(A) > 0
    n = len(voisins)
    visites = np.zeros(n, dtype=bool)
    tailles = []

    for depart in range(n):
        if visites[depart]:
            continue

        visites[depart] = True
        pile = [depart]
        taille = 0

        while pile:
            noeud = pile.pop()
            taille += 1

            nouveaux = np.flatnonzero(
                voisins[noeud] & ~visites
            )

            visites[nouveaux] = True
            pile.extend(nouveaux.tolist())

        tailles.append(taille)

    return {
        "n_composantes": len(tailles),
        "taille_plus_grande_composante": max(tailles),
        "n_noeuds_isoles": int(np.sum(voisins.sum(axis=1) == 0)),
    }


def construire_sous_graphe(A, aretes_ordonnees, fraction):
    """Conserve les premières arêtes d'une permutation commune."""
    nombre = int(np.floor(len(aretes_ordonnees) * fraction))
    selection = aretes_ordonnees[:nombre]

    resultat = np.zeros_like(A)

    resultat[selection[:, 0], selection[:, 1]] = 1.0
    resultat[selection[:, 1], selection[:, 0]] = 1.0

    return resultat


def etat_final(initial, A, dt, n_steps):
    ""
