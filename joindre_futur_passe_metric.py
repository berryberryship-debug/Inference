#!/usr/bin/env python3
"""
Analyse geometrique des futurs simules (version robuste).

Fichier :
    joindre_futur_passe_metric.py

Dependance locale :
    prototype_futur_passe.py   (meme dossier, ou --module)

Dependances externes :
    numpy
    matplotlib

Utilisation :
    python joindre_futur_passe_metric.py
    python joindre_futur_passe_metric.py --rapide
    python joindre_futur_passe_metric.py --autotest
    python joindre_futur_passe_metric.py --help

Le programme contient :
    - une configuration validee (dataclass) modifiable en ligne de commande ;
    - une journalisation console et fichier ;
    - une interface verifiee avec le prototype (noms, formes, finitude) ;
    - une metrique MDS avec projection lineaire regularisee, diagnostics
      de qualite et sauvegarde / chargement ;
    - une representation multimodale present/futur ;
    - des ensembles de branches futures ;
    - un controle de contexte constant avec bruits apparies ;
    - des permutations des scenes sans regeneration des branches ;
    - des volumes de covariance et leurs logarithmes ;
    - un ancrage present/futur et une distance present/futur ;
    - des contrastes apparies ;
    - un bootstrap temporel par blocs ;
    - des distributions de permutation pour les volumes, avec corrections
      pour comparaisons multiples (Holm, Benjamini-Hochberg) ;
    - des controles de coherence (determinisme, equivalence des contextes,
      non-mutation de l'etat present) ;
    - un rapport JSON ecrit de facon atomique, un tableau CSV, des donnees
      NPZ, plusieurs graphiques et un manifeste avec empreintes SHA-256 ;
    - des autotests numeriques (--autotest).

Limites :
    - modele synthetique non calibre ;
    - dissimilarites euclidiennes, pas de jugements humains ;
    - contexte et activites globales figes dans les branches futures ;
    - les branches sont des propositions latentes, pas des trajectoires
      biophysiques completes a un horizon temporel calibre ;
    - aucune interpretation clinique, de conscience ou de retrocausalite.
"""

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from math import lgamma, log, pi
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import logging
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------

VERSION_SCRIPT = "2.0-robuste"

try:
    DOSSIER_SCRIPT = Path(__file__).resolve().parent
except NameError:
    # Cellule interactive : __file__ n'existe pas.
    DOSSIER_SCRIPT = Path.cwd()

DOSSIER_SORTIE_DEFAUT = (
    DOSSIER_SCRIPT
    / "resultats"
    / "jonction_futur_passe_metric"
)

CHAMPS_MESURES = (
    "ancrage",
    "distance",
    "volume_futur",
    "log_volume_futur",
    "volume_attache",
    "log_volume_attache",
    "volume_joint",
    "log_volume_joint",
    "dispersion",
    "rang_covariance",
)

CHAMPS_SIMULATION = (
    "temps",
    "E",
    "I",
    "C",
    "H",
    "scene",
    "M_passe",
)

PARAMETRES_RESEAU = (
    "W_MEM_G",
    "W_MEM_C",
    "W_MEM_H",
    "W_HH",
    "W_HC",
    "W_HG",
    "decodeur_scene",
)

NOMBRE_COORDONNEES_SCENE = 4
DIMENSION_ETAT_GLOBAL = 2

journal = logging.getLogger("futur_passe")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Configuration:
    """
    Tous les parametres modifiables de l'analyse.

    Les valeurs par defaut reproduisent le script d'origine.
    """

    dimension_latente: int = 6
    nombre_branches: int = 64
    nombre_instants: int = 40
    nombre_repetitions: int = 100

    max_etats_entrainement: int = 120

    regularisation_metrique: float = 1e-2
    regularisation_covariance: float = 1e-8

    longueurs_blocs: tuple = (4, 8, 12)
    nombre_bootstrap: int = 2000

    graine_ensemble: int = 9000
    graine_permutations: int = 20000
    graine_bootstrap: int = 1234

    fraction_entrainement: float = 0.5

    module_prototype: str = "prototype_futur_passe"
    dossier_sortie: str = ""

    niveau_journal: str = "INFO"

    cible_standardisee: bool = False
    verifier_determinisme: bool = True
    graphiques_supplementaires: bool = True
    ecrire_branches_npz: bool = True

    seuil_alpha: float = 0.05

    def valider(self):
        erreurs = []

        entiers_positifs = (
            "dimension_latente",
            "nombre_branches",
            "nombre_instants",
            "nombre_repetitions",
            "max_etats_entrainement",
            "nombre_bootstrap",
        )

        for nom in entiers_positifs:
            valeur = getattr(self, nom)

            if not isinstance(valeur, (int, np.integer)) or valeur < 1:
                erreurs.append(
                    f"{nom} doit etre un entier >= 1 (recu {valeur!r})."
                )

        if not erreurs:
            if self.nombre_branches <= self.dimension_latente:
                erreurs.append(
                    "nombre_branches doit depasser dimension_latente."
                )

            if self.max_etats_entrainement <= self.dimension_latente:
                erreurs.append(
                    "max_etats_entrainement doit depasser "
                    "dimension_latente."
                )

        for nom in (
            "regularisation_metrique",
            "regularisation_covariance",
        ):
            valeur = getattr(self, nom)

            if not np.isfinite(valeur) or valeur <= 0.0:
                erreurs.append(
                    f"{nom} doit etre fini et strictement positif."
                )

        if not 0.0 < self.fraction_entrainement < 1.0:
            erreurs.append(
                "fraction_entrainement doit etre dans ]0, 1[."
            )

        if not 0.0 < self.seuil_alpha < 1.0:
            erreurs.append("seuil_alpha doit etre dans ]0, 1[.")

        if len(self.longueurs_blocs) == 0:
            erreurs.append("longueurs_blocs ne doit pas etre vide.")

        for longueur in self.longueurs_blocs:
            if not isinstance(longueur, (int, np.integer)) or longueur < 1:
                erreurs.append(
                    f"Longueur de bloc invalide : {longueur!r}."
                )

        if self.niveau_journal.upper() not in (
            "DEBUG", "INFO", "WARNING", "ERROR",
        ):
            erreurs.append(
                "niveau_journal doit etre DEBUG, INFO, WARNING ou ERROR."
            )

        if erreurs:
            raise ValueError(
                "Configuration invalide :\n  - "
                + "\n  - ".join(erreurs)
            )

        return self

    def dossier(self):
        if self.dossier_sortie:
            return Path(self.dossier_sortie).expanduser().resolve()

        return DOSSIER_SORTIE_DEFAUT

    def en_dictionnaire(self):
        donnees = asdict(self)
        donnees["longueurs_blocs"] = list(self.longueurs_blocs)
        return donnees


def analyser_arguments(argv=None):
    parseur = argparse.ArgumentParser(
        description=(
            "Analyse geometrique des futurs simules "
            "(modele synthetique)."
        ),
    )

    defaut = Configuration()

    parseur.add_argument(
        "--dimension", type=int, default=defaut.dimension_latente,
        help="dimension latente du MDS",
    )
    parseur.add_argument(
        "--branches", type=int, default=defaut.nombre_branches,
        help="nombre de branches futures par instant",
    )
    parseur.add_argument(
        "--instants", type=int, default=defaut.nombre_instants,
        help="nombre d'instants analyses",
    )
    parseur.add_argument(
        "--repetitions", type=int, default=defaut.nombre_repetitions,
        help="repetitions (bruits, permutations)",
    )
    parseur.add_argument(
        "--bootstrap", type=int, default=defaut.nombre_bootstrap,
        help="repetitions du bootstrap par blocs",
    )
    parseur.add_argument(
        "--graine-ensemble", type=int, default=defaut.graine_ensemble,
        help="graine des bruits des branches",
    )
    parseur.add_argument(
        "--graine-permutations", type=int,
        default=defaut.graine_permutations,
        help="graine des permutations de scenes",
    )
    parseur.add_argument(
        "--graine-bootstrap", type=int, default=defaut.graine_bootstrap,
        help="graine du bootstrap",
    )
    parseur.add_argument(
        "--fraction-entrainement", type=float,
        default=defaut.fraction_entrainement,
        help="part de la duree utilisee pour apprendre la metrique",
    )
    parseur.add_argument(
        "--module", default=defaut.module_prototype,
        help="nom du module prototype a importer",
    )
    parseur.add_argument(
        "--sortie", default="",
        help="dossier de sortie (defaut : resultats/... a cote du script)",
    )
    parseur.add_argument(
        "--niveau", default=defaut.niveau_journal,
        help="niveau du journal : DEBUG, INFO, WARNING, ERROR",
    )
    parseur.add_argument(
        "--cible-standardisee", action="store_true",
        help="calcule les distances cibles du MDS sur X standardise",
    )
    parseur.add_argument(
        "--sans-determinisme", action="store_true",
        help="saute le controle de determinisme",
    )
    parseur.add_argument(
        "--sans-graphiques-supplementaires", action="store_true",
        help="ne trace que le graphique principal",
    )
    parseur.add_argument(
        "--sans-branches-npz", action="store_true",
        help="n'enregistre pas les branches latentes dans le NPZ",
    )
    parseur.add_argument(
        "--rapide", action="store_true",
        help="mode d'essai : 8 instants, 10 repetitions, 200 bootstraps",
    )
    parseur.add_argument(
        "--autotest", action="store_true",
        help="execute les autotests numeriques puis quitte",
    )

    # parse_known_args : tolere les arguments injectes par un notebook.
    arguments, inconnus = parseur.parse_known_args(argv)

    if inconnus:
        print(
            "Arguments ignores : " + " ".join(inconnus),
            file=sys.stderr,
        )

    configuration = Configuration(
        dimension_latente=arguments.dimension,
        nombre_branches=arguments.branches,
        nombre_instants=arguments.instants,
        nombre_repetitions=arguments.repetitions,
        nombre_bootstrap=arguments.bootstrap,
        graine_ensemble=arguments.graine_ensemble,
        graine_permutations=arguments.graine_permutations,
        graine_bootstrap=arguments.graine_bootstrap,
        fraction_entrainement=arguments.fraction_entrainement,
        module_prototype=arguments.module,
        dossier_sortie=arguments.sortie,
        niveau_journal=arguments.niveau,
        cible_standardisee=arguments.cible_standardisee,
        verifier_determinisme=not arguments.sans_determinisme,
        graphiques_supplementaires=(
            not arguments.sans_graphiques_supplementaires
        ),
        ecrire_branches_npz=not arguments.sans_branches_npz,
    )

    if arguments.rapide:
        configuration.nombre_instants = min(
            configuration.nombre_instants, 8
        )
        configuration.nombre_repetitions = min(
            configuration.nombre_repetitions, 10
        )
        configuration.nombre_bootstrap = min(
            configuration.nombre_bootstrap, 200
        )

    return configuration, arguments


# ---------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------

def configurer_journal(niveau, fichier=None):
    journal.setLevel(logging.DEBUG)
    journal.propagate = False

    for gestionnaire in list(journal.handlers):
        journal.removeHandler(gestionnaire)
        gestionnaire.close()

    format_message = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, niveau.upper()))
    console.setFormatter(format_message)
    journal.addHandler(console)

    if fichier is not None:
        sortie_fichier = logging.FileHandler(
            fichier,
            mode="w",
            encoding="utf-8",
        )
        sortie_fichier.setLevel(logging.DEBUG)
        sortie_fichier.setFormatter(format_message)
        journal.addHandler(sortie_fichier)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def verifier_fini(tableau, nom):
    tableau = np.asarray(tableau, dtype=float)

    if not np.all(np.isfinite(tableau)):
        nombre = int(np.sum(~np.isfinite(tableau)))

        raise FloatingPointError(
            f"{nom} contient {nombre} valeur(s) non finie(s)."
        )

    return tableau


def verifier_forme(tableau, forme, nom):
    tableau = np.asarray(tableau)

    if tableau.shape != tuple(forme):
        raise ValueError(
            f"{nom} a la forme {tableau.shape}, "
            f"attendu {tuple(forme)}."
        )

    return tableau


def verifier_matrice(X, nom="X"):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"{nom} doit etre une matrice (ndim={X.ndim})."
        )

    verifier_fini(X, nom)

    return X


def empreinte_fichier(chemin, taille_bloc=1 << 20):
    sha = hashlib.sha256()

    with Path(chemin).open("rb") as fichier:
        while True:
            bloc = fichier.read(taille_bloc)

            if not bloc:
                break

            sha.update(bloc)

    return sha.hexdigest()


# ---------------------------------------------------------------------
# Fonctions numeriques
# ---------------------------------------------------------------------

def sigmoid(x):
    x = np.asarray(x, dtype=float)

    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -40.0, 40.0))
    )


def cosinus(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominateur = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominateur <= 1e-15:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) / denominateur,
            -1.0,
            1.0,
        )
    )


def log_ball_volume_factor(dimension):
    dimension = int(dimension)

    if dimension < 1:
        raise ValueError(
            "La dimension doit etre positive."
        )

    return (
        0.5 * dimension * log(pi)
        - lgamma(0.5 * dimension + 1.0)
    )


def ball_volume_factor(dimension):
    return float(
        np.exp(log_ball_volume_factor(dimension))
    )


def exponentielle_sure(valeur):
    valeur = float(valeur)

    if np.isneginf(valeur):
        return 0.0

    limite = np.log(np.finfo(float).max)

    if not np.isfinite(valeur) or valeur > limite:
        raise FloatingPointError(
            "Exponentielle non representable numeriquement."
        )

    return float(np.exp(valeur))


def distances_euclidiennes(X):
    X = verifier_matrice(X, "X")

    normes2 = np.sum(X * X, axis=1)

    D2 = (
        normes2[:, None]
        + normes2[None, :]
        - 2.0 * (X @ X.T)
    )

    np.maximum(D2, 0.0, out=D2)

    D = np.sqrt(D2)
    D = 0.5 * (D + D.T)

    np.fill_diagonal(D, 0.0)

    return D


def rang_effectif(valeurs_propres):
    """
    Rang effectif par rapport de participation :
    (somme lambda)^2 / somme lambda^2.
    """
    valeurs = np.maximum(
        np.asarray(valeurs_propres, dtype=float),
        0.0,
    )

    somme2 = float(np.sum(valeurs ** 2))

    if somme2 <= 0.0:
        return 0.0

    return float(np.sum(valeurs) ** 2 / somme2)


def correction_holm(p_valeurs):
    """
    Correction de Holm (controle du FWER), valeurs ajustees.
    Les NaN restent NaN et ne comptent pas dans le nombre de tests.
    """
    p = np.asarray(p_valeurs, dtype=float)
    ajuste = np.full(p.shape, np.nan)

    valide = np.isfinite(p)
    m = int(valide.sum())

    if m == 0:
        return ajuste

    indices = np.flatnonzero(valide)
    ordre = indices[np.argsort(p[indices], kind="stable")]

    maximum_courant = 0.0

    for rang, position in enumerate(ordre):
        valeur = (m - rang) * p[position]
        maximum_courant = max(maximum_courant, valeur)
        ajuste[position] = min(1.0, maximum_courant)

    return ajuste


def correction_benjamini_hochberg(p_valeurs):
    """
    Correction de Benjamini-Hochberg (controle du FDR),
    valeurs ajustees. Hypothese de dependance non verifiee ici.
    """
    p = np.asarray(p_valeurs, dtype=float)
    ajuste = np.full(p.shape, np.nan)

    valide = np.isfinite(p)
    m = int(valide.sum())

    if m == 0:
        return ajuste

    indices = np.flatnonzero(valide)
    ordre = indices[np.argsort(p[indices], kind="stable")]

    minimum_courant = 1.0

    for rang_inverse, position in enumerate(ordre[::-1]):
        rang = m - rang_inverse
        valeur = p[position] * m / rang
        minimum_courant = min(minimum_courant, valeur)
        ajuste[position] = min(1.0, minimum_courant)

    return ajuste


# ---------------------------------------------------------------------
# Metrique integree
# ---------------------------------------------------------------------

class SubjectiveMetric:
    """
    MDS classique puis projection lineaire regularisee.

    Le nom est conserve pour compatibilite avec le projet.
    Une cible euclidienne ne constitue pas une mesure subjective
    empirique.
    """

    def __init__(
        self,
        k=6,
        reg=1e-2,
        standardize=True,
    ):
        self.k = int(k)
        self.reg = float(reg)
        self.standardize = bool(standardize)

        if self.k < 1:
            raise ValueError(
                "k doit etre positif."
            )

        if (
            not np.isfinite(self.reg)
            or self.reg <= 0.0
        ):
            raise ValueError(
                "reg doit etre fini et positif."
            )

        self.moyenne_ = None
        self.ecart_type_ = None
        self.projection_ = None
        self.valeurs_propres_ = None
        self.valeurs_propres_toutes_ = None
        self.erreur_projection_ = None
        self.conditionnement_gramme_ = None
        self.nombre_axes_positifs_ = None

    # -- standardisation ------------------------------------------------

    def _standardiser(
        self,
        X,
        apprentissage=False,
    ):
        X = verifier_matrice(X, "X")

        if apprentissage:
            self.moyenne_ = X.mean(axis=0)

            if self.standardize:
                ecarts = X.std(axis=0)

                self.ecart_type_ = np.where(
                    ecarts > 1e-12,
                    ecarts,
                    1.0,
                )
            else:
                self.ecart_type_ = np.ones(
                    X.shape[1],
                    dtype=float,
                )

        if self.moyenne_ is None:
            raise RuntimeError(
                "La metrique n'est pas ajustee."
            )

        if X.shape[1] != len(self.moyenne_):
            raise ValueError(
                "Dimension d'entree incompatible : "
                f"{X.shape[1]} au lieu de {len(self.moyenne_)}."
            )

        return (
            X - self.moyenne_
        ) / self.ecart_type_

    def standardiser(self, X):
        """Applique la standardisation apprise (sans projection)."""
        return self._standardiser(X)

    # -- ajustement -------------------------------------------------------

    def fit(
        self,
        X,
        D,
        select_k=False,
        verbose=False,
    ):
        X = verifier_matrice(X, "X")
        D = np.asarray(D, dtype=float)

        n = len(X)

        if D.shape != (n, n):
            raise ValueError(
                f"D doit avoir la forme ({n}, {n})."
            )

        verifier_fini(D, "D")

        if np.any(D < 0.0):
            raise ValueError(
                "D contient des distances negatives."
            )

        if not np.allclose(
            D,
            D.T,
            atol=1e-10,
        ):
            raise ValueError(
                "D doit etre symetrique."
            )

        if not np.allclose(
            np.diag(D),
            0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "La diagonale de D doit etre nulle."
            )

        if n <= self.k:
            raise ValueError(
                "Il faut plus de points que de dimensions."
            )

        Xs = self._standardiser(
            X,
            apprentissage=True,
        )

        D2 = D * D

        B = -0.5 * (
            D2
            - D2.mean(axis=0, keepdims=True)
            - D2.mean(axis=1, keepdims=True)
            + D2.mean()
        )

        B = 0.5 * (B + B.T)

        valeurs, vecteurs = np.linalg.eigh(B)

        ordre = np.argsort(valeurs)[::-1]

        valeurs = valeurs[ordre]
        vecteurs = vecteurs[:, ordre]

        tolerance = max(
            1e-12,
            abs(valeurs[0]) * 1e-10,
        )

        nombre_positif = int(
            np.sum(valeurs > tolerance)
        )

        self.nombre_axes_positifs_ = nombre_positif
        self.valeurs_propres_toutes_ = valeurs.copy()

        if nombre_positif == 0:
            raise ValueError(
                "Aucun axe positif dans les donnees "
                "d'entrainement."
            )

        if select_k:
            self.k = min(
                self.k,
                nombre_positif,
            )

        elif nombre_positif < self.k:
            journal.warning(
                "Seulement %d axe(s) MDS positif(s) pour k=%d : "
                "les derniers axes sont nuls.",
                nombre_positif,
                self.k,
            )

        self.valeurs_propres_ = (
            valeurs[:self.k].copy()
        )

        Z_cible = (
            vecteurs[:, :self.k]
            * np.sqrt(
                np.maximum(
                    valeurs[:self.k],
                    0.0,
                )
            )[None, :]
        )

        gramme = Xs.T @ Xs

        gramme += (
            self.reg
            * np.eye(Xs.shape[1])
        )

        self.conditionnement_gramme_ = float(
            np.linalg.cond(gramme)
        )

        if self.conditionnement_gramme_ > 1e12:
            journal.warning(
                "Matrice de Gram mal conditionnee (cond=%.3e). "
                "Augmenter la regularisation de la metrique.",
                self.conditionnement_gramme_,
            )

        self.projection_ = np.linalg.solve(
            gramme,
            Xs.T @ Z_cible,
        )

        self.erreur_projection_ = float(
            np.sqrt(
                np.mean(
                    (
                        Xs @ self.projection_
                        - Z_cible
                    ) ** 2
                )
            )
        )

        if verbose:
            journal.info(
                "Metrique : %d etats, %d variables, k=%d",
                n,
                X.shape[1],
                self.k,
            )

        return self

    # -- application ------------------------------------------------------

    def transform(self, X):
        if self.projection_ is None:
            raise RuntimeError(
                "Appeler fit avant transform."
            )

        return (
            self._standardiser(X)
            @ self.projection_
        )

    def fit_transform(
        self,
        X,
        D,
        select_k=False,
        verbose=False,
    ):
        self.fit(
            X,
            D,
            select_k=select_k,
            verbose=verbose,
        )

        return self.transform(X)

    # -- diagnostics ------------------------------------------------------

    def diagnostics(self, X, D):
        """
        Qualite de la reproduction des distances cibles D par les
        distances entre projections de X.

        Retourne :
            correlation  : Pearson entre D et D_projete (hors diagonale)
            stress_kruskal : sqrt(sum (D-Dp)^2 / sum D^2)
            erreur_relative_mediane : mediane |D-Dp| / D
        """
        X = verifier_matrice(X, "X")
        D = np.asarray(D, dtype=float)

        Z = self.transform(X)
        Dp = distances_euclidiennes(Z)

        masque = ~np.eye(len(D), dtype=bool)

        d = D[masque]
        dp = Dp[masque]

        if np.std(d) > 1e-15 and np.std(dp) > 1e-15:
            correlation = float(np.corrcoef(d, dp)[0, 1])
        else:
            correlation = float("nan")

        somme2 = float(np.sum(d ** 2))

        stress = (
            float(np.sqrt(np.sum((d - dp) ** 2) / somme2))
            if somme2 > 0.0
            else float("nan")
        )

        positifs = d > 1e-12

        erreur_relative = (
            float(np.median(np.abs(d[positifs] - dp[positifs]) / d[positifs]))
            if np.any(positifs)
            else float("nan")
        )

        valeurs = self.valeurs_propres_toutes_
        positives = np.maximum(valeurs, 0.0)
        total = float(positives.sum())

        part_expliquee = (
            float(positives[:self.k].sum() / total)
            if total > 0.0
            else float("nan")
        )

        return {
            "correlation_distances": correlation,
            "stress_kruskal": stress,
            "erreur_relative_mediane": erreur_relative,
            "part_variance_mds_expliquee": part_expliquee,
            "nombre_axes_positifs": self.nombre_axes_positifs_,
            "conditionnement_gramme": self.conditionnement_gramme_,
            "erreur_projection_rms": self.erreur_projection_,
        }

    # -- persistance ------------------------------------------------------

    def sauvegarder(self, chemin):
        if self.projection_ is None:
            raise RuntimeError("Rien a sauvegarder : metrique non ajustee.")

        np.savez_compressed(
            chemin,
            k=self.k,
            reg=self.reg,
            standardize=self.standardize,
            moyenne=self.moyenne_,
            ecart_type=self.ecart_type_,
            projection=self.projection_,
            valeurs_propres=self.valeurs_propres_,
        )

    @classmethod
    def charger(cls, chemin):
        with np.load(chemin) as donnees:
            objet = cls(
                k=int(donnees["k"]),
                reg=float(donnees["reg"]),
                standardize=bool(donnees["standardize"]),
            )

            objet.moyenne_ = donnees["moyenne"].copy()
            objet.ecart_type_ = donnees["ecart_type"].copy()
            objet.projection_ = donnees["projection"].copy()
            objet.valeurs_propres_ = donnees["valeurs_propres"].copy()

        return objet


# ---------------------------------------------------------------------
# Interface avec le prototype existant
# ---------------------------------------------------------------------

class InterfacePrototype:
    """
    Enveloppe verifiee autour de prototype_futur_passe.py.

    Elle controle :
        - la presence des constantes et fonctions necessaires ;
        - la presence et la coherence des formes des parametres ;
        - la presence, la finitude et la coherence temporelle
          des champs de simulation.
    """

    def __init__(self, nom_module):
        self.nom_module = nom_module
        self.module = None

        self.N = None
        self.duree = None
        self.graine = None
        self.bruit_futur = 0.025

    def charger(self):
        if str(DOSSIER_SCRIPT) not in sys.path:
            sys.path.insert(0, str(DOSSIER_SCRIPT))

        try:
            self.module = importlib.import_module(self.nom_module)
        except ImportError as erreur:
            raise SystemExit(
                f"Impossible d'importer {self.nom_module!r}.\n"
                f"Placez le fichier dans : {DOSSIER_SCRIPT}\n"
                f"Detail : {erreur}"
            )

        necessaires = (
            "N",
            "DUREE",
            "SEED",
            "initialiser_parametres",
            "simuler",
        )

        absents = [
            nom
            for nom in necessaires
            if not hasattr(self.module, nom)
        ]

        if absents:
            raise RuntimeError(
                f"Elements absents de {self.nom_module} : "
                + ", ".join(absents)
            )

        self.N = int(self.module.N)
        self.duree = float(self.module.DUREE)
        self.graine = int(self.module.SEED)

        self.bruit_futur = float(
            getattr(self.module, "BRUIT_FUTUR", 0.025)
        )

        if self.N < 1:
            raise ValueError("N doit etre positif.")

        if not np.isfinite(self.duree) or self.duree <= 0.0:
            raise ValueError("DUREE doit etre finie et positive.")

        if not np.isfinite(self.bruit_futur) or self.bruit_futur < 0.0:
            raise ValueError("BRUIT_FUTUR doit etre fini et >= 0.")

        return self

    @property
    def chemin_fichier(self):
        fichier = getattr(self.module, "__file__", None)

        return Path(fichier) if fichier else None

    def executer(self):
        """
        Accepte :
            simuler(rng, points, parametres)

        ou :
            simuler(rng, points, parametres, graine_futur=...)
        """
        if self.module is None:
            raise RuntimeError("Appeler charger() avant executer().")

        rng = np.random.default_rng(self.graine)

        points = rng.uniform(
            0.0,
            1.0,
            size=(self.N, 2),
        )

        parametres = self.module.initialiser_parametres(
            rng,
            points,
        )

        self._verifier_parametres(parametres)

        signature = inspect.signature(self.module.simuler)

        arguments = {}

        if "graine_futur" in signature.parameters:
            arguments["graine_futur"] = 1

        brut = self.module.simuler(
            rng,
            points,
            parametres,
            **arguments,
        )

        resultats = self._verifier_resultats(brut, parametres)

        return points, parametres, resultats

    def _verifier_parametres(self, parametres):
        for nom in PARAMETRES_RESEAU:
            if nom not in parametres:
                raise KeyError(
                    f"Parametre absent du prototype : {nom}"
                )

            matrice = np.asarray(parametres[nom], dtype=float)

            if matrice.ndim != 2:
                raise ValueError(
                    f"{nom} doit etre une matrice (ndim={matrice.ndim})."
                )

            verifier_fini(matrice, nom)

        dM, g1 = np.asarray(parametres["W_MEM_G"]).shape
        dM2, dC = np.asarray(parametres["W_MEM_C"]).shape
        dM3, dH = np.asarray(parametres["W_MEM_H"]).shape

        if not (dM == dM2 == dM3):
            raise ValueError(
                "Les matrices W_MEM_* n'ont pas le meme nombre de lignes."
            )

        if g1 != DIMENSION_ETAT_GLOBAL:
            raise ValueError(
                f"W_MEM_G doit avoir {DIMENSION_ETAT_GLOBAL} colonnes "
                f"(etat global = [mean E, mean I]) : recu {g1}."
            )

        verifier_forme(parametres["W_HH"], (dH, dH), "W_HH")
        verifier_forme(parametres["W_HC"], (dH, dC), "W_HC")
        verifier_forme(
            parametres["W_HG"],
            (dH, DIMENSION_ETAT_GLOBAL),
            "W_HG",
        )
        verifier_forme(
            parametres["decodeur_scene"],
            (self.N, dH),
            "decodeur_scene",
        )

        self.dimension_H = dH
        self.dimension_C = dC
        self.dimension_M = dM

    def _verifier_resultats(self, brut, parametres):
        resultats = dict(brut)

        for champ in CHAMPS_SIMULATION:
            if champ not in brut:
                raise KeyError(
                    f"Champ absent des resultats : {champ}"
                )

            resultats[champ] = verifier_fini(
                brut[champ],
                champ,
            )

        temps = resultats["temps"]

        if temps.ndim != 1 or len(temps) < 2:
            raise ValueError(
                "Historique temporel invalide."
            )

        if np.any(np.diff(temps) <= 0.0):
            raise ValueError(
                "Les temps doivent etre strictement croissants."
            )

        T = len(temps)

        formes_attendues = {
            "E": (T, self.N),
            "I": (T, self.N),
            "scene": (T, self.N),
            "H": (T, self.dimension_H),
            "C": (T, self.dimension_C),
            "M_passe": (T, self.dimension_M),
        }

        for champ, forme in formes_attendues.items():
            verifier_forme(resultats[champ], forme, champ)

        return resultats


def encoder_memoire(
    etat_global,
    contexte,
    H,
    parametres,
):
    """
    Encodeur commun aux etats presentes et simules.
    """
    return np.tanh(
        parametres["W_MEM_G"] @ etat_global
        + parametres["W_MEM_C"] @ contexte
        + parametres["W_MEM_H"] @ H
    )


def generer_futur(
    H,
    contexte,
    etat_global,
    parametres,
    bruit,
    amplitude_bruit,
):
    """
    Proposition latente conditionnelle.

    Cette formule conserve le generateur propose.
    Ce n'est pas une integration Euler a l'horizon DT.
    """
    return np.tanh(
        parametres["W_HH"] @ H
        + parametres["W_HC"] @ contexte
        + parametres["W_HG"] @ etat_global
        + amplitude_bruit * bruit
    )


# ---------------------------------------------------------------------
# Representation multimodale
# ---------------------------------------------------------------------

def resume_scene(scene, points):
    scene = np.asarray(scene, dtype=float)
    points = np.asarray(points, dtype=float)

    if scene.ndim != 1:
        raise ValueError(
            "scene doit etre un vecteur."
        )

    if points.shape != (len(scene), 2):
        raise ValueError(
            "points doit avoir la forme (N, 2)."
        )

    poids = np.maximum(scene, 0.0)
    somme = float(poids.sum())

    if somme <= 1e-12:
        centre = points.mean(axis=0)
    else:
        centre = (
            poids[:, None] * points
        ).sum(axis=0) / somme

    return np.array([
        scene.mean(),
        scene.std(),
        centre[0],
        centre[1],
    ], dtype=float)


def construire_vecteur_multimodal(
    H,
    C,
    etat_global,
    M,
    scene,
    points,
):
    return np.concatenate([
        np.asarray(H, dtype=float),
        np.asarray(C, dtype=float),
        np.asarray(etat_global, dtype=float),
        np.asarray(M, dtype=float),
        resume_scene(scene, points),
    ])


def construire_etat_present(
    resultats,
    index,
    points,
):
    E = np.asarray(resultats["E"][index], dtype=float)
    I = np.asarray(resultats["I"][index], dtype=float)
    C = np.asarray(resultats["C"][index], dtype=float)
    H = np.asarray(resultats["H"][index], dtype=float)
    M = np.asarray(resultats["M_passe"][index], dtype=float)
    scene = np.asarray(resultats["scene"][index], dtype=float)

    global_EI = np.array([
        E.mean(),
        I.mean(),
    ])

    return {
        "H": H.copy(),
        "C": C.copy(),
        "M_passe": M.copy(),
        "etat_global": global_EI,
        "X_present": construire_vecteur_multimodal(
            H,
            C,
            global_EI,
            M,
            scene,
            points,
        ),
    }


def calculer_contexte_constant(
    resultats,
    indices,
):
    # Conversion en tableau AVANT l'indexation par tableau d'indices.
    return np.asarray(
        resultats["C"],
        dtype=float,
    )[indices].mean(axis=0)


# ---------------------------------------------------------------------
# Ensembles futurs et permutations
# ---------------------------------------------------------------------

def generer_ensemble_futur(
    etat,
    parametres,
    points,
    bruits,
    amplitude_bruit,
    contexte_futur=None,
):
    bruits = np.asarray(bruits, dtype=float)

    if bruits.ndim != 2 or bruits.shape[1] != len(etat["H"]):
        raise ValueError(
            "bruits doit avoir la forme (branches, dimension_H)."
        )

    if contexte_futur is None:
        contexte = etat["C"].copy()
    else:
        contexte = np.asarray(
            contexte_futur,
            dtype=float,
        ).copy()

    if contexte.shape != etat["C"].shape:
        raise ValueError(
            "Forme du contexte incompatible."
        )

    H_futurs = []
    M_futurs = []
    scenes = []
    X_futurs = []

    for bruit in bruits:
        Hf = generer_futur(
            etat["H"],
            contexte,
            etat["etat_global"],
            parametres,
            bruit,
            amplitude_bruit,
        )

        Mf = encoder_memoire(
            etat["etat_global"],
            contexte,
            Hf,
            parametres,
        )

        scene = sigmoid(
            parametres["decodeur_scene"] @ Hf
        )

        H_futurs.append(Hf)
        M_futurs.append(Mf)
        scenes.append(scene)

        X_futurs.append(
            construire_vecteur_multimodal(
                Hf,
                contexte,
                etat["etat_global"],
                Mf,
                scene,
                points,
            )
        )

    return {
        "X_present": etat["X_present"].copy(),
        "X_futurs": np.asarray(X_futurs),
        "H_futurs": np.asarray(H_futurs),
        "M_futurs": np.asarray(M_futurs),
        "scenes_futures": np.asarray(scenes),
        "contexte_utilise": contexte,
        "etat_global": etat["etat_global"].copy(),
    }


def permuter_ensemble_scenes(
    ensemble,
    permutation,
):
    """
    Les quatre dernieres coordonnees sont le resume de scene.

    Permuter ces lignes conserve la distribution des scenes,
    mais change leur association avec H et M.
    """
    permutation = np.asarray(permutation)

    if (
        permutation.ndim != 1
        or len(permutation) != len(ensemble["X_futurs"])
        or not np.array_equal(
            np.sort(permutation),
            np.arange(len(permutation)),
        )
    ):
        raise ValueError(
            "permutation doit etre une permutation de 0..n-1."
        )

    resultat = dict(ensemble)

    X = ensemble["X_futurs"].copy()

    X[:, -NOMBRE_COORDONNEES_SCENE:] = (
        ensemble["X_futurs"][
            permutation,
            -NOMBRE_COORDONNEES_SCENE:,
        ]
    )

    resultat["X_futurs"] = X

    resultat["permutation_scenes"] = (
        permutation.copy()
    )

    return resultat


# ---------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------

def statistiques_covariance(
    Z,
    regularisation=1e-8,
):
    Z = verifier_matrice(Z, "Z")

    n, k = Z.shape

    if n <= k:
        raise ValueError(
            "Le nombre de branches doit depasser "
            "la dimension latente."
        )

    covariance = np.atleast_2d(
        np.cov(
            Z,
            rowvar=False,
            ddof=1,
        )
    )

    covariance = 0.5 * (
        covariance + covariance.T
    )

    valeurs = np.linalg.eigvalsh(
        covariance
    )

    valeurs = np.maximum(
        valeurs,
        0.0,
    )

    covariance_reg = (
        covariance
        + regularisation * np.eye(k)
    )

    signe, logdet = np.linalg.slogdet(
        covariance_reg
    )

    if signe <= 0 or not np.isfinite(logdet):
        raise FloatingPointError(
            "Covariance regularisee invalide."
        )

    log_volume = (
        log_ball_volume_factor(k)
        + 0.5 * logdet
    )

    return {
        "volume": exponentielle_sure(log_volume),
        "log_volume": float(log_volume),
        "dispersion": float(np.trace(covariance)),
        "rang": int(
            np.linalg.matrix_rank(covariance)
        ),
        "rang_effectif": rang_effectif(valeurs),
        "valeurs_propres": valeurs,
    }


def volume_covariance_latente(Z, regularisation=1e-8):
    return statistiques_covariance(
        Z,
        regularisation,
    )["volume"]


def analyser_volume_futur(
    ensemble,
    metrique,
    regularisation=1e-8,
):
    Zp = metrique.transform(
        ensemble["X_present"][None, :]
    )[0]

    Zf = metrique.transform(
        ensemble["X_futurs"]
    )

    centre = Zf.mean(axis=0)

    geometrie = statistiques_covariance(
        Zf,
        regularisation,
    )

    ancrage = cosinus(
        Zp,
        centre,
    )

    distance = float(
        np.linalg.norm(Zp - centre)
    )

    attachement = float(
        np.clip(
            0.5 * (ancrage + 1.0),
            0.0,
            1.0,
        )
    )

    if attachement > 0.0:
        log_attache = (
            geometrie["log_volume"]
            + np.log(attachement)
        )
    else:
        log_attache = -np.inf

    joint = statistiques_covariance(
        np.vstack([
            Zp[None, :],
            Zf,
        ]),
        regularisation,
    )

    return {
        "ancrage": ancrage,
        "distance": distance,
        "volume_futur": geometrie["volume"],
        "log_volume_futur": geometrie["log_volume"],
        "volume_attache": exponentielle_sure(
            log_attache
        ),
        "log_volume_attache": float(log_attache),
        "volume_joint": joint["volume"],
        "log_volume_joint": joint["log_volume"],
        "dispersion": geometrie["dispersion"],
        "rang_covariance": geometrie["rang"],
        "rang_effectif": geometrie["rang_effectif"],
        "centre_futur": centre,
        "Z_present": Zp,
        "Z_futurs": Zf,
        "valeurs_propres_covariance": (
            geometrie["valeurs_propres"]
        ),
    }


# ---------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------

def transformer_ancrage(A):
    return np.arctanh(
        np.clip(
            np.asarray(A, dtype=float),
            -0.999999,
            0.999999,
        )
    )


def indices_bootstrap_blocs(
    n,
    longueur_bloc,
    rng,
):
    nombre_blocs = int(
        np.ceil(n / longueur_bloc)
    )

    debuts = rng.integers(
        0,
        n,
        size=nombre_blocs,
    )

    indices = (
        debuts[:, None]
        + np.arange(longueur_bloc)[None, :]
    ) % n

    return indices.ravel()[:n]


def bootstrap_moyenne_contraste(
    contraste,
    longueur_bloc,
    nombre_bootstrap,
    graine,
):
    contraste = np.asarray(
        contraste,
        dtype=float,
    )

    if contraste.ndim != 1:
        raise ValueError(
            "Le contraste doit etre un vecteur."
        )

    if not np.all(np.isfinite(contraste)):
        return {
            "statut": "non calcule : contraste non fini",
            "estimation": None,
            "intervalle_95": None,
        }

    n = len(contraste)

    if not 1 <= longueur_bloc < n:
        raise ValueError(
            "Longueur de bloc incompatible."
        )

    rng = np.random.default_rng(graine)

    estimations = np.empty(
        nombre_bootstrap
    )

    for repetition in range(nombre_bootstrap):
        indices = indices_bootstrap_blocs(
            n,
            longueur_bloc,
            rng,
        )

        estimations[repetition] = (
            contraste[indices].mean()
        )

    return {
        "statut": (
            "descriptif, conditionnel a cette trajectoire"
        ),
        "estimation": float(contraste.mean()),
        "intervalle_95": np.quantile(
            estimations,
            [0.025, 0.975],
        ).tolist(),
        "ecart_type_bootstrap": float(
            estimations.std(ddof=1)
        ),
    }


def rang_permutation_superieur(
    reel,
    permutations,
):
    reel = np.asarray(
        reel,
        dtype=float,
    )

    permutations = np.asarray(
        permutations,
        dtype=float,
    )

    if permutations.shape[1:] != reel.shape:
        raise ValueError(
            "Formes de permutation incompatibles."
        )

    if not (
        np.all(np.isfinite(reel))
        and np.all(np.isfinite(permutations))
    ):
        return np.full(
            reel.shape,
            np.nan,
        )

    tolerance = (
        1e-12
        * np.maximum(1.0, np.abs(reel))
    )

    return (
        1
        + np.sum(
            permutations
            >= reel[None, :] - tolerance[None, :],
            axis=0,
        )
    ) / (len(permutations) + 1)


def creer_stockage(forme):
    return {
        champ: np.empty(
            forme,
            dtype=float,
        )
        for champ in CHAMPS_MESURES
    }


def enregistrer_mesure(
    stockage,
    position,
    mesure,
):
    for champ in CHAMPS_MESURES:
        stockage[champ][position] = mesure[champ]


def resume_descriptif(stockage, moyenner_repetitions):
    """
    Resume par champ : moyenne, ecart-type, minimum et maximum
    de la serie temporelle (moyennee sur les repetitions si besoin).
    """
    resume = {}

    for champ in CHAMPS_MESURES:
        serie = np.asarray(stockage[champ], dtype=float)

        if moyenner_repetitions:
            serie = serie.mean(axis=0)

        if not np.all(np.isfinite(serie)):
            resume[champ] = {
                "statut": "contient des valeurs non finies",
            }
            continue

        resume[champ] = {
            "moyenne": float(serie.mean()),
            "ecart_type": float(serie.std(ddof=1)) if len(serie) > 1 else 0.0,
            "minimum": float(serie.min()),
            "maximum": float(serie.max()),
        }

    return resume


def construire_statistiques(
    reel_rep,
    constant_rep,
    reel,
    permute,
    configuration,
):
    contrastes = {
        "ancrage_contexte": (
            transformer_ancrage(
                reel_rep["ancrage"]
            )
            - transformer_ancrage(
                constant_rep["ancrage"]
            )
        ).mean(axis=0),

        "distance_contexte": (
            reel_rep["distance"]
            - constant_rep["distance"]
        ).mean(axis=0),

        "log_volume_futur_contexte": (
            reel_rep["log_volume_futur"]
            - constant_rep["log_volume_futur"]
        ).mean(axis=0),

        "log_volume_attache_contexte": (
            reel_rep["log_volume_attache"]
            - constant_rep["log_volume_attache"]
        ).mean(axis=0),

        "log_volume_futur_permutation": (
            reel["log_volume_futur"]
            - permute["log_volume_futur"].mean(axis=0)
        ),

        "log_volume_attache_permutation": (
            reel["log_volume_attache"]
            - permute["log_volume_attache"].mean(axis=0)
        ),
    }

    rapport = {}

    for nom, serie in contrastes.items():
        fini = bool(
            np.all(np.isfinite(serie))
        )

        entree = {
            "contraste_par_instant": serie,
            "moyenne": (
                float(serie.mean())
                if fini
                else None
            ),
            "nombre_instants": int(len(serie)),
            "bootstrap_blocs": {},
        }

        if fini and len(serie) > 1:
            ecart = float(serie.std(ddof=1))

            entree["ecart_type_instants"] = ecart

            entree["taille_effet_standardisee"] = (
                float(serie.mean() / ecart)
                if ecart > 1e-15
                else None
            )

            entree["fraction_instants_positifs"] = float(
                np.mean(serie > 0.0)
            )

        if nom.startswith("log_volume") and fini:
            entree["ratio_geometrique"] = (
                exponentielle_sure(
                    float(serie.mean())
                )
            )

        for longueur in configuration.longueurs_blocs:
            if 1 <= longueur < len(serie):
                entree["bootstrap_blocs"][
                    str(longueur)
                ] = bootstrap_moyenne_contraste(
                    serie,
                    longueur,
                    configuration.nombre_bootstrap,
                    configuration.graine_bootstrap,
                )

        rapport[nom] = entree

    rapport["permutation_volumes"] = {}

    alpha = configuration.seuil_alpha

    for champ in (
        "log_volume_futur",
        "log_volume_attache",
    ):
        rangs = rang_permutation_superieur(
            reel[champ],
            permute[champ],
        )

        holm = correction_holm(rangs)
        bh = correction_benjamini_hochberg(rangs)

        finis = np.isfinite(rangs)

        rapport["permutation_volumes"][champ] = {
            "rangs_unilateraux_par_instant": rangs,
            "rangs_holm": holm,
            "rangs_benjamini_hochberg": bh,
            "nombre_sous_alpha_brut": int(
                np.sum(rangs[finis] <= alpha)
            ),
            "nombre_sous_alpha_holm": int(
                np.sum(holm[finis] <= alpha)
            ),
            "nombre_sous_alpha_bh": int(
                np.sum(bh[finis] <= alpha)
            ),
            "alpha": alpha,
            "interpretation": (
                "Distribution de reassignation des scenes. "
                "Pas une validation biologique ni une inference "
                "generale sur plusieurs reseaux. "
                "Les rangs bruts ne sont pas corriges pour les "
                "comparaisons multiples ; les colonnes Holm et "
                "Benjamini-Hochberg le sont, mais les instants sont "
                "autocorreles, ce qui limite leur portee."
            ),
        }

    rapport[
        "invariance_centre_scenes_permutees"
    ] = {
        "ecart_max_ancrage": float(
            np.max(
                np.abs(
                    permute["ancrage"]
                    - reel["ancrage"][None, :]
                )
            )
        ),
        "ecart_max_distance": float(
            np.max(
                np.abs(
                    permute["distance"]
                    - reel["distance"][None, :]
                )
            )
        ),
        "interpretation": (
            "Avec une metrique affine, permuter les resumes "
            "de scenes entre branches conserve leur centre. "
            "Ancrage et distance doivent donc rester inchanges "
            "a l'arrondi numerique pres."
        ),
    }

    return rapport


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def convertir_json(objet):
    if isinstance(objet, dict):
        return {
            str(cle): convertir_json(valeur)
            for cle, valeur in objet.items()
        }

    if isinstance(objet, (list, tuple)):
        return [
            convertir_json(valeur)
            for valeur in objet
        ]

    if isinstance(objet, np.ndarray):
        return convertir_json(
            objet.tolist()
        )

    if isinstance(objet, (bool, np.bool_)):
        return bool(objet)

    if isinstance(objet, (np.integer, int)):
        return int(objet)

    if isinstance(objet, (np.floating, float)):
        valeur = float(objet)

        if np.isfinite(valeur):
            return valeur

        return None

    if isinstance(objet, Path):
        return str(objet)

    return objet


def ecrire_json_atomique(chemin, objet):
    """
    Ecrit dans un fichier temporaire du meme dossier puis remplace
    le fichier final : un plantage ne laisse pas de JSON tronque.
    """
    chemin = Path(chemin)

    descripteur, nom_temporaire = tempfile.mkstemp(
        dir=str(chemin.parent),
        prefix=chemin.name + ".",
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            descripteur,
            "w",
            encoding="utf-8",
        ) as fichier:
            json.dump(
                convertir_json(objet),
                fichier,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        os.replace(nom_temporaire, chemin)

    except BaseException:
        if os.path.exists(nom_temporaire):
            os.remove(nom_temporaire)

        raise


def exporter_csv(
    dossier,
    temps,
    reel,
    reel_rep,
    constant_rep,
    permute,
):
    chemin = dossier / "mesures_futures.csv"

    conditions = {
        "reel": {
            cle: reel[cle]
            for cle in CHAMPS_MESURES
        },

        "reel_repete_moyen": {
            cle: reel_rep[cle].mean(axis=0)
            for cle in CHAMPS_MESURES
        },

        "contexte_constant_moyen": {
            cle: constant_rep[cle].mean(axis=0)
            for cle in CHAMPS_MESURES
        },

        "scenes_permutees_moyen": {
            cle: permute[cle].mean(axis=0)
            for cle in CHAMPS_MESURES
        },
    }

    colonnes = ["temps"]

    for condition in conditions:
        for champ in CHAMPS_MESURES:
            colonnes.append(
                f"{condition}_{champ}"
            )

    with chemin.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fichier:
        auteur = csv.DictWriter(
            fichier,
            fieldnames=colonnes,
        )

        auteur.writeheader()

        for index, temps_courant in enumerate(temps):
            ligne = {
                "temps": float(temps_courant)
            }

            for condition, mesures in conditions.items():
                for champ in CHAMPS_MESURES:
                    ligne[
                        f"{condition}_{champ}"
                    ] = float(
                        mesures[champ][index]
                    )

            auteur.writerow(ligne)

    return chemin


def exporter_npz(
    dossier,
    configuration,
    temps,
    indices_analyse,
    metrique,
    presents_latents,
    centres_futurs,
    branches_latentes,
    spectres_covariance,
    rangs_effectifs,
    reel,
    reel_rep,
    constant_rep,
    permute,
):
    donnees = {
        "temps": temps,
        "indices_analyse": indices_analyse,

        "presents_latents": np.asarray(
            presents_latents
        ),

        "centres_futurs": np.asarray(
            centres_futurs
        ),

        "spectres_covariance": np.asarray(
            spectres_covariance
        ),

        "rangs_effectifs": np.asarray(
            rangs_effectifs
        ),

        "metrique_moyenne": metrique.moyenne_,
        "metrique_ecart_type": metrique.ecart_type_,
        "metrique_projection": metrique.projection_,
    }

    if configuration.ecrire_branches_npz:
        donnees["branches_latentes"] = np.asarray(
            branches_latentes
        )

    for nom, stockage in (
        ("reel", reel),
        ("reel_repete", reel_rep),
        ("constant", constant_rep),
        ("permute", permute),
    ):
        for champ, valeurs in stockage.items():
            donnees[f"{nom}_{champ}"] = valeurs

    chemin = dossier / "ensembles_futurs.npz"

    np.savez_compressed(chemin, **donnees)

    return chemin


# ---------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------

def sauver_figure(fig, chemin):
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

    return chemin


def tracer_mesures(
    dossier,
    temps,
    reel_rep,
    constant_rep,
    permute,
):
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        sharex=True,
    )

    panneaux = (
        (
            "log_volume_futur",
            "Log-volume futur",
        ),
        (
            "ancrage",
            "Ancrage present / centre futur",
        ),
        (
            "distance",
            "Distance present / centre futur",
        ),
        (
            "log_volume_attache",
            "Log-volume futur attache",
        ),
    )

    for axe, (champ, titre) in zip(
        axes,
        panneaux,
    ):
        axe.plot(
            temps,
            reel_rep[champ].mean(axis=0),
            label=(
                "Contexte actuel, "
                "moyenne des repetitions"
            ),
        )

        axe.plot(
            temps,
            constant_rep[champ].mean(axis=0),
            label=(
                "Contexte constant, "
                "bruits apparies"
            ),
            linestyle="--",
        )

        axe.plot(
            temps,
            permute[champ].mean(axis=0),
            label=(
                "Scenes permutees, "
                "ensemble fixe"
            ),
            linestyle=":",
        )

        axe.set_ylabel(titre)
        axe.grid(alpha=0.25)
        axe.legend(fontsize=8)

    axes[-1].set_xlabel(
        "Temps du modele"
    )

    fig.suptitle(
        "Geometrie des propositions futures "
        "- modele synthetique"
    )

    return sauver_figure(
        fig,
        dossier / "mesures_futures.png",
    )


def tracer_contrastes(dossier, statistiques, configuration):
    noms = [
        nom
        for nom, entree in statistiques.items()
        if isinstance(entree, dict) and "moyenne" in entree
    ]

    longueur_ref = None

    for longueur in configuration.longueurs_blocs:
        longueur_ref = str(longueur)

        if all(
            longueur_ref in statistiques[nom]["bootstrap_blocs"]
            for nom in noms
        ):
            break

    moyennes = []
    bas = []
    haut = []

    for nom in noms:
        entree = statistiques[nom]
        moyenne = entree["moyenne"]

        moyennes.append(
            np.nan if moyenne is None else moyenne
        )

        intervalle = (
            entree["bootstrap_blocs"]
            .get(longueur_ref, {})
            .get("intervalle_95")
        )

        if intervalle is None or moyenne is None:
            bas.append(0.0)
            haut.append(0.0)
        else:
            bas.append(moyenne - intervalle[0])
            haut.append(intervalle[1] - moyenne)

    fig, axe = plt.subplots(figsize=(10, 5))

    positions = np.arange(len(noms))

    axe.errorbar(
        moyennes,
        positions,
        xerr=[bas, haut],
        fmt="o",
        capsize=4,
    )

    axe.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    axe.set_yticks(positions)
    axe.set_yticklabels(noms, fontsize=8)
    axe.set_xlabel(
        "Contraste moyen "
        f"(IC 95 % bootstrap, blocs de {longueur_ref})"
    )
    axe.grid(alpha=0.25)

    fig.suptitle(
        "Contrastes apparies - intervalles descriptifs"
    )

    return sauver_figure(
        fig,
        dossier / "contrastes_bootstrap.png",
    )


def tracer_spectres(dossier, spectres_covariance):
    spectres = np.asarray(spectres_covariance, dtype=float)

    spectres = np.sort(spectres, axis=1)[:, ::-1]
    plancher = 1e-30

    moyenne = np.maximum(spectres.mean(axis=0), plancher)
    minimum = np.maximum(spectres.min(axis=0), plancher)
    maximum = np.maximum(spectres.max(axis=0), plancher)

    axes_x = np.arange(1, spectres.shape[1] + 1)

    fig, axe = plt.subplots(figsize=(8, 5))

    axe.fill_between(axes_x, minimum, maximum, alpha=0.25)
    axe.plot(axes_x, moyenne, marker="o")
    axe.set_yscale("log")
    axe.set_xlabel("Rang de la valeur propre")
    axe.set_ylabel("Valeur propre de la covariance")
    axe.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Spectre de covariance des branches futures "
        "(moyenne et plage sur les instants)"
    )

    return sauver_figure(
        fig,
        dossier / "spectres_covariance.png",
    )


def tracer_projection_latente(
    dossier,
    temps,
    presents_latents,
    branches_latentes,
):
    nombre = len(branches_latentes)

    choix = sorted(set([0, nombre // 2, nombre - 1]))

    fig, axes = plt.subplots(
        1,
        len(choix),
        figsize=(5 * len(choix), 4.5),
        squeeze=False,
    )

    for axe, position in zip(axes[0], choix):
        Zf = np.asarray(branches_latentes[position], dtype=float)
        Zp = np.asarray(presents_latents[position], dtype=float)

        centre = Zf.mean(axis=0)

        _, _, vt = np.linalg.svd(
            Zf - centre,
            full_matrices=False,
        )

        base = vt[:2].T

        if base.shape[1] < 2:
            base = np.hstack([
                base,
                np.zeros((base.shape[0], 2 - base.shape[1])),
            ])

        projete = (Zf - centre) @ base
        present = (Zp - centre) @ base

        axe.scatter(
            projete[:, 0],
            projete[:, 1],
            s=12,
            alpha=0.6,
            label="branches futures",
        )

        axe.scatter(
            [0.0],
            [0.0],
            marker="+",
            s=120,
            label="centre futur",
        )

        axe.scatter(
            [present[0]],
            [present[1]],
            marker="*",
            s=160,
            label="present",
        )

        axe.set_title(f"t = {temps[position]:.2f}")
        axe.set_xlabel("PC1 (du nuage futur)")
        axe.set_ylabel("PC2")
        axe.grid(alpha=0.25)
        axe.legend(fontsize=7)

    fig.suptitle(
        "Projection 2D des branches latentes (ACP locale)"
    )

    return sauver_figure(
        fig,
        dossier / "projection_latente.png",
    )


def tracer_rangs_permutation(dossier, statistiques, configuration):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    for axe, champ in zip(
        axes,
        ("log_volume_futur", "log_volume_attache"),
    ):
        donnees = statistiques["permutation_volumes"][champ]

        brut = np.asarray(
            donnees["rangs_unilateraux_par_instant"],
            dtype=float,
        )

        holm = np.asarray(donnees["rangs_holm"], dtype=float)
        instants = np.arange(len(brut))

        axe.plot(instants, brut, marker="o", label="brut")
        axe.plot(instants, holm, marker="s", label="Holm")
        axe.axhline(
            configuration.seuil_alpha,
            color="grey",
            linestyle="--",
            linewidth=1,
        )

        axe.set_title(champ)
        axe.set_xlabel("Instant analyse")
        axe.grid(alpha=0.25)
        axe.legend(fontsize=8)

    axes[0].set_ylabel("Rang unilateral (valeur p de permutation)")

    fig.suptitle(
        "Rangs de permutation par instant (instants autocorreles)"
    )

    return sauver_figure(
        fig,
        dossier / "rangs_permutation.png",
    )


# ---------------------------------------------------------------------
# Controles de coherence
# ---------------------------------------------------------------------

def mesure_base_instant(
    numero,
    index,
    resultats,
    points,
    parametres,
    metrique,
    configuration,
    amplitude_bruit,
):
    """
    Recalcule, de facon autonome, la mesure de l'ensemble de base d'un
    instant. Sert au controle de determinisme.
    """
    etat = construire_etat_present(
        resultats,
        index,
        points,
    )

    rng_base = np.random.default_rng(
        np.random.SeedSequence([
            configuration.graine_ensemble,
            numero,
            0,
        ])
    )

    bruits = rng_base.normal(
        size=(
            configuration.nombre_branches,
            len(etat["H"]),
        )
    )

    ensemble = generer_ensemble_futur(
        etat,
        parametres,
        points,
        bruits,
        amplitude_bruit,
    )

    return analyser_volume_futur(
        ensemble,
        metrique,
        configuration.regularisation_covariance,
    )


def controle_equivalence_contexte(
    etat,
    parametres,
    points,
    bruits,
    amplitude_bruit,
    ensemble_base,
):
    """
    Un contexte constant egal au contexte actuel doit redonner
    exactement l'ensemble de base : verifie le chemin de code.
    """
    ensemble = generer_ensemble_futur(
        etat,
        parametres,
        points,
        bruits,
        amplitude_bruit,
        contexte_futur=etat["C"],
    )

    return float(
        np.max(
            np.abs(
                ensemble["X_futurs"]
                - ensemble_base["X_futurs"]
            )
        )
    )


def controle_determinisme(mesure_a, valeurs_b):
    ecarts = {}

    for champ in CHAMPS_MESURES:
        ecarts[champ] = float(
            abs(mesure_a[champ] - valeurs_b[champ])
            if np.isfinite(mesure_a[champ])
            and np.isfinite(valeurs_b[champ])
            else (
                0.0
                if mesure_a[champ] == valeurs_b[champ]
                else float("inf")
            )
        )

    return ecarts


# ---------------------------------------------------------------------
# Autotests numeriques
# ---------------------------------------------------------------------

def _assert_proche(a, b, tolerance=1e-9, message=""):
    if not abs(a - b) <= tolerance:
        raise AssertionError(
            f"{message} : {a!r} different de {b!r} "
            f"(tolerance {tolerance})."
        )


def test_facteur_boule():
    _assert_proche(ball_volume_factor(1), 2.0, 1e-12, "boule k=1")
    _assert_proche(ball_volume_factor(2), pi, 1e-12, "boule k=2")
    _assert_proche(
        ball_volume_factor(3),
        4.0 * pi / 3.0,
        1e-12,
        "boule k=3",
    )


def test_cosinus():
    _assert_proche(cosinus([1, 0], [0, 1]), 0.0, 1e-15, "orthogonaux")
    _assert_proche(cosinus([1, 2], [2, 4]), 1.0, 1e-12, "paralleles")
    _assert_proche(cosinus([1, 2], [-1, -2]), -1.0, 1e-12, "opposes")
    _assert_proche(cosinus([0, 0], [1, 2]), 0.0, 1e-15, "vecteur nul")


def test_distances_euclidiennes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))

    D = distances_euclidiennes(X)

    maximum = 0.0

    for i in range(len(X)):
        for j in range(len(X)):
            maximum = max(
                maximum,
                abs(D[i, j] - np.linalg.norm(X[i] - X[j])),
            )

    _assert_proche(maximum, 0.0, 1e-9, "distances vs force brute")

    if not np.allclose(D, D.T):
        raise AssertionError("D non symetrique.")


def test_mds_reproduit_distances():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))

    D = distances_euclidiennes(X)

    metrique = SubjectiveMetric(
        k=5,
        reg=1e-10,
        standardize=False,
    )

    metrique.fit(X, D)

    diagnostic = metrique.diagnostics(X, D)

    if diagnostic["stress_kruskal"] > 1e-5:
        raise AssertionError(
            f"Stress trop eleve : {diagnostic['stress_kruskal']}"
        )

    if diagnostic["correlation_distances"] < 0.999999:
        raise AssertionError("Correlation des distances trop faible.")


def test_metrique_sauvegarde_chargement():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))

    D = distances_euclidiennes(X)

    metrique = SubjectiveMetric(k=3).fit(X, D)

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "metrique.npz"

        metrique.sauvegarder(chemin)
        copie = SubjectiveMetric.charger(chemin)

    ecart = float(
        np.max(np.abs(metrique.transform(X) - copie.transform(X)))
    )

    _assert_proche(ecart, 0.0, 1e-12, "metrique rechargee")


def test_covariance_et_echelle():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(200, 3))

    a = statistiques_covariance(Z, 1e-12)
    b = statistiques_covariance(2.0 * Z, 1e-12)

    _assert_proche(
        b["log_volume"] - a["log_volume"],
        3.0 * np.log(2.0),
        1e-6,
        "mise a l'echelle du log-volume",
    )

    covariance = np.cov(Z, rowvar=False, ddof=1)

    attendu = (
        log_ball_volume_factor(3)
        + 0.5 * np.linalg.slogdet(
            covariance + 1e-12 * np.eye(3)
        )[1]
    )

    _assert_proche(a["log_volume"], attendu, 1e-9, "log-volume")

    if a["rang"] != 3:
        raise AssertionError("Rang attendu : 3.")


def test_covariance_degeneree():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(50, 2))
    Z = np.hstack([base, base[:, :1]])

    resultat = statistiques_covariance(Z, 1e-8)

    if resultat["rang"] != 2:
        raise AssertionError(
            f"Rang attendu 2, obtenu {resultat['rang']}."
        )

    if not np.isfinite(resultat["log_volume"]):
        raise AssertionError("Log-volume non fini sur nuage degenere.")


def test_bootstrap_blocs():
    rng = np.random.default_rng(6)

    indices = indices_bootstrap_blocs(10, 4, rng)

    if len(indices) != 10:
        raise AssertionError("Longueur d'indices incorrecte.")

    if indices.min() < 0 or indices.max() > 9:
        raise AssertionError("Indices hors bornes.")

    # Dans chaque bloc complet, les indices se suivent modulo n.
    for debut in (0, 4):
        bloc = indices[debut:debut + 4]
        suites = (np.diff(bloc) % 10) == 1

        if not np.all(suites):
            raise AssertionError("Blocs non contigus.")


def test_bootstrap_reproductible():
    serie = np.sin(np.linspace(0, 6, 40))

    a = bootstrap_moyenne_contraste(serie, 4, 300, 7)
    b = bootstrap_moyenne_contraste(serie, 4, 300, 7)

    if a["intervalle_95"] != b["intervalle_95"]:
        raise AssertionError("Bootstrap non reproductible.")

    if not (
        a["intervalle_95"][0]
        <= a["estimation"]
        <= a["intervalle_95"][1]
    ):
        raise AssertionError("Estimation hors de son intervalle.")


def test_rang_permutation():
    permutations = np.zeros((9, 2))

    rangs = rang_permutation_superieur(
        np.array([1.0, -1.0]),
        permutations,
    )

    _assert_proche(rangs[0], 1.0 / 10.0, 1e-15, "reel superieur")
    _assert_proche(rangs[1], 1.0, 1e-15, "reel inferieur")


def test_corrections_multiples():
    p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])

    holm = correction_holm(p)
    bh = correction_benjamini_hochberg(p)

    for nom, ajuste in (("Holm", holm), ("BH", bh)):
        fini = np.isfinite(p)

        if np.any(ajuste[fini] < p[fini] - 1e-15):
            raise AssertionError(f"{nom} : valeur ajustee < brute.")

        if np.any(ajuste[fini] > 1.0):
            raise AssertionError(f"{nom} : valeur ajustee > 1.")

        if not np.isnan(ajuste[4]):
            raise AssertionError(f"{nom} : NaN non preserve.")

    _assert_proche(holm[0], 0.04, 1e-12, "Holm p=0.01")
    _assert_proche(bh[0], 0.04, 1e-12, "BH p=0.01")

    if np.any(bh[np.isfinite(bh)] > holm[np.isfinite(holm)] + 1e-12):
        raise AssertionError("BH doit etre inferieur ou egal a Holm.")


def test_permutation_scenes():
    rng = np.random.default_rng(8)

    X = rng.normal(size=(12, 10))
    ensemble = {"X_futurs": X}

    permutation = rng.permutation(12)

    resultat = permuter_ensemble_scenes(ensemble, permutation)

    Xp = resultat["X_futurs"]

    if not np.array_equal(
        Xp[:, :-NOMBRE_COORDONNEES_SCENE],
        X[:, :-NOMBRE_COORDONNEES_SCENE],
    ):
        raise AssertionError("Les colonnes hors scene ont change.")

    _assert_proche(
        float(np.max(np.abs(
            Xp[:, -NOMBRE_COORDONNEES_SCENE:].mean(axis=0)
            - X[:, -NOMBRE_COORDONNEES_SCENE:].mean(axis=0)
        ))),
        0.0,
        1e-12,
        "moyenne des resumes de scene",
    )

    try:
        permuter_ensemble_scenes(ensemble, np.zeros(12, dtype=int))
    except ValueError:
        pass
    else:
        raise AssertionError("Une fausse permutation doit etre refusee.")


def test_resume_scene():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    resume = resume_scene(np.zeros(4), points)

    if resume.shape != (NOMBRE_COORDONNEES_SCENE,):
        raise AssertionError("Forme du resume de scene incorrecte.")

    _assert_proche(resume[2], 0.5, 1e-12, "centre x de repli")
    _assert_proche(resume[3], 0.5, 1e-12, "centre y de repli")

    resume = resume_scene(np.array([1.0, 0.0, 0.0, 0.0]), points)

    _assert_proche(resume[2], 0.0, 1e-12, "centre x pondere")


def test_conversion_json():
    objet = {
        "a": np.float64(1.5),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": float("nan"),
        "e": np.array([1.0, np.inf]),
        "f": (1, 2),
        "g": Path("x"),
    }

    texte = json.dumps(convertir_json(objet), allow_nan=False)
    relu = json.loads(texte)

    if relu["c"] is not True:
        raise AssertionError("np.bool_ mal converti.")

    if relu["d"] is not None or relu["e"][1] is not None:
        raise AssertionError("Valeurs non finies mal converties.")


def test_exponentielle_sure():
    _assert_proche(exponentielle_sure(-np.inf), 0.0, 0.0, "exp(-inf)")

    try:
        exponentielle_sure(1e6)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Un depassement doit lever une erreur.")


def test_configuration():
    Configuration().valider()

    try:
        Configuration(nombre_branches=3, dimension_latente=6).valider()
    except ValueError:
        pass
    else:
        raise AssertionError("Configuration invalide acceptee.")

    try:
        Configuration(regularisation_covariance=0.0).valider()
    except ValueError:
        pass
    else:
        raise AssertionError("Regularisation nulle acceptee.")


def test_ecriture_json_atomique():
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "x.json"

        ecrire_json_atomique(chemin, {"a": np.array([1, 2])})

        with chemin.open(encoding="utf-8") as fichier:
            if json.load(fichier) != {"a": [1, 2]}:
                raise AssertionError("JSON relu incorrect.")

        try:
            ecrire_json_atomique(chemin, {"a": object()})
        except TypeError:
            pass
        else:
            raise AssertionError("Objet non serialisable accepte.")

        restes = [
            nom for nom in os.listdir(dossier) if nom.endswith(".tmp")
        ]

        if restes:
            raise AssertionError("Fichier temporaire non supprime.")


AUTOTESTS = (
    test_facteur_boule,
    test_cosinus,
    test_distances_euclidiennes,
    test_mds_reproduit_distances,
    test_metrique_sauvegarde_chargement,
    test_covariance_et_echelle,
    test_covariance_degeneree,
    test_bootstrap_blocs,
    test_bootstrap_reproductible,
    test_rang_permutation,
    test_corrections_multiples,
    test_permutation_scenes,
    test_resume_scene,
    test_conversion_json,
    test_exponentielle_sure,
    test_configuration,
    test_ecriture_json_atomique,
)


def executer_autotests():
    reussis = 0
    echecs = []

    for test in AUTOTESTS:
        try:
            test()
        except Exception as erreur:
            echecs.append((test.__name__, erreur))
            print(f"  ECHEC  {test.__name__} : {erreur}")
        else:
            reussis += 1
            print(f"  ok     {test.__name__}")

    print(
        f"\nAutotests : {reussis}/{len(AUTOTESTS)} reussis."
    )

    return len(echecs) == 0


# ---------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------

def echantillon_regulier(indices, nombre):
    """
    Choisit `nombre` elements regulierement espaces dans `indices`
    (arrondi, sans doublon tant que nombre <= len(indices)).
    """
    positions = np.linspace(
        0,
        len(indices) - 1,
        nombre,
    ).round().astype(int)

    return indices[positions]


def executer_analyse(configuration):
    debut = time.time()

    configuration.valider()

    dossier = configuration.dossier()
    dossier.mkdir(parents=True, exist_ok=True)

    configurer_journal(
        configuration.niveau_journal,
        dossier / "journal.log",
    )

    journal.info("=== Jonction futur passe / metrique integree ===")
    journal.info("Version du script : %s", VERSION_SCRIPT)
    journal.info("Dimension latente : %d", configuration.dimension_latente)
    journal.info("Branches : %d", configuration.nombre_branches)
    journal.info("Instants : %d", configuration.nombre_instants)
    journal.info(
        "Repetitions / permutations : %d",
        configuration.nombre_repetitions,
    )
    journal.info("Sortie : %s", dossier)

    # -- prototype ------------------------------------------------------

    interface = InterfacePrototype(
        configuration.module_prototype
    ).charger()

    journal.info("Simulation du prototype...")

    points, parametres, resultats = interface.executer()

    amplitude_bruit = interface.bruit_futur

    temps_complets = resultats["temps"]

    seuil_temporel = (
        configuration.fraction_entrainement * interface.duree
    )

    indices_train = np.flatnonzero(
        temps_complets <= seuil_temporel
    )

    indices_eval = np.flatnonzero(
        temps_complets > seuil_temporel
    )

    if len(indices_train) <= configuration.dimension_latente:
        raise ValueError(
            "Pas assez d'etats d'entrainement "
            f"({len(indices_train)} <= dimension latente)."
        )

    if len(indices_eval) < configuration.nombre_instants:
        raise ValueError(
            "Pas assez d'etats d'evaluation "
            f"({len(indices_eval)} < {configuration.nombre_instants})."
        )

    # -- metrique -------------------------------------------------------

    nombre_train = min(
        configuration.max_etats_entrainement,
        len(indices_train),
    )

    echantillon_train = echantillon_regulier(
        indices_train,
        nombre_train,
    )

    X_train = np.asarray([
        construire_etat_present(
            resultats,
            index,
            points,
        )["X_present"]
        for index in echantillon_train
    ])

    journal.info(
        "Apprentissage MDS : %d etats, %d variables...",
        len(X_train),
        X_train.shape[1],
    )

    if configuration.cible_standardisee:
        moyenne = X_train.mean(axis=0)
        ecarts = X_train.std(axis=0)
        ecarts = np.where(ecarts > 1e-12, ecarts, 1.0)

        D_train = distances_euclidiennes(
            (X_train - moyenne) / ecarts
        )

        journal.info("Cible MDS : distances sur X standardise.")
    else:
        D_train = distances_euclidiennes(X_train)

        journal.info("Cible MDS : distances sur X brut.")

    metrique = SubjectiveMetric(
        k=configuration.dimension_latente,
        reg=configuration.regularisation_metrique,
        standardize=True,
    )

    metrique.fit(
        X_train,
        D_train,
        select_k=False,
        verbose=True,
    )

    diagnostics_metrique = metrique.diagnostics(
        X_train,
        D_train,
    )

    journal.info(
        "Qualite metrique : correlation=%.4f, stress=%.4f, "
        "part MDS expliquee=%.4f",
        diagnostics_metrique["correlation_distances"],
        diagnostics_metrique["stress_kruskal"],
        diagnostics_metrique["part_variance_mds_expliquee"],
    )

    metrique.sauvegarder(dossier / "metrique.npz")

    contexte_constant = calculer_contexte_constant(
        resultats,
        indices_train,
    )

    # -- boucle principale -----------------------------------------------

    indices_analyse = echantillon_regulier(
        indices_eval,
        configuration.nombre_instants,
    )

    temps = temps_complets[indices_analyse]

    nombre_instants = configuration.nombre_instants
    nombre_repetitions = configuration.nombre_repetitions

    reel = creer_stockage((nombre_instants,))

    forme_repetitions = (
        nombre_repetitions,
        nombre_instants,
    )

    reel_rep = creer_stockage(forme_repetitions)
    constant_rep = creer_stockage(forme_repetitions)
    permute = creer_stockage(forme_repetitions)

    presents_latents = []
    centres_futurs = []
    branches_latentes = []
    spectres_covariance = []
    rangs_effectifs = []

    ecart_etat_present_max = 0.0
    ecart_equivalence_contexte_max = 0.0

    reg_cov = configuration.regularisation_covariance

    for numero, index in enumerate(indices_analyse):
        instant_debut = time.time()

        etat = construire_etat_present(
            resultats,
            index,
            points,
        )

        sauvegarde_etat = {
            cle: valeur.copy()
            for cle, valeur in etat.items()
        }

        forme_bruit = (
            configuration.nombre_branches,
            len(etat["H"]),
        )

        rng_base = np.random.default_rng(
            np.random.SeedSequence([
                configuration.graine_ensemble,
                numero,
                0,
            ])
        )

        bruits_base = rng_base.normal(
            size=forme_bruit
        )

        try:
            ensemble_base = generer_ensemble_futur(
                etat,
                parametres,
                points,
                bruits_base,
                amplitude_bruit,
            )

            mesure_base = analyser_volume_futur(
                ensemble_base,
                metrique,
                reg_cov,
            )

        except Exception as erreur:
            raise RuntimeError(
                f"Echec sur l'ensemble de base, instant {numero} "
                f"(indice {int(index)})."
            ) from erreur

        enregistrer_mesure(
            reel,
            numero,
            mesure_base,
        )

        presents_latents.append(mesure_base["Z_present"])
        centres_futurs.append(mesure_base["centre_futur"])
        branches_latentes.append(mesure_base["Z_futurs"])
        rangs_effectifs.append(mesure_base["rang_effectif"])

        spectres_covariance.append(
            mesure_base["valeurs_propres_covariance"]
        )

        if numero == 0:
            ecart_equivalence_contexte_max = (
                controle_equivalence_contexte(
                    etat,
                    parametres,
                    points,
                    bruits_base,
                    amplitude_bruit,
                    ensemble_base,
                )
            )

        for repetition in range(nombre_repetitions):
            try:
                if repetition == 0:
                    bruits = bruits_base
                    mesure_actuelle = mesure_base

                else:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([
                            configuration.graine_ensemble,
                            numero,
                            repetition,
                        ])
                    )

                    bruits = rng.normal(size=forme_bruit)

                    ensemble_actuel = generer_ensemble_futur(
                        etat,
                        parametres,
                        points,
                        bruits,
                        amplitude_bruit,
                    )

                    mesure_actuelle = analyser_volume_futur(
                        ensemble_actuel,
                        metrique,
                        reg_cov,
                    )

                ensemble_constant = generer_ensemble_futur(
                    etat,
                    parametres,
                    points,
                    bruits,
                    amplitude_bruit,
                    contexte_futur=contexte_constant,
                )

                mesure_constante = analyser_volume_futur(
                    ensemble_constant,
                    metrique,
                    reg_cov,
                )

                rng_permutation = np.random.default_rng(
                    np.random.SeedSequence([
                        configuration.graine_permutations,
                        numero,
                        repetition,
                    ])
                )

                permutation = rng_permutation.permutation(
                    configuration.nombre_branches
                )

                ensemble_permute = permuter_ensemble_scenes(
                    ensemble_base,
                    permutation,
                )

                mesure_permutee = analyser_volume_futur(
                    ensemble_permute,
                    metrique,
                    reg_cov,
                )

            except Exception as erreur:
                raise RuntimeError(
                    f"Echec a l'instant {numero}, "
                    f"repetition {repetition}."
                ) from erreur

            enregistrer_mesure(
                reel_rep,
                (repetition, numero),
                mesure_actuelle,
            )

            enregistrer_mesure(
                constant_rep,
                (repetition, numero),
                mesure_constante,
            )

            enregistrer_mesure(
                permute,
                (repetition, numero),
                mesure_permutee,
            )

        for cle in etat:
            ecart = float(
                np.max(
                    np.abs(
                        etat[cle]
                        - sauvegarde_etat[cle]
                    )
                )
            )

            ecart_etat_present_max = max(
                ecart_etat_present_max,
                ecart,
            )

        if ecart_etat_present_max != 0.0:
            raise RuntimeError(
                "Une generation a modifie "
                "l'etat present fourni."
            )

        duree_instant = time.time() - instant_debut
        restants = nombre_instants - numero - 1

        journal.info(
            "Instant %d/%d | t=%.2f | A=%.6f | D=%.6e | logV=%.6f "
            "| %.1fs (reste ~%.0fs)",
            numero + 1,
            nombre_instants,
            temps[numero],
            mesure_base["ancrage"],
            mesure_base["distance"],
            mesure_base["log_volume_futur"],
            duree_instant,
            restants * duree_instant,
        )

    # -- controles --------------------------------------------------------

    controles = {
        "ecart_equivalence_contexte_identique": (
            ecart_equivalence_contexte_max
        ),
        "ecart_etat_present_max": ecart_etat_present_max,
    }

    if ecart_equivalence_contexte_max != 0.0:
        journal.warning(
            "Equivalence de contexte non exacte : ecart=%.3e",
            ecart_equivalence_contexte_max,
        )

    if configuration.verifier_determinisme:
        mesure_rejouee = mesure_base_instant(
            0,
            indices_analyse[0],
            resultats,
            points,
            parametres,
            metrique,
            configuration,
            amplitude_bruit,
        )

        valeurs_enregistrees = {
            champ: reel[champ][0]
            for champ in CHAMPS_MESURES
        }

        ecarts = controle_determinisme(
            mesure_rejouee,
            valeurs_enregistrees,
        )

        controles["determinisme_ecarts_max_par_champ"] = ecarts

        pire = max(ecarts.values())

        journal.info("Determinisme (instant 0) : ecart max = %.3e", pire)

        if pire > 1e-9:
            journal.warning(
                "Le recalcul de l'instant 0 differe de l'original."
            )

    # -- statistiques et rapport -----------------------------------------

    statistiques = construire_statistiques(
        reel_rep,
        constant_rep,
        reel,
        permute,
        configuration,
    )

    rapport = {
        "statut": (
            "Analyse exploratoire "
            "d'un modele synthetique"
        ),

        "version_script": VERSION_SCRIPT,

        "parametres": {
            **configuration.en_dictionnaire(),
            "nombre_etats_entrainement": len(X_train),
            "seuil_entrainement_evaluation": seuil_temporel,
            "bruit_futur": amplitude_bruit,
            "dimension_H": interface.dimension_H,
            "dimension_C": interface.dimension_C,
            "dimension_M": interface.dimension_M,
            "dimension_X": int(X_train.shape[1]),
            "N": interface.N,
            "duree": interface.duree,
        },

        "metrique": {
            "cible": (
                "distances euclidiennes sur X standardise"
                if configuration.cible_standardisee
                else "distances euclidiennes sur X brut"
            ),
            "valeurs_propres_mds": (
                metrique.valeurs_propres_
            ),
            "origine_cosinus": (
                "origine du repere MDS appris"
            ),
            "diagnostics": diagnostics_metrique,
        },

        "temps": temps,
        "indices_analyse": indices_analyse,
        "contexte_constant": contexte_constant,

        "resume_descriptif": {
            "reel_ensemble_reference": resume_descriptif(
                reel,
                moyenner_repetitions=False,
            ),
            "reel_repetitions": resume_descriptif(
                reel_rep,
                moyenner_repetitions=True,
            ),
            "contexte_constant": resume_descriptif(
                constant_rep,
                moyenner_repetitions=True,
            ),
            "scenes_permutees": resume_descriptif(
                permute,
                moyenner_repetitions=True,
            ),
        },

        "rang_effectif_moyen": float(
            np.mean(rangs_effectifs)
        ),

        "reel_ensemble_reference": reel,

        "reel_repetitions": reel_rep,

        "contexte_constant_repetitions_appariees": (
            constant_rep
        ),

        "scenes_permutees": permute,

        "statistiques": statistiques,

        "controles": controles,

        "invariance_etat_present": {
            "ecart_max": ecart_etat_present_max,
            "portee": (
                "Verification locale d'absence "
                "de mutation des tableaux fournis "
                "au generateur. Ce n'est pas une "
                "experience physique de non-signalisation."
            ),
        },

        "limites": [
            "Une seule realisation des poids du reseau.",
            (
                "Les branches ne sont pas "
                "des observations independantes."
            ),
            (
                "Les instants de la trajectoire "
                "sont autocorreles."
            ),
            (
                "Les intervalles par blocs "
                "sont descriptifs et conditionnels."
            ),
            (
                "Les longueurs de blocs sont "
                "une grille de sensibilite."
            ),
            (
                "L'ancrage cosinus depend "
                "de l'origine du repere latent."
            ),
            (
                "Le volume regularise est positif "
                "meme pour un nuage degenere."
            ),
            (
                "Le coefficient angulaire "
                "d'attachement ne penalise "
                "pas la distance."
            ),
            (
                "Le contexte et les activites globales "
                "sont figes dans les branches."
            ),
            (
                "La cible euclidienne n'est pas "
                "une metrique subjective empirique."
            ),
            (
                "Les corrections de Holm et de "
                "Benjamini-Hochberg supposent des tests "
                "exploitables ; les instants autocorreles "
                "en limitent la portee."
            ),
            (
                "Les controles supplementaires "
                "temporels et d'ablation "
                "ne sont pas inclus."
            ),
            (
                "Aucune inference clinique, "
                "biologique validee ou de conscience."
            ),
        ],
    }

    chemin_rapport = dossier / "rapport.json"

    ecrire_json_atomique(chemin_rapport, rapport)

    chemin_csv = exporter_csv(
        dossier,
        temps,
        reel,
        reel_rep,
        constant_rep,
        permute,
    )

    chemin_npz = exporter_npz(
        dossier,
        configuration,
        temps,
        indices_analyse,
        metrique,
        presents_latents,
        centres_futurs,
        branches_latentes,
        spectres_covariance,
        rangs_effectifs,
        reel,
        reel_rep,
        constant_rep,
        permute,
    )

    chemins_figures = [
        tracer_mesures(
            dossier,
            temps,
            reel_rep,
            constant_rep,
            permute,
        )
    ]

    if configuration.graphiques_supplementaires:
        chemins_figures.append(
            tracer_contrastes(
                dossier,
                statistiques,
                configuration,
            )
        )

        chemins_figures.append(
            tracer_spectres(
                dossier,
                spectres_covariance,
            )
        )

        chemins_figures.append(
            tracer_projection_latente(
                dossier,
                temps,
                presents_latents,
                branches_latentes,
            )
        )

        chemins_figures.append(
            tracer_rangs_permutation(
                dossier,
                statistiques,
                configuration,
            )
        )

    # -- manifeste --------------------------------------------------------

    sorties = [
        chemin_rapport,
        chemin_csv,
        chemin_npz,
        dossier / "metrique.npz",
        *chemins_figures,
    ]

    manifeste = {
        "version_script": VERSION_SCRIPT,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "duree_secondes": time.time() - debut,
        "ligne_de_commande": list(sys.argv),
        "python": sys.version,
        "plateforme": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "configuration": configuration.en_dictionnaire(),
        "prototype": {
            "module": configuration.module_prototype,
            "fichier": interface.chemin_fichier,
            "sha256": (
                empreinte_fichier(interface.chemin_fichier)
                if interface.chemin_fichier
                and interface.chemin_fichier.exists()
                else None
            ),
        },
        "sorties": {
            chemin.name: {
                "octets": chemin.stat().st_size,
                "sha256": empreinte_fichier(chemin),
            }
            for chemin in sorties
        },
        "note": (
            "Le journal.log n'est pas liste : il est encore ouvert "
            "au moment de l'ecriture du manifeste."
        ),
    }

    ecrire_json_atomique(dossier / "manifeste.json", manifeste)

    # -- resume console ----------------------------------------------------

    journal.info("=== Contrastes moyens ===")

    for nom, entree in statistiques.items():
        if (
            isinstance(entree, dict)
            and "moyenne" in entree
        ):
            moyenne = entree["moyenne"]

            if moyenne is None:
                journal.info("%s : non fini", nom)
            else:
                journal.info("%s : %.6e", nom, moyenne)

            if "ratio_geometrique" in entree:
                journal.info(
                    "  Ratio geometrique : %.6f",
                    entree["ratio_geometrique"],
                )

    for champ, donnees in statistiques["permutation_volumes"].items():
        journal.info(
            "Permutation %s : %d/%d sous alpha=%.2f (brut), "
            "%d (Holm), %d (BH)",
            champ,
            donnees["nombre_sous_alpha_brut"],
            nombre_instants,
            donnees["alpha"],
            donnees["nombre_sous_alpha_holm"],
            donnees["nombre_sous_alpha_bh"],
        )

    invariance = statistiques[
        "invariance_centre_scenes_permutees"
    ]

    journal.info(
        "Ecart maximal d'ancrage sous permutation : %.6e",
        invariance["ecart_max_ancrage"],
    )

    journal.info(
        "Ecart maximal de distance sous permutation : %.6e",
        invariance["ecart_max_distance"],
    )

    journal.info("Resultats : %s", dossier)

    journal.info(
        "Analyse exploratoire uniquement. "
        "Aucune interpretation clinique ou de conscience."
    )

    return rapport


# ---------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------

def main(argv=None):
    configuration, arguments = analyser_arguments(argv)

    if arguments.autotest:
        configurer_journal("WARNING")

        print("=== Autotests numeriques ===")

        return 0 if executer_autotests() else 1

    try:
        configuration.valider()
    except ValueError as erreur:
        print(erreur, file=sys.stderr)
        return 2

    try:
        executer_analyse(configuration)
    except SystemExit:
        raise
    except Exception:
        journal.error("Echec de l'analyse :\n%s", traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
