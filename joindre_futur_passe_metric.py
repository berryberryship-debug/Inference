#!/usr/bin/env python3
"""
Analyse geometrique des futurs simules.

Fichier :
    joindre_futur_passe_metric.py

Dependance locale :
    prototype_futur_passe.py

Dependances externes :
    numpy
    matplotlib

Le programme contient :
    - une metrique MDS avec projection lineaire regularisee ;
    - une representation multimodale present/futur ;
    - des ensembles de branches futures ;
    - un controle de contexte constant avec bruits apparies ;
    - des permutations des scenes sans regeneration des branches ;
    - des volumes de covariance et leurs logarithmes ;
    - un ancrage present/futur et une distance present/futur ;
    - des contrastes apparies ;
    - un bootstrap temporel par blocs ;
    - des distributions de permutation pour les volumes ;
    - un rapport JSON, un tableau CSV, des donnees NPZ et un graphique.

Limites :
    - modele synthetique non calibre ;
    - dissimilarites euclidiennes, pas de jugements humains ;
    - contexte et activites globales figes dans les branches futures ;
    - les branches sont des propositions latentes, pas des trajectoires
      biophysiques completes a un horizon temporel calibre ;
    - aucune interpretation clinique, de conscience ou de retrocausalite.
"""

from pathlib import Path
from math import lgamma, log, pi
import csv
import inspect
import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import prototype_futur_passe as prototype


# ---------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------

DOSSIER_SCRIPT = Path(__file__).resolve().parent

DOSSIER_SORTIE = (
    DOSSIER_SCRIPT
    / "resultats"
    / "jonction_futur_passe_metric"
)

DIMENSION_LATENTE = 6
NOMBRE_BRANCHES = 64
NOMBRE_INSTANTS = 40
NOMBRE_PERMUTATIONS = 100

MAX_ETATS_ENTRAINEMENT = 120

REGULARISATION_METRIQUE = 1e-2
REGULARISATION_COVARIANCE = 1e-8

LONGUEURS_BLOCS = [4, 8, 12]
NOMBRE_BOOTSTRAP = 2000

GRAINE_ENSEMBLE = 9000
GRAINE_PERMUTATIONS = 20000
GRAINE_BOOTSTRAP = 1234

BRUIT_FUTUR = float(
    getattr(prototype, "BRUIT_FUTUR", 0.025)
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
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X doit etre une matrice."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contient des valeurs non finies."
        )

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
        self.erreur_projection_ = None

    def _standardiser(
        self,
        X,
        apprentissage=False,
    ):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X doit etre une matrice."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contient des valeurs non finies."
            )

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
                "Dimension d'entree incompatible."
            )

        return (
            X - self.moyenne_
        ) / self.ecart_type_

    def fit(
        self,
        X,
        D,
        select_k=False,
        verbose=False,
    ):
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X doit avoir deux dimensions."
            )

        n = len(X)

        if D.shape != (n, n):
            raise ValueError(
                "D doit avoir la forme (n, n)."
            )

        if not np.all(np.isfinite(D)):
            raise ValueError(
                "D contient des valeurs non finies."
            )

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
            print(
                f"Metrique : {n} etats, "
                f"{X.shape[1]} variables, "
                f"k={self.k}",
                flush=True,
            )

        return self

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


# ---------------------------------------------------------------------
# Interface avec le prototype existant
# ---------------------------------------------------------------------

def verifier_prototype():
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
        if not hasattr(prototype, nom)
    ]

    if absents:
        raise RuntimeError(
            "Elements absents de prototype_futur_passe.py : "
            + ", ".join(absents)
        )


def executer_prototype():
    """
    Accepte :
        simuler(rng, points, parametres)

    ou :
        simuler(
            rng,
            points,
            parametres,
            graine_futur=...
        )
    """
    rng = np.random.default_rng(
        int(prototype.SEED)
    )

    points = rng.uniform(
        0.0,
        1.0,
        size=(int(prototype.N), 2),
    )

    parametres = prototype.initialiser_parametres(
        rng,
        points,
    )

    parametres_necessaires = (
        "W_MEM_G",
        "W_MEM_C",
        "W_MEM_H",
        "W_HH",
        "W_HC",
        "W_HG",
        "decodeur_scene",
    )

    for nom in parametres_necessaires:
        if nom not in parametres:
            raise KeyError(
                f"Parametre absent du prototype : {nom}"
            )

    signature = inspect.signature(
        prototype.simuler
    )

    arguments = {}

    if "graine_futur" in signature.parameters:
        arguments["graine_futur"] = 1

    resultats = prototype.simuler(
        rng,
        points,
        parametres,
        **arguments,
    )

    champs = (
        "temps",
        "E",
        "I",
        "C",
        "H",
        "scene",
        "M_passe",
    )

    for champ in champs:
        if champ not in resultats:
            raise KeyError(
                f"Champ absent des resultats : {champ}"
            )

        tableau = np.asarray(
            resultats[champ],
            dtype=float,
        )

        if not np.all(np.isfinite(tableau)):
            raise FloatingPointError(
                f"Valeurs non finies dans {champ}."
            )

    temps = np.asarray(
        resultats["temps"],
        dtype=float,
    )

    if temps.ndim != 1 or len(temps) < 2:
        raise ValueError(
            "Historique temporel invalide."
        )

    if np.any(np.diff(temps) <= 0.0):
        raise ValueError(
            "Les temps doivent etre strictement croissants."
        )

    for champ in champs[1:]:
        if len(resultats[champ]) != len(temps):
            raise ValueError(
                f"Longueur temporelle incoherente : {champ}"
            )

    return points, parametres, resultats


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
        + BRUIT_FUTUR * bruit
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
    E = np.asarray(
        resultats["E"][index],
        dtype=float,
    )

    I = np.asarray(
        resultats["I"][index],
        dtype=float,
    )

    C = np.asarray(
        resultats["C"][index],
        dtype=float,
    )

    H = np.asarray(
        resultats["H"][index],
        dtype=float,
    )

    M = np.asarray(
        resultats["M_passe"][index],
        dtype=float,
    )

    scene = np.asarray(
        resultats["scene"][index],
        dtype=float,
    )

    global_EI = np.array([
        E.mean(),
        I.mean(),
    ])

    return {
        "H": H,
        "C": C,
        "M_passe": M,
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
    return np.asarray(
        resultats["C"][indices],
        dtype=float,
    ).mean(axis=0)


# ---------------------------------------------------------------------
# Ensembles futurs et permutations
# ---------------------------------------------------------------------

def generer_ensemble_futur(
    etat,
    parametres,
    points,
    bruits,
    contexte_futur=None,
):
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
    resultat = dict(ensemble)

    X = ensemble["X_futurs"].copy()

    X[:, -4:] = (
        ensemble["X_futurs"][permutation, -4:]
    )

    resultat["X_futurs"] = X

    resultat["permutation_scenes"] = (
        permutation.copy()
    )

    return resultat


# ---------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------

def statistiques_covariance(Z):
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError(
            "Z doit etre une matrice."
        )

    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "Z contient des valeurs non finies."
        )

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
        + REGULARISATION_COVARIANCE
        * np.eye(k)
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
        "valeurs_propres": valeurs,
    }


def volume_covariance_latente(Z):
    return statistiques_covariance(Z)["volume"]


def analyser_volume_futur(
    ensemble,
    metrique,
):
    Zp = metrique.transform(
        ensemble["X_present"][None, :]
    )[0]

    Zf = metrique.transform(
        ensemble["X_futurs"]
    )

    centre = Zf.mean(axis=0)

    geometrie = statistiques_covariance(
        Zf
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
        ])
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

    rng = np.random.default_rng(
        GRAINE_BOOTSTRAP
    )

    estimations = np.empty(
        NOMBRE_BOOTSTRAP
    )

    for repetition in range(NOMBRE_BOOTSTRAP):
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


def construire_statistiques(
    reel_rep,
    constant_rep,
    reel,
    permute,
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
            "bootstrap_blocs": {},
        }

        if nom.startswith("log_volume") and fini:
            entree["ratio_geometrique"] = (
                exponentielle_sure(
                    float(serie.mean())
                )
            )

        for longueur in LONGUEURS_BLOCS:
            if 1 <= longueur < len(serie):
                entree["bootstrap_blocs"][
                    str(longueur)
                ] = bootstrap_moyenne_contraste(
                    serie,
                    longueur,
                )

        rapport[nom] = entree

    rapport["permutation_volumes"] = {}

    for champ in (
        "log_volume_futur",
        "log_volume_attache",
    ):
        rangs = rang_permutation_superieur(
            reel[champ],
            permute[champ],
        )

        rapport["permutation_volumes"][champ] = {
            "rangs_unilateraux_par_instant": rangs,
            "interpretation": (
                "Distribution de reassignation des scenes. "
                "Pas une validation biologique ni une inference "
                "generale sur plusieurs reseaux. "
                "Les rangs ponctuels ne sont pas corriges "
                "pour les comparaisons multiples."
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

    if isinstance(objet, (np.integer, int)):
        return int(objet)

    if isinstance(objet, (np.floating, float)):
        valeur = float(objet)

        if np.isfinite(valeur):
            return valeur

        return None

    return objet


def exporter_csv(
    temps,
    reel,
    reel_rep,
    constant_rep,
    permute,
):
    chemin = (
        DOSSIER_SORTIE
        / "mesures_futures.csv"
    )

    conditions = {
        "reel": reel,

        "reel_repete_moyen": {
            cle: valeur.mean(axis=0)
            for cle, valeur in reel_rep.items()
        },

        "contexte_constant_moyen": {
            cle: valeur.mean(axis=0)
            for cle, valeur in constant_rep.items()
        },

        "scenes_permutees_moyen": {
            cle: valeur.mean(axis=0)
            for cle, valeur in permute.items()
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


# ---------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------

def tracer_mesures(
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
        "— modele synthetique"
    )

    fig.tight_layout()

    fig.savefig(
        DOSSIER_SORTIE / "mesures_futures.png",
        dpi=150,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------

def main():
    verifier_prototype()

    if NOMBRE_BRANCHES <= DIMENSION_LATENTE:
        raise ValueError(
            "NOMBRE_BRANCHES doit depasser "
            "DIMENSION_LATENTE."
        )

    if NOMBRE_PERMUTATIONS < 1:
        raise ValueError(
            "NOMBRE_PERMUTATIONS doit etre positif."
        )

    if MAX_ETATS_ENTRAINEMENT <= DIMENSION_LATENTE:
        raise ValueError(
            "MAX_ETATS_ENTRAINEMENT doit depasser "
            "DIMENSION_LATENTE."
        )

    seuil_temporel = (
        0.5 * float(prototype.DUREE)
    )

    DOSSIER_SORTIE.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Jonction futur passe / metrique integree ===",
        flush=True,
    )

    print(
        f"Dimension latente : {DIMENSION_LATENTE}",
        flush=True,
    )

    print(
        f"Branches : {NOMBRE_BRANCHES}",
        flush=True,
    )

    print(
        f"Instants : {NOMBRE_INSTANTS}",
        flush=True,
    )

    print(
        "Repetitions / permutations : "
        f"{NOMBRE_PERMUTATIONS}",
        flush=True,
    )

    print(
        "Simulation du prototype...",
        flush=True,
    )

    points, parametres, resultats = (
        executer_prototype()
    )

    temps_complets = np.asarray(
        resultats["temps"],
        dtype=float,
    )

    indices_train = np.flatnonzero(
        temps_complets <= seuil_temporel
    )

    indices_eval = np.flatnonzero(
        temps_complets > seuil_temporel
    )

    if len(indices_train) <= DIMENSION_LATENTE:
        raise ValueError(
            "Pas assez d'etats d'entrainement."
        )

    if len(indices_eval) < NOMBRE_INSTANTS:
        raise ValueError(
            "Pas assez d'etats d'evaluation."
        )

    nombre_train = min(
        MAX_ETATS_ENTRAINEMENT,
        len(indices_train),
    )

    positions_train = np.linspace(
        0,
        len(indices_train) - 1,
        nombre_train,
        dtype=int,
    )

    echantillon_train = (
        indices_train[positions_train]
    )

    X_train = np.asarray([
        construire_etat_present(
            resultats,
            index,
            points,
        )["X_present"]
        for index in echantillon_train
    ])

    print(
        "Apprentissage MDS : "
        f"{len(X_train)} etats, "
        f"{X_train.shape[1]} variables...",
        flush=True,
    )

    D_train = distances_euclidiennes(
        X_train
    )

    metrique = SubjectiveMetric(
        k=DIMENSION_LATENTE,
        reg=REGULARISATION_METRIQUE,
        standardize=True,
    )

    metrique.fit(
        X_train,
        D_train,
        select_k=False,
        verbose=True,
    )

    contexte_constant = (
        calculer_contexte_constant(
            resultats,
            indices_train,
        )
    )

    positions_eval = np.linspace(
        0,
        len(indices_eval) - 1,
        NOMBRE_INSTANTS,
        dtype=int,
    )

    indices_analyse = (
        indices_eval[positions_eval]
    )

    temps = (
        temps_complets[indices_analyse]
    )

    reel = creer_stockage(
        (NOMBRE_INSTANTS,)
    )

    forme_repetitions = (
        NOMBRE_PERMUTATIONS,
        NOMBRE_INSTANTS,
    )

    reel_rep = creer_stockage(
        forme_repetitions
    )

    constant_rep = creer_stockage(
        forme_repetitions
    )

    permute = creer_stockage(
        forme_repetitions
    )

    presents_latents = []
    centres_futurs = []
    branches_latentes = []
    spectres_covariance = []

    ecart_etat_present_max = 0.0

    for numero, index in enumerate(
        indices_analyse
    ):
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
            NOMBRE_BRANCHES,
            len(etat["H"]),
        )

        rng_base = np.random.default_rng(
            np.random.SeedSequence([
                GRAINE_ENSEMBLE,
                numero,
                0,
            ])
        )

        bruits_base = rng_base.normal(
            size=forme_bruit
        )

        ensemble_base = (
            generer_ensemble_futur(
                etat,
                parametres,
                points,
                bruits_base,
            )
        )

        mesure_base = analyser_volume_futur(
            ensemble_base,
            metrique,
        )

        enregistrer_mesure(
            reel,
            numero,
            mesure_base,
        )

        presents_latents.append(
            mesure_base["Z_present"]
        )

        centres_futurs.append(
            mesure_base["centre_futur"]
        )

        branches_latentes.append(
            mesure_base["Z_futurs"]
        )

        spectres_covariance.append(
            mesure_base[
                "valeurs_propres_covariance"
            ]
        )

        for repetition in range(
            NOMBRE_PERMUTATIONS
        ):
            if repetition == 0:
                bruits = bruits_base
                mesure_actuelle = mesure_base

            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([
                        GRAINE_ENSEMBLE,
                        numero,
                        repetition,
                    ])
                )

                bruits = rng.normal(
                    size=forme_bruit
                )

                ensemble_actuel = (
                    generer_ensemble_futur(
                        etat,
                        parametres,
                        points,
                        bruits,
                    )
                )

                mesure_actuelle = (
                    analyser_volume_futur(
                        ensemble_actuel,
                        metrique,
                    )
                )

            ensemble_constant = (
                generer_ensemble_futur(
                    etat,
                    parametres,
                    points,
                    bruits,
                    contexte_futur=contexte_constant,
                )
            )

            mesure_constante = (
                analyser_volume_futur(
                    ensemble_constant,
                    metrique,
                )
            )

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

            rng_permutation = (
                np.random.default_rng(
                    np.random.SeedSequence([
                        GRAINE_PERMUTATIONS,
                        numero,
                        repetition,
                    ])
                )
            )

            permutation = (
                rng_permutation.permutation(
                    NOMBRE_BRANCHES
                )
            )

            ensemble_permute = (
                permuter_ensemble_scenes(
                    ensemble_base,
                    permutation,
                )
            )

            mesure_permutee = (
                analyser_volume_futur(
                    ensemble_permute,
                    metrique,
                )
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

        print(
            f"Instant {numero + 1}/{NOMBRE_INSTANTS} | "
            f"t={temps[numero]:.2f} | "
            f"A={mesure_base['ancrage']:.6f} | "
            f"D={mesure_base['distance']:.6e} | "
            f"logV={mesure_base['log_volume_futur']:.6f}",
            flush=True,
        )

    statistiques = construire_statistiques(
        reel_rep,
        constant_rep,
        reel,
        permute,
    )

    rapport = {
        "statut": (
            "Analyse exploratoire "
            "d'un modele synthetique"
        ),

        "parametres": {
            "dimension_latente": DIMENSION_LATENTE,
            "nombre_branches": NOMBRE_BRANCHES,
            "nombre_instants": NOMBRE_INSTANTS,
            "nombre_repetitions": NOMBRE_PERMUTATIONS,
            "nombre_etats_entrainement": len(X_train),
            "seuil_entrainement_evaluation": seuil_temporel,
            "regularisation_covariance": (
                REGULARISATION_COVARIANCE
            ),
            "regularisation_metrique": (
                REGULARISATION_METRIQUE
            ),
            "bruit_futur": BRUIT_FUTUR,
            "longueurs_blocs": LONGUEURS_BLOCS,
            "nombre_bootstrap": NOMBRE_BOOTSTRAP,
        },

        "metrique": {
            "cible": (
                "distances euclidiennes "
                "des etats d'entrainement"
            ),
            "valeurs_propres_mds": (
                metrique.valeurs_propres_
            ),
            "erreur_projection_rms": (
                metrique.erreur_projection_
            ),
            "origine_cosinus": (
                "origine du repere MDS appris"
            ),
        },

        "temps": temps,
        "indices_analyse": indices_analyse,
        "contexte_constant": contexte_constant,

        "reel_ensemble_reference": reel,

        "reel_repetitions": reel_rep,

        "contexte_constant_repetitions_appariees": (
            constant_rep
        ),

        "scenes_permutees": permute,

        "statistiques": statistiques,

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

    with (
        DOSSIER_SORTIE / "rapport.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fichier:
        json.dump(
            convertir_json(rapport),
            fichier,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    exporter_csv(
        temps,
        reel,
        reel_rep,
        constant_rep,
        permute,
    )

    donnees_npz = {
        "temps": temps,
        "indices_analyse": indices_analyse,

        "presents_latents": np.asarray(
            presents_latents
        ),

        "centres_futurs": np.asarray(
            centres_futurs
        ),

        "branches_latentes": np.asarray(
            branches_latentes
        ),

        "spectres_covariance": np.asarray(
            spectres_covariance
        ),

        "metrique_moyenne": (
            metrique.moyenne_
        ),

        "metrique_ecart_type": (
            metrique.ecart_type_
        ),

        "metrique_projection": (
            metrique.projection_
        ),
    }

    for nom, stockage in (
        ("reel", reel),
        ("reel_repete", reel_rep),
        ("constant", constant_rep),
        ("permute", permute),
    ):
        for champ, valeurs in stockage.items():
            donnees_npz[
                f"{nom}_{champ}"
            ] = valeurs

    np.savez_compressed(
        DOSSIER_SORTIE / "ensembles_futurs.npz",
        **donnees_npz,
    )

    tracer_mesures(
        temps,
        reel_rep,
        constant_rep,
        permute,
    )

    print(
        "\n=== Contrastes moyens ===",
        flush=True,
    )

    for nom, entree in statistiques.items():
        if (
            isinstance(entree, dict)
            and "moyenne" in entree
        ):
            moyenne = entree["moyenne"]

            if moyenne is None:
                print(
                    f"{nom} : non fini",
                    flush=True,
                )
            else:
                print(
                    f"{nom} : {moyenne:.6e}",
                    flush=True,
                )

            if "ratio_geometrique" in entree:
                print(
                    "  Ratio geometrique : "
                    f"{entree['ratio_geometrique']:.6f}",
                    flush=True,
                )

    invariance = statistiques[
        "invariance_centre_scenes_permutees"
    ]

    print(
        "\nEcart maximal d'ancrage sous permutation : "
        f"{invariance['ecart_max_ancrage']:.6e}",
        flush=True,
    )

    print(
        "Ecart maximal de distance sous permutation : "
        f"{invariance['ecart_max_distance']:.6e}",
        flush=True,
    )

    print(
        f"\nResultats : {DOSSIER_SORTIE}",
        flush=True,
    )

    print(
        "Analyse exploratoire uniquement. "
        "Aucune interpretation clinique ou de conscience.",
        flush=True,
    )


if __name__ == "__main__":
    main()
