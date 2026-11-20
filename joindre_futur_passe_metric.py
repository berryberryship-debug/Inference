#!/usr/bin/env bash
set -euo pipefail

cd ~

# ---------------------------------------------------------------------
# 1. Creation / mise a jour de metrique_subjective.py
# ---------------------------------------------------------------------

cat > metrique_subjective.py <<'PYTHON'
#!/usr/bin/env python3
"""
Metrique subjective technique.

Cette implementation utilise :
    - une standardisation optionnelle des variables ;
    - un positionnement multidimensionnel classique (MDS) ;
    - une projection lineaire regularisee permettant de transformer
      de nouveaux etats dans le meme espace latent.

La matrice D fournie a fit() est une matrice de dissimilarites.
Ici, elle est construite a partir de distances euclidiennes, mais
elle pourrait ensuite provenir d'annotations humaines ou de
donnees comportementales.

Ce module ne mesure pas la conscience ni une experience subjective.
Il fournit uniquement une reduction de dimension technique.
"""

import numpy as np


def ball_volume_factor(dimension):
    from math import gamma, pi

    dimension = int(dimension)

    if dimension < 1:
        raise ValueError(
            "La dimension doit etre un entier positif."
        )

    return float(
        pi ** (dimension / 2.0)
        / gamma(dimension / 2.0 + 1.0)
    )


class SubjectiveMetric:
    def __init__(
        self,
        k=6,
        reg=1e-2,
        standardize=True,
    ):
        self.k = int(k)

        if self.k < 1:
            raise ValueError(
                "k doit etre un entier positif."
            )

        self.reg = float(reg)

        if self.reg <= 0.0:
            raise ValueError(
                "reg doit etre strictement positif."
            )

        self.standardize = bool(standardize)

        self.moyenne_ = None
        self.ecart_type_ = None
        self.projection_ = None
        self.dimensions_latentes_ = None

    def _standardiser(self, X, apprentissage=False):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X doit avoir la forme (n_etats, n_features)."
            )

        if apprentissage:
            self.moyenne_ = X.mean(axis=0)

            ecart_type = X.std(axis=0)

            ecart_type = np.where(
                ecart_type < 1e-12,
                1.0,
                ecart_type,
            )

            self.ecart_type_ = ecart_type

        if self.moyenne_ is None or self.ecart_type_ is None:
            raise RuntimeError(
                "La metrique doit etre ajustee avant transform()."
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
                "X doit avoir la forme (n_etats, n_features)."
            )

        if D.ndim != 2:
            raise ValueError(
                "D doit avoir la forme (n_etats, n_etats)."
            )

        if X.shape[0] != D.shape[0]:
            raise ValueError(
                "X et D doivent avoir le meme nombre d'etats."
            )

        if D.shape[0] != D.shape[1]:
            raise ValueError(
                "D doit etre carree."
            )

        if not np.allclose(D, D.T, atol=1e-10):
            raise ValueError(
                "D doit etre symetrique."
            )

        if np.any(np.diag(D) != 0.0):
            raise ValueError(
                "La diagonale de D doit etre nulle."
            )

        if not np.all(np.isfinite(D)):
            raise ValueError(
                "D contient des valeurs non finies."
            )

        X_standardise = self._standardiser(
            X,
            apprentissage=True,
        )

        n_etats = X.shape[0]

        if n_etats < self.k + 1:
            raise ValueError(
                "Il faut au moins k + 1 etats pour ajuster "
                "une metrique de dimension k."
            )

        D2 = D ** 2

        J = np.eye(n_etats) - (
            np.ones((n_etats, n_etats))
            / n_etats
        )

        B = -0.5 * (J @ D2 @ J)
        B = 0.5 * (B + B.T)

        valeurs, vecteurs = np.linalg.eigh(B)

        ordre = np.argsort(valeurs)[::-1]

        valeurs = valeurs[ordre]
        vecteurs = vecteurs[:, ordre]

        if select_k:
            positif = valeurs > 1e-10

            nombre_dimensions = max(
                1,
                int(np.sum(positif)),
            )

            self.k = min(
                self.k,
                nombre_dimensions,
            )

        valeurs = valeurs[: self.k]
        vecteurs = vecteurs[:, : self.k]

        valeurs = np.maximum(valeurs, 0.0)

        Z_train = vecteurs * np.sqrt(valeurs)[None, :]

        self.dimensions_latentes_ = self.k

        gramme = X_standardise.T @ X_standardise

        gramme_regularise = (
            gramme
            + self.reg
            * np.eye(gramme.shape[0])
        )

        self.projection_ = np.linalg.solve(
            gramme_regularise,
            X_standardise.T @ Z_train,
        )

        if verbose:
            print(
                "Metrique ajustee : "
                f"{self.k} dimensions latentes."
            )

        return self

    def transform(self, X):
        if self.projection_ is None:
            raise RuntimeError(
                "La metrique doit etre ajustee avant transform()."
            )

        X_standardise = self._standardiser(
            X,
            apprentissage=False,
        )

        return X_standardise @ self.projection_

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
PYTHON


# ---------------------------------------------------------------------
# 2. Reecriture complete de joindre_futur_passe_metric.py
# ---------------------------------------------------------------------

cat > joindre_futur_passe_metric.py <<'PYTHON'
#!/usr/bin/env python3
"""
Jonction entre le prototype du futur passe et la metrique subjective.

Le script :
    - simule le prototype hippocampo-cingulaire ;
    - construit des representations multimodales present/futur ;
    - entraine SubjectiveMetric sur la periode d'entrainement ;
    - genere 64 branches futures par etat present ;
    - mesure volume, ancrage, distance et volume attache ;
    - compare la condition reelle a deux controles ;
    - repete le controle par permutation ;
    - calcule des contrastes apparies, des p-values empiriques
      et des intervalles bootstrap par blocs.

Les branches futures d'un meme etat present ne sont pas des
observations independantes : elles sont traitees comme un ensemble
interne de simulations conditionnelles.
"""

from pathlib import Path
import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from prototype_futur_passe import (
    N,
    N_PAS,
    DT,
    DUREE,
    SEED,
    initialiser_parametres,
    simuler,
    generer_futur,
    encoder_memoire,
    sigmoid,
    cosinus,
)

from metrique_subjective import (
    SubjectiveMetric,
    ball_volume_factor,
)


# Seuil temporel local au script de jonction.
# Il separe les etats d'entrainement et d'evaluation.
T0_CAUSALITE = 0.5 * DUREE

DIMENSION_LATENTE = 6
NOMBRE_BRANCHES = 64
NOMBRE_INSTANTS = 40
NOMBRE_PERMUTATIONS = 100

GRAINE_ENSEMBLE = 9000
GRAINE_PERMUTATIONS = 20000

REGULARISATION = 1e-8

LONGUEURS_BLOCS = [4, 8, 12]
NOMBRE_BOOTSTRAP = 2000
GRAINE_BOOTSTRAP = 1234

DOSSIER_SORTIE = Path(
    "resultats/jonction_futur_passe_metric"
)


def resume_scene(scene, points):
    scene = np.asarray(scene, dtype=float)
    points = np.asarray(points, dtype=float)

    if scene.ndim != 1:
        raise ValueError(
            "scene doit etre un vecteur de forme (N,)."
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
        float(scene.mean()),
        float(scene.std()),
        float(centre[0]),
        float(centre[1]),
    ])


def distances_euclidiennes(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X doit avoir la forme (n_etats, n_features)."
        )

    differences = (
        X[:, None, :]
        - X[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    distances = 0.5 * (
        distances + distances.T
    )

    np.fill_diagonal(
        distances,
        0.0,
    )

    return distances


def construire_vecteur_multimodal(
    H,
    C,
    etat_global,
    M,
    scene,
    points,
):
    H = np.asarray(H, dtype=float)
    C = np.asarray(C, dtype=float)
    etat_global = np.asarray(
        etat_global,
        dtype=float,
    )
    M = np.asarray(M, dtype=float)

    scene_resume = resume_scene(
        scene,
        points,
    )

    return np.concatenate([
        H,
        C,
        etat_global,
        M,
        scene_resume,
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

    M_passe = np.asarray(
        resultats["M_passe"][index],
        dtype=float,
    )

    scene = np.asarray(
        resultats["scene"][index],
        dtype=float,
    )

    etat_global = np.array([
        E.mean(),
        I.mean(),
    ])

    X_present = construire_vecteur_multimodal(
        H,
        C,
        etat_global,
        M_passe,
        scene,
        points,
    )

    return {
        "E": E,
        "I": I,
        "C": C,
        "H": H,
        "M_passe": M_passe,
        "scene": scene,
        "etat_global": etat_global,
        "X_present": X_present,
    }


def calculer_contexte_constant(
    resultats,
    indices_entrainement,
):
    C_train = np.asarray(
        resultats["C"][indices_entrainement],
        dtype=float,
    )

    if C_train.ndim != 2:
        raise ValueError(
            "C_train doit avoir la forme (temps, DIM_C)."
        )

    return C_train.mean(axis=0)


def generer_ensemble_futur(
    resultats,
    index,
    parametres,
    points,
    nombre_branches=NOMBRE_BRANCHES,
    graine=GRAINE_ENSEMBLE,
    contexte_futur=None,
    permuter_scenes=False,
):
    etat = construire_etat_present(
        resultats,
        index,
        points,
    )

    if contexte_futur is None:
        contexte_utilise = etat["C"].copy()
    else:
        contexte_utilise = np.asarray(
            contexte_futur,
            dtype=float,
        ).copy()

    if contexte_utilise.shape != etat["C"].shape:
        raise ValueError(
            "contexte_futur doit avoir la meme forme que C."
        )

    H_futurs = []
    M_futurs = []
    scenes_futures = []

    for branche in range(nombre_branches):
        rng_futur = np.random.default_rng(
            graine + branche
        )

        H_futur = generer_futur(
            etat["H"],
            contexte_utilise,
            etat["etat_global"],
            parametres,
            rng_futur,
        )

        M_futur = encoder_memoire(
            etat["etat_global"],
            contexte_utilise,
            H_futur,
            parametres,
        )

        scene_future = sigmoid(
            parametres["decodeur_scene"] @ H_futur
        )

        H_futurs.append(H_futur)
        M_futurs.append(M_futur)
        scenes_futures.append(scene_future)

    H_futurs = np.asarray(H_futurs)
    M_futurs = np.asarray(M_futurs)
    scenes_futures = np.asarray(scenes_futures)

    permutation = np.arange(nombre_branches)

    if permuter_scenes:
        rng_permutation = np.random.default_rng(
            graine + 987654
        )

        permutation = rng_permutation.permutation(
            nombre_branches
        )

        scenes_utilisees = (
            scenes_futures[permutation]
        )
    else:
        scenes_utilisees = scenes_futures

    X_futurs = []

    for branche in range(nombre_branches):
        X_future = construire_vecteur_multimodal(
            H_futurs[branche],
            contexte_utilise,
            etat["etat_global"],
            M_futurs[branche],
            scenes_utilisees[branche],
            points,
        )

        X_futurs.append(X_future)

    return {
        "X_present": etat["X_present"],
        "X_futurs": np.asarray(X_futurs),
        "H_futurs": H_futurs,
        "M_futurs": M_futurs,
        "scenes_futures": scenes_futures,
        "scenes_utilisees": scenes_utilisees,
        "permutation_scenes": permutation,
        "contexte_utilise": contexte_utilise,
        "etat_global": etat["etat_global"],
    }


def permuter_ensemble_scenes(
    ensemble,
    points,
    graine,
):
    """
    Permute uniquement les scenes entre branches.

    H_futurs, M_futurs et le contexte restent inchanges.
    """
    nombre_branches = len(
        ensemble["H_futurs"]
    )

    rng = np.random.default_rng(
        graine
    )

    permutation = rng.permutation(
        nombre_branches
    )

    scenes_utilisees = (
        ensemble["scenes_futures"][permutation]
    )

    X_futurs = []

    for branche in range(nombre_branches):
        X_future = construire_vecteur_multimodal(
            ensemble["H_futurs"][branche],
            ensemble["contexte_utilise"],
            ensemble["etat_global"],
            ensemble["M_futurs"][branche],
            scenes_utilisees[branche],
            points,
        )

        X_futurs.append(X_future)

    resultat = dict(ensemble)

    resultat["X_futurs"] = np.asarray(
        X_futurs
    )

    resultat["scenes_utilisees"] = (
        scenes_utilisees
    )

    resultat["permutation_scenes"] = (
        permutation
    )

    return resultat


def volume_covariance_latente(
    Z,
    regularisation=REGULARISATION,
):
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError(
            "Z doit avoir deux dimensions."
        )

    nombre_points, dimension = Z.shape

    if nombre_points <= dimension:
        raise ValueError(
            "Le nombre de branches doit etre "
            "superieur a la dimension latente."
        )

    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "Z contient des valeurs non finies."
        )

    covariance = np.cov(
        Z,
        rowvar=False,
        ddof=1,
    )

    covariance = np.atleast_2d(
        covariance
    )

    covariance_regularisee = (
        covariance
        + regularisation
        * np.eye(dimension)
    )

    signe, logdet = np.linalg.slogdet(
        covariance_regularisee
    )

    if signe <= 0 or not np.isfinite(logdet):
        return 0.0

    facteur = ball_volume_factor(
        dimension
    )

    return float(
        facteur * np.exp(0.5 * logdet)
    )


def analyser_volume_futur(
    ensemble,
    metrique,
):
    X_present = ensemble["X_present"]
    X_futurs = ensemble["X_futurs"]

    if X_futurs.shape[0] <= metrique.k:
        raise ValueError(
            "Il faut plus de branches futures "
            "que de dimensions latentes."
        )

    Z_present = metrique.transform(
        X_present[None, :]
    )[0]

    Z_futurs = metrique.transform(
        X_futurs
    )

    centre_futur = Z_futurs.mean(
        axis=0
    )

    volume_futur = volume_covariance_latente(
        Z_futurs
    )

    ancrage_cosinus = cosinus(
        Z_present,
        centre_futur,
    )

    distance_present_futur = float(
        np.linalg.norm(
            Z_present - centre_futur
        )
    )

    coefficient_attachement = (
        ancrage_cosinus + 1.0
    ) / 2.0

    volume_futur_attache = (
        coefficient_attachement
        * volume_futur
    )

    Z_joint = np.vstack([
        Z_present[None, :],
        Z_futurs,
    ])

    volume_joint = volume_covariance_latente(
        Z_joint
    )

    return {
        "volume_futur": float(
            volume_futur
        ),
        "centre_futur": centre_futur,
        "ancrage_cosinus": float(
            ancrage_cosinus
        ),
        "coefficient_attachement": float(
            coefficient_attachement
        ),
        "distance_present_futur": (
            distance_present_futur
        ),
        "volume_futur_attache": float(
            volume_futur_attache
        ),
        "volume_joint_present_futur": float(
            volume_joint
        ),
        "Z_present": Z_present,
        "Z_futurs": Z_futurs,
    }


def transformer_ancrage(ancrage):
    ancrage = np.clip(
        np.asarray(ancrage, dtype=float),
        -0.999999,
        0.999999,
    )

    return np.arctanh(ancrage)


def transformer_volume(volume):
    volume = np.asarray(
        volume,
        dtype=float,
    )

    return np.log(
        np.maximum(volume, 1e-12)
    )


def calculer_contrastes(
    reel,
    controle_constant,
    controle_permute,
    transformation,
):
    reel = transformation(reel)

    constant = transformation(
        controle_constant
    )

    permute = transformation(
        controle_permute
    )

    moyenne_constant = constant.mean(
        axis=0
    )

    moyenne_permute = permute.mean(
        axis=0
    )

    contraste_constant = (
        reel - moyenne_constant
    )

    contraste_permute = (
        reel - moyenne_permute
    )

    return {
        "reel": reel,
        "constant": constant,
        "permute": permute,
        "contraste_constant": (
            contraste_constant
        ),
        "contraste_permute": (
            contraste_permute
        ),
    }


def p_empirique_superieur(
    valeur_reelle,
    valeurs_controle,
):
    valeurs_controle = np.asarray(
        valeurs_controle,
        dtype=float,
    )

    valeurs_controle = valeurs_controle[
        np.isfinite(valeurs_controle)
    ]

    if len(valeurs_controle) == 0:
        return np.nan

    return float(
        (
            1
            + np.sum(
                valeurs_controle
                >= valeur_reelle
            )
        )
        / (len(valeurs_controle) + 1)
    )


def p_empiriques_point_par_point(
    reel,
    controles,
):
    reel = np.asarray(
        reel,
        dtype=float,
    )

    controles = np.asarray(
        controles,
        dtype=float,
    )

    if controles.ndim != 2:
        raise ValueError(
            "controles doit avoir la forme (B, T)."
        )

    if controles.shape[1] != len(reel):
        raise ValueError(
            "Les controles et reel doivent "
            "avoir le meme nombre d'etats."
        )

    p_values = np.full(
        len(reel),
        np.nan,
    )

    for t in range(len(reel)):
        p_values[t] = p_empirique_superieur(
            reel[t],
            controles[:, t],
        )

    return p_values


def moyenne_valide(valeurs):
    valeurs = np.asarray(
        valeurs,
        dtype=float,
    )

    valeurs = valeurs[
        np.isfinite(valeurs)
    ]

    if len(valeurs) == 0:
        return np.nan

    return float(
        np.mean(valeurs)
    )


def indices_bootstrap_blocs(
    n,
    longueur_bloc,
    rng,
):
    if longueur_bloc < 1:
        raise ValueError(
            "longueur_bloc doit etre positive."
        )

    indices = []

    while len(indices) < n:
        debut = rng.integers(0, n)

        bloc = [
            (debut + offset) % n
            for offset in range(longueur_bloc)
        ]

        indices.extend(bloc)

    return np.asarray(
        indices[:n],
        dtype=int,
    )


def bootstrap_moyenne_contraste(
    contraste,
    longueur_bloc,
    nombre_bootstrap=NOMBRE_BOOTSTRAP,
    graine=GRAINE_BOOTSTRAP,
):
    contraste = np.asarray(
        contraste,
        dtype=float,
    )

    valide = np.isfinite(contraste)

    if not np.all(valide):
        contraste = contraste[valide]

    n = len(contraste)

    if n < longueur_bloc:
        raise ValueError(
            "La serie est trop courte pour "
            "la longueur de bloc choisie."
        )

    rng = np.random.default_rng(
        graine
    )

    estimations = np.zeros(
        nombre_bootstrap
    )

    for repetition in range(
        nombre_bootstrap
    ):
        indices = indices_bootstrap_blocs(
            n,
            longueur_bloc,
            rng,
        )

        estimations[repetition] = np.mean(
            contraste[indices]
        )

    return {
        "estimation": float(
            np.mean(contraste)
        ),
        "intervalle_95": np.quantile(
            estimations,
            [0.025, 0.975],
        ).tolist(),
    }


def ratio_geometrique(
    volume_reel,
    volume_controle,
):
    reel = transformer_volume(
        volume_reel
    )

    controle = transformer_volume(
        volume_controle
    )

    difference = (
        np.mean(reel)
        - np.mean(controle, axis=0).mean()
    )

    return float(
        np.exp(difference)
    )


def tracer_mesures(
    temps,
    volumes_attaches,
    ancrages,
    distances,
    volumes_futurs,
):
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        sharex=True,
    )

    axes[0].plot(
        temps,
        volumes_futurs,
        color="tab:blue",
    )
    axes[0].set_ylabel("Volume futur")
    axes[0].set_title(
        "Volume de covariance des futurs simules"
    )

    axes[1].plot(
        temps,
        ancrages,
        color="tab:orange",
    )
    axes[1].set_ylabel("Ancrage present-futur")
    axes[1].set_title(
        "Similarite cosinus present / centre futur"
    )

    axes[2].plot(
        temps,
        distances,
        color="tab:red",
    )
    axes[2].set_ylabel("Distance present-futur")
    axes[2].set_title(
        "Distance euclidienne present / centre futur"
    )

    axes[3].plot(
        temps,
        volumes_attaches,
        color="tab:green",
    )
    axes[3].set_ylabel("Volume futur attache")
    axes[3].set_xlabel("Temps du modele")
    axes[3].set_title(
        "Volume futur pondere par l'ancrage"
    )

    fig.tight_layout()

    chemin = DOSSIER_SORTIE / "mesures_futures.png"
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def main():
    DOSSIER_SORTIE.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Jonction futur passe / metrique subjective ===")
    print(f"Dimension latente : {DIMENSION_LATENTE}")
    print(f"Branches futures : {NOMBRE_BRANCHES}")
    print(f"Instants analyses : {NOMBRE_INSTANTS}")
    print(f"Permutations : {NOMBRE_PERMUTATIONS}")

    rng = np.random.default_rng(SEED)

    points = rng.uniform(
        0.0,
        1.0,
        size=(N, 2),
    )

    parametres = initialiser_parametres(
        rng,
        points,
    )

    resultats = simuler(
        rng,
        points,
        parametres,
        graine_futur=1,
    )

    indices_entrainement = np.where(
        resultats["temps"] <= T0_CAUSALITE
    )[0]

    contexte_constant = calculer_contexte_constant(
        resultats,
        indices_entrainement,
    )

    indices_entrainement_echantillon = (
        indices_entrainement[::10]
    )

    X_train = []

    for index in indices_entrainement_echantillon:
        etat = construire_etat_present(
            resultats,
            index,
            points,
        )

        X_train.append(etat["X_present"])

    X_train = np.asarray(X_train)

    D_train = distances_euclidiennes(
        X_train
    )

    metrique = SubjectiveMetric(
        k=DIMENSION_LATENTE,
        reg=1e-2,
        standardize=True,
    )

    metrique.fit(
        X_train,
        D_train,
        select_k=False,
        verbose=False,
    )

    indices_analyse = np.linspace(
        0,
        len(resultats["temps"]) - 1,
        NOMBRE_INSTANTS,
        dtype=int,
    )

    temps = np.array([
        resultats["temps"][index]
        for index in indices_analyse
    ])

    ancrage_reel = np.zeros(NOMBRE_INSTANTS)
    volume_attache_reel = np.zeros(NOMBRE_INSTANTS)
    volume_futur_reel = np.zeros(NOMBRE_INSTANTS)
    distance_reel = np.zeros(NOMBRE_INSTANTS)

    ancrage_constante = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    volume_attache_constante = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    volume_futur_constante = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    distance_constante = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )

    ancrage_permute = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    volume_attache_permute = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    volume_futur_permute = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )
    distance_permute = np.zeros(
        (NOMBRE_PERMUTATIONS, NOMBRE_INSTANTS)
    )

    for numero, index in enumerate(
        indices_analyse
    ):
        graine = (
            GRAINE_ENSEMBLE
            + numero * 100
        )

        ensemble_reel = generer_ensemble_futur(
            resultats,
            index,
            parametres,
            points,
            nombre_branches=NOMBRE_BRANCHES,
            graine=graine,
        )

        mesure_reelle = analyser_volume_futur(
            ensemble_reel,
            metrique,
        )

        ancrage_reel[numero] = mesure_reelle[
            "ancrage_cosinus"
        ]
        volume_attache_reel[numero] = mesure_reelle[
            "volume_futur_attache"
        ]
        volume_futur_reel[numero] = mesure_reelle[
            "volume_futur"
        ]
        distance_reel[numero] = mesure_reelle[
            "distance_present_futur"
        ]

        for repetition in range(
            NOMBRE_PERMUTATIONS
        ):
            graine_constante = (
                GRAINE_PERMUTATIONS
                + 100000
                + repetition * 100
                + numero
            )

            ensemble_contexte_constant = (
                generer_ensemble_futur(
                    resultats,
                    index,
                    parametres,
                    points,
                    nombre_branches=NOMBRE_BRANCHES,
                    graine=graine_constante,
                    contexte_futur=contexte_constant,
                    permuter_scenes=False,
                )
            )

            mesure_contexte_constant = (
                analyser_volume_futur(
                    ensemble_contexte_constant,
                    metrique,
                )
            )

            ancrage_constante[
                repetition,
                numero,
            ] = mesure_contexte_constant[
                "ancrage_cosinus"
            ]

            volume_attache_constante[
                repetition,
                numero,
            ] = mesure_contexte_constant[
                "volume_futur_attache"
            ]

            volume_futur_constante[
                repetition,
                numero,
            ] = mesure_contexte_constant[
                "volume_futur"
            ]

            distance_constante[
                repetition,
                numero,
            ] = mesure_contexte_constant[
                "distance_present_futur"
            ]

        for repetition in range(
            NOMBRE_PERMUTATIONS
        ):
            ensemble_permute = (
                permuter_ensemble_scenes(
                    ensemble_reel,
                    points,
                    graine=(
                        GRAINE_PERMUTATIONS
                        + repetition * 100
                        + numero
                    ),
                )
            )

            mesure_permutee = (
                analyser_volume_futur(
                    ensemble_permute,
                    metrique,
                )
            )

            ancrage_permute[
                repetition, numero
            ] = mesure_permutee[
                "ancrage_cosinus"
            ]

            volume_attache_permute[
                repetition, numero
            ] = mesure_permutee[
                "volume_futur_attache"
            ]

            volume_futur_permute[
                repetition, numero
            ] = mesure_permutee[
                "volume_futur"
            ]

            distance_permute[
                repetition, numero
            ] = mesure_permutee[
                "distance_present_futur"
            ]

        print(
            f"Instant {numero + 1}/{NOMBRE_INSTANTS} : "
            f"t={temps[numero]:.2f}, "
            f"A={ancrage_reel[numero]:.4f}, "
            f"V={volume_attache_reel[numero]:.4e}, "
            f"D={distance_reel[numero]:.4f}"
        )

    resultat_ancrage = calculer_contrastes(
        ancrage_reel,
        ancrage_constante,
        ancrage_permute,
        transformer_ancrage,
    )

    resultat_volume = calculer_contrastes(
        volume_attache_reel,
        volume_attache_constante,
        volume_attache_permute,
        transformer_volume,
    )

    resultat_distance = calculer_contrastes(
        distance_reel,
        distance_constante,
        distance_permute,
        lambda x: x,
    )

    p_ancrage_constant = (
        p_empiriques_point_par_point(
            resultat_ancrage["reel"],
            resultat_ancrage["constant"],
        )
    )

    p_ancrage_permute = (
        p_empiriques_point_par_point(
            resultat_ancrage["reel"],
            resultat_ancrage["permute"],
        )
    )

    p_volume_constant = (
        p_empiriques_point_par_point(
            resultat_volume["reel"],
            resultat_volume["constant"],
        )
    )

    p_volume_permute = (
        p_empiriques_point_par_point(
            resultat_volume["reel"],
            resultat_volume["permute"],
        )
    )

    p_distance_constant = (
        p_empiriques_point_par_point(
            -resultat_distance["reel"],
            -resultat_distance["constant"],
        )
    )

    p_distance_permute = (
        p_empiriques_point_par_point(
            -resultat_distance["reel"],
            -resultat_distance["permute"],
        )
    )

    longueurs_blocs_valides = sorted(
        {
            int(longueur)
            for longueur in LONGUEURS_BLOCS
            if 1 <= int(longueur) <= len(temps)
        }
    )

    bootstraps = {}

    for longueur_bloc in longueurs_blocs_valides:
        bootstraps[
            f"bloc_{longueur_bloc}"
        ] = {
            "ancrage_constant": (
                bootstrap_moyenne_contraste(
                    resultat_ancrage[
                        "contraste_constant"
                    ],
                    longueur_bloc,
                )
            ),
            "ancrage_permute": (
                bootstrap_moyenne_contraste(
                    resultat_ancrage[
                        "contraste_permute"
                    ],
                    longueur_bloc,
                )
            ),
            "volume_constant": (
                bootstrap_moyenne_contraste(
                    resultat_volume[
                        "contraste_constant"
                    ],
                    longueur_bloc,
                )
            ),
            "volume_permute": (
                bootstrap_moyenne_contraste(
                    resultat_volume[
                        "contraste_permute"
                    ],
                    longueur_bloc,
                )
            ),
            "distance_constant": (
                bootstrap_moyenne_contraste(
                    resultat_distance[
                        "contraste_constant"
                    ],
                    longueur_bloc,
                )
            ),
            "distance_permute": (
                bootstrap_moyenne_contraste(
                    resultat_distance[
                        "contraste_permute"
                    ],
                    longueur_bloc,
                )
            ),
        }

    ratio_volume_constant = ratio_geometrique(
        volume_attache_reel,
        volume_attache_constante,
    )

    ratio_volume_permute = ratio_geometrique(
        volume_attache_reel,
        volume_attache_permute,
    )

    rapport = {
        "statut": "Jonction futur passe / metrique subjective",
        "dimension_latente": DIMENSION_LATENTE,
        "nombre_branches": NOMBRE_BRANCHES,
        "nombre_instants": NOMBRE_INSTANTS,
        "nombre_permutations": NOMBRE_PERMUTATIONS,
        "temps": temps.tolist(),
        "ancrage_reel": ancrage_reel.tolist(),
        "volume_attache_reel": volume_attache_reel.tolist(),
        "volume_futur_reel": volume_futur_reel.tolist(),
        "distance_reel": distance_reel.tolist(),
        "ancrage_constante": ancrage_constante.tolist(),
        "volume_attache_constante": volume_attache_constante.tolist(),
        "volume_futur_constante": volume_futur_constante.tolist(),
        "distance_constante": distance_constante.tolist(),
        "ancrage_permute": ancrage_permute.tolist(),
        "volume_attache_permute": volume_attache_permute.tolist(),
        "volume_futur_permute": volume_futur_permute.tolist(),
        "distance_permute": distance_permute.tolist(),
        "p_ancrage_constant": p_ancrage_constant.tolist(),
        "p_ancrage_permute": p_ancrage_permute.tolist(),
        "p_volume_constant": p_volume_constant.tolist(),
        "p_volume_permute": p_volume_permute.tolist(),
        "p_distance_constant": p_distance_constant.tolist(),
        "p_distance_permute": p_distance_permute.tolist(),
        "bootstraps": bootstraps,
        "ratio_volume_constant": ratio_volume_constant,
        "ratio_volume_permute": ratio_volume_permute,
    }

    with (DOSSIER_SORTIE / "rapport.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            rapport,
            f,
            indent=2,
            allow_nan=False,
        )

    tracer_mesures(
        temps,
        volume_attache_reel,
        ancrage_reel,
        distance_reel,
        volume_futur_reel,
    )

    print("\n=== Contrastes moyens ===")

    print(
        "Ancrage reel - contexte constant : "
        f"{moyenne_valide(resultat_ancrage['contraste_constant']):.6f}"
    )
    print(
        "Ancrage reel - scenes permutees : "
        f"{moyenne_valide(resultat_ancrage['contraste_permute']):.6f}"
    )
    print(
        "log volume attache reel - contexte constant : "
        f"{moyenne_valide(resultat_volume['contraste_constant']):.6f}"
    )
    print(
        "log volume attache reel - scenes permutees : "
        f"{moyenne_valide(resultat_volume['contraste_permute']):.6f}"
    )
    print(
        "Distance reel - contexte constant : "
        f"{moyenne_valide(resultat_distance['contraste_constant']):.6f}"
    )
    print(
        "Distance reel - scenes permutees : "
        f"{moyenne_valide(resultat_distance['contraste_permute']):.6f}"
    )

    print("\n=== Ratios geometriques ===")
    print(
        "Ratio volume attache reel / contexte constant : "
        f"{ratio_volume_constant:.6f}"
    )
    print(
        "Ratio volume attache reel / scenes permutees : "
        f"{ratio_volume_permute:.6f}"
    )

    print(f"\nResultats : {DOSSIER_SORTIE}")
    print(
        "Les branches futures sont des repetitions internes, "
        "pas des sujets independants."
    )
    print(
        "Aucune interpretation clinique ou de conscience."
    )


if __name__ == "__main__":
    main()
PYTHON

python -m py_compile metrique_subjective.py
python -m py_compile joindre_futur_passe_metric.py

python -X faulthandler -u joindre_futur_passe_metric.py \
  2>&1 | tee erreur_termux.txt
