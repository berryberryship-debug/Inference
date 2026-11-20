cd ~

cat > analyse_phase_relative.py <<'PYTHON'
#!/usr/bin/env python3
"""Phase relative E-I par Hilbert, moyenne circulaire et bootstrap par blocs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, hilbert, periodogram, sosfiltfilt

FICHIER_TRAJECTOIRE = Path(
    "resultats/experience_longue/trajectoire_longue.npz"
)
DOSSIER = Path("resultats/analyse_phase_relative")

FENETRES = [
    (240.0, 320.0, "240-320"),
    (320.0, 400.0, "320-400"),
]

NIVEAU = 0.95
CYCLES_PAR_BLOC = 2
REPETITIONS = 5000
SEED_BOOTSTRAP = 1234


def normaliser_angle_deg(angle):
    """Ramene un angle dans [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def segments_ic_circulaire_deg(borne_basse, borne_haute):
    """Convertit un arc bootstrap non replie en segments affichables."""
    largeur = borne_haute - borne_basse

    if largeur < 0:
        raise ValueError("Les bornes de l'IC doivent etre ordonnees.")

    if largeur >= 360.0:
        return "cercle complet"

    basse = normaliser_angle_deg(borne_basse)
    haute = normaliser_angle_deg(borne_haute)

    if basse <= haute:
        return [(basse, haute)]

    return [
        (basse, 180.0),
        (-180.0, haute),
    ]


def bootstrap_ci_phase(
    delta,
    fs,
    f0,
    niveau=NIVEAU,
    cycles_par_bloc=CYCLES_PAR_BLOC,
    repetitions=REPETITIONS,
    seed=SEED_BOOTSTRAP,
):
    """IC bootstrap par blocs pour une moyenne de phase circulaire."""
    delta = np.asarray(delta)
    n = len(delta)

    phase_estimee = float(np.angle(np.mean(np.exp(1j * delta))))
    longueur_bloc = max(
        4,
        int(round(cycles_par_bloc * fs / f0)),
    )

    if longueur_bloc >= n:
        raise ValueError(
            "Fenetre trop courte pour un bootstrap par blocs."
        )

    rng = np.random.default_rng(seed)
    phases_bootstrap = np.empty(repetitions, dtype=float)
    nombre_de_depart = n - longueur_bloc + 1

    for b in range(repetitions):
        indices = []

        while len(indices) < n:
            depart = int(rng.integers(0, nombre_de_depart))
            indices.extend(range(depart, depart + longueur_bloc))

        echantillon = delta[np.asarray(indices[:n])]
        phases_bootstrap[b] = np.angle(
            np.mean(np.exp(1j * echantillon))
        )

    deviations = np.angle(
        np.exp(1j * (phases_bootstrap - phase_estimee))
    )

    alpha = 1.0 - niveau
    quantile_bas = float(np.quantile(deviations, alpha / 2.0))
    quantile_haut = float(np.quantile(deviations, 1.0 - alpha / 2.0))

    ci_bas = phase_estimee + quantile_bas
    ci_haut = phase_estimee + quantile_haut

    return {
        "phase": phase_estimee,
        "ci_bas": ci_bas,
        "ci_haut": ci_haut,
        "longueur_bloc": longueur_bloc,
    }


def phase_relative_fenetre(temps, E, I, debut, fin):
    """Estime la phase relative E-I dans une fenetre temporelle."""
    masque = (temps >= debut) & (temps <= fin)
    t = temps[masque]

    if t.size < 20:
        raise ValueError("Fenetre trop courte.")

    x = E[masque].mean(axis=1)
    y = I[masque].mean(axis=1)

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    duree = float(t[-1] - t[0])

    x = detrend(x, type="linear")
    y = detrend(y, type="linear")

    frequences, puissance_E = periodogram(x, fs=fs, window="hann")
    _, puissance_I = periodogram(y, fs=fs, window="hann")

    valide = (frequences > 0.0) & (frequences < 0.9 * fs / 2.0)
    puissance_commune = puissance_E + puissance_I
    f0 = float(frequences[valide][np.argmax(puissance_commune[valide])])

    df = float(frequences[1] - frequences[0])
    largeur = max(2.0 * df, 0.25 * f0)
    fbas = max(df, f0 - largeur)
    fhaut = min(0.9 * fs / 2.0, f0 + largeur)

    if not (0.0 < fbas < fhaut < fs / 2.0):
        raise ValueError("Bande de filtrage incompatible.")

    sos = butter(
        4,
        [fbas, fhaut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    x_filtre = sosfiltfilt(sos, x)
    y_filtre = sosfiltfilt(sos, y)

    z_E = hilbert(x_filtre)
    z_I = hilbert(y_filtre)

    delta_phi = np.angle(z_E * np.conj(z_I))

    nombre_bord = int(np.ceil(2.0 * fs / f0))
    if len(delta_phi) <= 2 * nombre_bord + 10:
        raise ValueError("Fenetre trop courte apres exclusion des bords.")

    delta = delta_phi[nombre_bord:-nombre_bord]

    ic = bootstrap_ci_phase(
        delta=delta,
        fs=fs,
        f0=f0,
    )

    phase = ic["phase"]
    ci_bas = ic["ci_bas"]
    ci_haut = ic["ci_haut"]

    retard_I_sur_E = phase / (2.0 * np.pi * f0)
    retard_bas = ci_bas / (2.0 * np.pi * f0)
    retard_haut = ci_haut / (2.0 * np.pi * f0)

    return {
        "fenetre": [float(debut), float(fin)],
        "frequence_dominante": f0,
        "periode": 1.0 / f0,
        "phase_rad": phase,
        "phase_deg": float(np.degrees(phase)),
        "phase_IC95_deg": (
            float(np.degrees(ci_bas)),
            float(np.degrees(ci_haut)),
        ),
        "retard_I_sur_E": retard_I_sur_E,
        "retard_IC95": (retard_bas, retard_haut),
        "concentration_R": float(abs(np.mean(np.exp(1j * delta)))),
        "longueur_bloc": int(ic["longueur_bloc"]),
        "nombre_cycles_fenetre": float(duree * f0),
        "delta_phi": delta,
    }


def tracer_phase_et_ic(ax, resultat, rayon_interieur, rayon_exterieur, couleur, etiquette):
    phase = resultat["phase_rad"]
    ci_bas, ci_haut = resultat["phase_IC95_deg"]

    theta = np.linspace(np.radians(ci_bas), np.radians(ci_haut), 400)

    ax.fill_between(
        theta,
        rayon_interieur,
        rayon_exterieur,
        color=couleur,
        alpha=0.30,
        label=(
            f"{etiquette} : {np.degrees(phase):.1f} deg "
            f"(IC95% {ci_bas:.1f}-{ci_haut:.1f} deg)"
        ),
    )
    ax.plot(
        [phase, phase],
        [rayon_interieur, rayon_exterieur],
        color=couleur,
        linewidth=2.5,
    )
    ax.plot(
        phase,
        (rayon_interieur + rayon_exterieur) / 2.0,
        "o",
        color=couleur,
        markersize=8,
    )


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    if not FICHIER_TRAJECTOIRE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_TRAJECTOIRE}. "
            "Lance d'abord experience_longue.py."
        )

    with np.load(FICHIER_TRAJECTOIRE, allow_pickle=False) as donnees:
        temps = donnees["temps"]
        E = donnees["E"]
        I = donnees["I"]

    if E.shape != I.shape or E.shape[0] != len(temps):
        raise ValueError("Dimensions incoherentes dans la trajectoire.")

    resultats = []
    print("=== Phase relative E-I, transformee de Hilbert ===")

    for debut, fin, etiquette in FENETRES:
        resultat = phase_relative_fenetre(temps, E, I, debut, fin)
        resultats.append(resultat)

        segments = segments_ic_circulaire_deg(
            resultat["phase_IC95_deg"][0],
            resultat["phase_IC95_deg"][1],
        )

        if isinstance(segments, str):
            texte_ic = segments
        else:
            texte_ic = " union ".join(
                f"[{bas:.1f}, {haut:.1f}] deg"
                for bas, haut in segments
            )

        print(f"\nFenetre {etiquette}")
        print(f"Frequence dominante : {resultat['frequence_dominante']:.6g}")
        print(f"Periode estimee : {resultat['periode']:.6g}")
        print(
            f"Phase relative E-I : {resultat['phase_deg']:.2f} deg ; "
            f"IC95% circulaire : {texte_ic}"
        )
        print(
            f"Retard de I par rapport a E : "
            f"{resultat['retard_I_sur_E']:.4g} ; "
            f"IC95% : "
            f"[{resultat['retard_IC95'][0]:.4g}, "
            f"{resultat['retard_IC95'][1]:.4g}]"
        )
        print(f"Concentration circulaire R : {resultat['concentration_R']:.3f}")
        print(f"Longueur de bloc bootstrap : {resultat['longueur_bloc']}")

    difference = np.angle(
        np.exp(
            1j * (
                resultats[1]["phase_rad"]
                - resultats[0]["phase_rad"]
            )
        )
    )

    print(
        f"\nDifference circulaire entre fenetres : "
        f"{np.degrees(difference):.2f} deg"
    )

    rapport = []
    for resultat in resultats:
        rapport.append(
            {
                cle: valeur
                for cle, valeur in resultat.items()
                if cle != "delta_phi"
            }
        )

    rapport[0]["difference_circulaire_avec_fenetre_1_rad"] = float(difference)
    rapport[0]["difference_circulaire_avec_fenetre_1_deg"] = float(
        np.degrees(difference)
    )

    with (DOSSIER / "phase_relative.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "statut": "Analyse de phase relative, non calibree",
                "niveau_confiance": NIVEAU,
                "cycles_par_bloc": CYCLES_PAR_BLOC,
                "repetitions_bootstrap": REPETITIONS,
                "seed_bootstrap": SEED_BOOTSTRAP,
                "fenetres": rapport,
            },
            f,
            indent=2,
            allow_nan=False,
        )

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": "polar"},
    )

    tracer_phase_et_ic(
        ax,
        resultats[0],
        0.55,
        0.90,
        "tab:blue",
        "Fenetre 240-320",
    )
    tracer_phase_et_ic(
        ax,
        resultats[1],
        1.05,
        1.40,
        "tab:orange",
        "Fenetre 320-400",
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    angles = np.arange(0.0, 360.0, 45.0)
    etiquettes = [
        "0 deg", "45 deg", "90 deg", "135 deg",
        "180 deg", "-135 deg", "-90 deg", "-45 deg",
    ]
    ax.set_thetagrids(angles, labels=etiquettes)
    ax.set_ylim(0.0, 1.55)
    ax.set_yticks([0.725, 1.225])
    ax.set_yticklabels(["240-320", "320-400"])
    ax.set_title(
        "Phase relative E-I et IC bootstrap",
        pad=25.0,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        DOSSIER / "phase_relative_polaire.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nRapport : {DOSSIER / 'phase_relative.json'}")
    print(f"Figure : {DOSSIER / 'phase_relative_polaire.png'}")
    print(
        "L'IC est conditionnel a la frequence dominante, au filtrage "
        "et a la fenetre choisie."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
