cd ~

cat > analyse_derives_longue.py <<'PYTHON'
#!/usr/bin/env python3
"""Diagnostics de derive sur huit fenetres tardives d'une trajectoire longue."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, hilbert, periodogram, sosfiltfilt
from scipy.stats import linregress
from scipy.stats import t as loi_t

FICHIER_TRAJECTOIRE = Path(
    "resultats/simulation_tres_longue/trajectoire_tres_longue.npz"
)
DOSSIER = Path("resultats/analyse_derives_longue")

BORNES_TARDIVES = [
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 500.0),
    (500.0, 600.0),
    (600.0, 700.0),
    (700.0, 800.0),
    (800.0, 900.0),
    (900.0, 1000.0),
]

NIVEAU = 0.95


def temps_pics_paraboliques(temps, signal):
    """Pics echantillonnes, avec interpolation parabolique de leur instant."""
    if signal.size < 3:
        return np.array([], dtype=float)

    indices = np.flatnonzero(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
    ) + 1

    if indices.size == 0:
        return np.array([], dtype=float)

    instants = []
    pas = float(temps[1] - temps[0])

    for indice in indices:
        if indice == 0 or indice == signal.size - 1:
            instants.append(float(temps[indice]))
            continue

        y_gauche = signal[indice - 1]
        y_centre = signal[indice]
        y_droite = signal[indice + 1]
        denominateur = y_gauche - 2.0 * y_centre + y_droite

        if abs(denominateur) < 1e-30:
            decalage = 0.0
        else:
            decalage = 0.5 * (y_gauche - y_droite) / denominateur
            decalage = float(np.clip(decalage, -0.5, 0.5))

        instants.append(float(temps[indice] + decalage * pas))

    return np.asarray(instants, dtype=float)


def phase_relative_fenetre(temps, E, I, debut, fin):
    """Phase relative E-I par Hilbert, avec filtrage et exclusion des bords."""
    masque = (temps >= debut) & (temps <= fin)
    t = temps[masque]

    if t.size < 20:
        raise ValueError("Fenetre trop courte.")

    x = E[masque].mean(axis=1)
    y = I[masque].mean(axis=1)

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

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
    vecteur = np.mean(np.exp(1j * delta))

    return {
        "frequence_dominante": f0,
        "phase_rad": float(np.angle(vecteur)),
        "concentration_R": float(abs(vecteur)),
    }


def tendance_lineaire(x, y, niveau=NIVEAU):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        raise ValueError("Au moins trois fenetres sont necessaires.")

    ajustement = linregress(x, y)
    ddl = len(x) - 2
    alpha = 1.0 - niveau
    quantile = float(loi_t.ppf(1.0 - alpha / 2.0, ddl))
    marge = quantile * float(ajustement.stderr)

    return {
        "pente": float(ajustement.slope),
        "ic_bas": float(ajustement.slope - marge),
        "ic_haut": float(ajustement.slope + marge),
        "p_value": float(ajustement.pvalue),
        "r2": float(ajustement.rvalue**2),
    }


def diagnostics_fenetres(temps, E, I, bornes):
    diagnostics = []

    for debut, fin in bornes:
        masque = (temps >= debut) & (temps <= fin)
        t = temps[masque]

        moyenne_E = E[masque].mean(axis=1)
        moyenne_I = I[masque].mean(axis=1)

        pics_E = temps_pics_paraboliques(t, moyenne_E)
        pics_I = temps_pics_paraboliques(t, moyenne_I)

        if pics_E.size < 3 or pics_I.size < 3:
            raise RuntimeError(
                f"Pas assez de pics dans [{debut}, {fin}]."
            )

        periodes_E = np.diff(pics_E)
        periodes_I = np.diff(pics_I)

        phase = phase_relative_fenetre(temps, E, I, debut, fin)

        diagnostics.append(
            {
                "debut": float(debut),
                "fin": float(fin),
                "centre": float(0.5 * (debut + fin)),
                "amplitude_E": float(np.ptp(moyenne_E)),
                "amplitude_I": float(np.ptp(moyenne_I)),
                "periode_E": float(periodes_E.mean()),
                "periode_I": float(periodes_I.mean()),
                "phase_deg": float(np.degrees(phase["phase_rad"])),
                "R": phase["concentration_R"],
            }
        )

    phases_rad = np.radians(
        [diagnostic["phase_deg"] for diagnostic in diagnostics]
    )
    phases_depliees_deg = np.degrees(np.unwrap(phases_rad))

    for diagnostic, phase_depliee in zip(
        diagnostics, phases_depliees_deg
    ):
        diagnostic["phase_depliee_deg"] = float(phase_depliee)

    return diagnostics


def analyser_derives(diagnostics):
    centres = np.array(
        [diagnostic["centre"] for diagnostic in diagnostics],
        dtype=float,
    )

    series = {
        "amplitude_E": np.array(
            [d["amplitude_E"] for d in diagnostics], dtype=float
        ),
        "amplitude_I": np.array(
            [d["amplitude_I"] for d in diagnostics], dtype=float
        ),
        "periode_E": np.array(
            [d["periode_E"] for d in diagnostics], dtype=float
        ),
        "periode_I": np.array(
            [d["periode_I"] for d in diagnostics], dtype=float
        ),
        "phase_depliee_deg": np.array(
            [d["phase_depliee_deg"] for d in diagnostics], dtype=float
        ),
    }

    resultats = {}

    for nom, valeurs in series.items():
        tendance = tendance_lineaire(centres, valeurs)
        moyenne = float(valeurs.mean())
        excursion_relative = float(
            (valeurs.max() - valeurs.min()) / abs(moyenne)
        )

        resultats[nom] = {
            **tendance,
            "moyenne": moyenne,
            "excursion_relative": excursion_relative,
        }

    return resultats


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    if not FICHIER_TRAJECTOIRE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_TRAJECTOIRE}. "
            "Lance d'abord simulation_tres_longue.py."
        )

    with np.load(FICHIER_TRAJECTOIRE, allow_pickle=False) as donnees:
        temps = donnees["temps"]
        E = donnees["E"]
        I = donnees["I"]

    if E.shape != I.shape or E.shape[0] != len(temps):
        raise ValueError("Dimensions incoherentes dans la trajectoire.")

    print("=== Diagnostics de derive, huit fenetres tardives ===")
    print(f"Duree totale : {temps[-1]:.1f}")

    diagnostics = diagnostics_fenetres(
        temps,
        E,
        I,
        BORNES_TARDIVES,
    )

    for diagnostic in diagnostics:
        print(
            f"\nFenetre [{diagnostic['debut']:g}, {diagnostic['fin']:g}]"
        )
        print(
            f"  Amplitude E / I : "
            f"{diagnostic['amplitude_E']:.8g} / "
            f"{diagnostic['amplitude_I']:.8g}"
        )
        print(
            f"  Periode E / I : "
            f"{diagnostic['periode_E']:.8g} / "
            f"{diagnostic['periode_I']:.8g}"
        )
        print(
            f"  Phase relative : {diagnostic['phase_deg']:.4f} deg ; "
            f"R = {diagnostic['R']:.5f}"
        )

    derives = analyser_derives(diagnostics)

    print("\n=== Tendances lineaires ===")

    for nom, resultat in derives.items():
        print(f"\n{nom}")
        print(f"  moyenne : {resultat['moyenne']:.8g}")
        print(
            f"  pente : {resultat['pente']:.8g} ; "
            f"IC95% : [{resultat['ic_bas']:.8g}, "
            f"{resultat['ic_haut']:.8g}]"
        )
        print(f"  p : {resultat['p_value']:.5g}")
        print(
            f"  excursion relative : "
            f"{100.0 * resultat['excursion_relative']:.5g}%"
        )

    rapport = {
        "statut": "Diagnostics de derive sur une trajectoire longue unique",
        "fichier_trajectoire": str(FICHIER_TRAJECTOIRE),
        "fenetres": diagnostics,
        "tendances": derives,
        "limites": [
            "Huit fenetres de 100 unites de temps.",
            "Environ neuf points par cycle a l'enregistrement.",
            "L'absence de significativite ne prouve pas une absence de derive.",
            "Aucune marge d'equivalence predefinie n'est imposee ici.",
            "La phase depend du filtrage et de la frequence dominante.",
        ],
    }

    with (DOSSIER / "derives.json").open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print(f"\nRapport enregistre : {DOSSIER / 'derives.json'}")
    print(
        "Une phase stable doit montrer une pente de phase depliee "
        "proche de zero."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
