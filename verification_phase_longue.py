cd ~

cat > verification_phase_longue.py <<'PYTHON'
#!/usr/bin/env python3
"""Verification de la derive de phase relative sur 200-1000."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, hilbert, periodogram, sosfiltfilt
from scipy.stats import linregress

FICHIER_TRAJECTOIRE = Path(
    "resultats/simulation_tres_longue/trajectoire_tres_longue.npz"
)
DOSSIER = Path("resultats/verification_phase_longue")

DEBUT = 200.0
FIN = 1000.0


def filtrer_autour_dominante(signal, fs):
    x = detrend(signal, type="linear")
    frequences, puissance = periodogram(x, fs=fs, window="hann")

    valide = (frequences > 0.0) & (frequences < 0.9 * fs / 2.0)
    f0 = float(frequences[valide][np.argmax(puissance[valide])])

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

    return sosfiltfilt(sos, x), f0


def periode_par_phase_depliee(temps, signal_filtre):
    phase = np.unwrap(np.angle(hilbert(signal_filtre)))
    pente, _ = np.polyfit(temps, phase, 1)
    frequence = float(pente / (2.0 * np.pi))

    if frequence <= 0.0:
        raise ValueError("Frequence estimee non positive.")

    return 1.0 / frequence, frequence


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    with np.load(FICHIER_TRAJECTOIRE, allow_pickle=False) as donnees:
        temps = donnees["temps"]
        E = donnees["E"]
        I = donnees["I"]

    masque = (temps >= DEBUT) & (temps <= FIN)
    t = temps[masque]
    moyenne_E = E[masque].mean(axis=1)
    moyenne_I = I[masque].mean(axis=1)

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    E_filtre, f0 = filtrer_autour_dominante(moyenne_E, fs)
    I_filtre, _ = filtrer_autour_dominante(moyenne_I, fs)

    phase_E = np.unwrap(np.angle(hilbert(E_filtre)))
    phase_I = np.unwrap(np.angle(hilbert(I_filtre)))

    delta_phi = phase_E - phase_I

    pente, _ = np.polyfit(t, delta_phi, 1)
    ajustement = linregress(t, delta_phi)

    periode_E, frequence_E = periode_par_phase_depliee(t, E_filtre)
    periode_I, frequence_I = periode_par_phase_depliee(t, I_filtre)

    pente_predite_rad = 2.0 * np.pi * (frequence_E - frequence_I)
    pente_predite_deg = float(np.degrees(pente_predite_rad))

    derive_totale_deg = float(pente * (t[-1] - t[0]))
    derive_predite_deg = float(pente_predite_deg * (t[-1] - t[0]))

    rapport = {
        "fenetre": [DEBUT, FIN],
        "frequence_dominante": f0,
        "periode_E_phase": periode_E,
        "periode_I_phase": periode_I,
        "frequence_E_phase": frequence_E,
        "frequence_I_phase": frequence_I,
        "pente_phase_relative_rad_par_temps": float(pente),
        "pente_phase_relative_deg_par_temps": float(
            np.degrees(pente)
        ),
        "p_value": float(ajustement.pvalue),
        "r2": float(ajustement.rvalue**2),
        "derive_totale_deg": derive_totale_deg,
        "pente_predite_deg_par_temps": pente_predite_deg,
        "derive_predite_deg": derive_predite_deg,
        "difference_pente_deg_par_temps": float(
            np.degrees(pente) - pente_predite_deg
        ),
    }

    with (DOSSIER / "verification_phase.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(rapport, f, indent=2, allow_nan=False)

    print("=== Verification de la derive de phase relative ===")
    print(f"Fenetre : [{DEBUT:g}, {FIN:g}]")
    print(f"Frequence dominante : {f0:.8g}")
    print(f"Periode phase E : {periode_E:.8g}")
    print(f"Periode phase I : {periode_I:.8g}")
    print(
        f"Pente phase relative : {np.degrees(pente):.8g} deg / unite de temps"
    )
    print(f"p : {ajustement.pvalue:.5g}")
    print(f"R2 : {ajustement.rvalue:.5g}")
    print(f"Derive totale observee : {derive_totale_deg:.5g} deg")
    print(
        f"Pente predite par difference de periodes : "
        f"{pente_predite_deg:.8g} deg / unite de temps"
    )
    print(f"Derive predite : {derive_predite_deg:.5g} deg")
    print(
        f"Difference observee - predite : "
        f"{np.degrees(pente) - pente_predite_deg:.8g} "
        "deg / unite de temps"
    )
    print(f"Rapport : {DOSSIER / 'verification_phase.json'}")
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
