cd ~
cp -p robustesse_regime.py robustesse_regime.py.avant_version_corrigee.bak 2>/dev/null || true

cat > robustesse_regime.py <<'PYTHON'
#!/usr/bin/env python3
"""Robustesse locale du regime a w_EE = 8, perturbations centrees sur 0.5."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, hilbert, periodogram, sosfiltfilt

import stabilite_neuronale as modele

W_EE = 8.0
DUREE = 240.0
DT = 0.05
INTERVALLE_ENREGISTREMENT = 10

GRAINES = [11, 22, 33, 44, 55]
AMPLITUDES = [1e-4, 1e-3, 1e-2, 1e-1]
FENETRE_ANALYSE = (160.0, 240.0)

DOSSIER = Path("resultats/robustesse_regime")


def construire_rhs(L, w_EE):
    P, Q = modele.entrees(w_EE)

    def rhs(y):
        E, I = y
        dE = (
            -E
            + modele.sigmoid(w_EE * E - modele.W_EI * I + P)
            - modele.D_E * (L @ E)
        ) / modele.TAU_E
        dI = (
            -I
            + modele.sigmoid(modele.W_IE * E - modele.W_II * I + Q)
            - modele.D_I * (L @ I)
        ) / modele.TAU_I
        return np.array([dE, dI])

    return rhs


def simuler(L, etat_initial):
    rhs = construire_rhs(L, W_EE)
    nombre_pas = int(round(DUREE / DT))
    y = etat_initial.copy()

    temps = [0.0]
    E = [y[0].copy()]
    I = [y[1].copy()]

    for numero_pas in range(nombre_pas):
        a = rhs(y)
        b = rhs(y + DT * a / 2.0)
        c = rhs(y + DT * b / 2.0)
        d = rhs(y + DT * c)
        y = y + DT * (a + 2.0 * b + 2.0 * c + d) / 6.0

        if not np.all(np.isfinite(y)):
            raise RuntimeError("Etat non fini.")

        if np.any(y < -1e-9) or np.any(y > 1.0 + 1e-9):
            raise RuntimeError("Etat hors de [0, 1].")

        if (numero_pas + 1) % INTERVALLE_ENREGISTREMENT == 0:
            temps.append((numero_pas + 1) * DT)
            E.append(y[0].copy())
            I.append(y[1].copy())

    return np.asarray(temps), np.asarray(E), np.asarray(I)


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


def filtrer_autour_dominante(signal, fs):
    """Filtre passe-bande etroit autour de la frequence dominante."""
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
    """Periode par regression lineaire de la phase de Hilbert depliee."""
    phase = np.unwrap(np.angle(hilbert(signal_filtre)))
    pente, _ = np.polyfit(temps, phase, 1)
    frequence = float(pente / (2.0 * np.pi))

    if frequence <= 0.0:
        raise ValueError("Frequence estimee non positive.")

    return 1.0 / frequence, frequence


def mesures_pics(temps, E, I, debut, fin):
    masque = (temps >= debut) & (temps <= fin)
    t = temps[masque]
    moyenne_E = E[masque].mean(axis=1)
    moyenne_I = I[masque].mean(axis=1)

    instants_E = temps_pics_paraboliques(t, moyenne_E)
    instants_I = temps_pics_paraboliques(t, moyenne_I)

    if instants_E.size < 3 or instants_I.size < 3:
        raise RuntimeError("Pas assez de pics pour estimer une periode.")

    intervalles_E = np.diff(instants_E)
    intervalles_I = np.diff(instants_I)

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    E_filtre, _ = filtrer_autour_dominante(moyenne_E, fs)
    I_filtre, _ = filtrer_autour_dominante(moyenne_I, fs)

    periode_E_phase, frequence_E_phase = periode_par_phase_depliee(
        t, E_filtre
    )
    periode_I_phase, frequence_I_phase = periode_par_phase_depliee(
        t, I_filtre
    )

    return {
        "amplitude_E": float(np.ptp(moyenne_E)),
        "amplitude_I": float(np.ptp(moyenne_I)),
        "intervalle_moyen_E": float(intervalles_E.mean()),
        "intervalle_moyen_I": float(intervalles_I.mean()),
        "ecart_type_intervalle_E": float(intervalles_E.std()),
        "ecart_type_intervalle_I": float(intervalles_I.std()),
        "nombre_pics_E": int(instants_E.size),
        "nombre_pics_I": int(instants_I.size),
        "periode_phase_E": periode_E_phase,
        "periode_phase_I": periode_I_phase,
        "frequence_phase_E": frequence_E_phase,
        "frequence_phase_I": frequence_I_phase,
    }


def phase_relative_fenetre(temps, E, I, debut, fin):
    """Phase relative E-I par Hilbert, moyenne circulaire et bootstrap."""
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


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)

    print("=== Robustesse locale, w_EE = 8 ===")
    print("Perturbations centrees sur 0.5.")
    print(f"Graines : {GRAINES}")
    print(f"Amplitudes : {AMPLITUDES}")
    print(f"Duree : {DUREE} ; fenetre d'analyse : {FENETRE_ANALYSE}")

    rng = np.random.default_rng(modele.SEED)
    points = rng.uniform(0.0, 1.0, size=(modele.N, 3))
    adjacency = modele.construire_graphe(points)
    L = modele.laplacien(adjacency)

    resultats = []

    for amplitude in AMPLITUDES:
        for graine in GRAINES:
            etiquette = f"A={amplitude:g}, graine={graine}"
            print(f"\n{etiquette}")

            rng_perturbation = np.random.default_rng(graine)
            etat_initial = 0.5 + amplitude * rng_perturbation.normal(
                size=(2, modele.N)
            )

            temps, E, I = simuler(L, etat_initial)

            suffixe = f"A{amplitude:.0e}_graine_{graine}"
            np.savez_compressed(
                DOSSIER / f"trajectoire_{suffixe}.npz",
                temps=temps,
                E=E,
                I=I,
            )

            pics = mesures_pics(
                temps, E, I, FENETRE_ANALYSE[0], FENETRE_ANALYSE[1]
            )
            phase = phase_relative_fenetre(
                temps, E, I, FENETRE_ANALYSE[0], FENETRE_ANALYSE[1]
            )

            resultat = {
                "amplitude": amplitude,
                "graine": graine,
                **pics,
                "frequence_dominante": phase["frequence_dominante"],
                "phase_relative_deg": float(
                    np.degrees(phase["phase_rad"])
                ),
                "concentration_R": phase["concentration_R"],
            }
            resultats.append(resultat)

            print(
                f"  Amplitude E / I : "
                f"{pics['amplitude_E']:.6g} / {pics['amplitude_I']:.6g}"
            )
            print(
                f"  Periode pics E / I : "
                f"{pics['intervalle_moyen_E']:.6g} / "
                f"{pics['intervalle_moyen_I']:.6g}"
            )
            print(
                f"  Periode phase E / I : "
                f"{pics['periode_phase_E']:.6g} / "
                f"{pics['periode_phase_I']:.6g}"
            )
            print(
                f"  Phase E-I : {resultat['phase_relative_deg']:.2f} deg ; "
                f"R = {resultat['concentration_R']:.3f}"
            )

    amplitudes_E = np.array(
        [r["amplitude_E"] for r in resultats], dtype=float
    )
    amplitudes_I = np.array(
        [r["amplitude_I"] for r in resultats], dtype=float
    )
    periodes_E = np.array(
        [r["periode_phase_E"] for r in resultats], dtype=float
    )
    periodes_I = np.array(
        [r["periode_phase_I"] for r in resultats], dtype=float
    )
    phases = np.array(
        [
            np.exp(1j * np.radians(r["phase_relative_deg"]))
            for r in resultats
        ]
    )

    phase_moyenne = float(np.angle(phases.mean()))
    concentration_globale = float(abs(phases.mean()))

    resume = {
        "statut": "Robustesse locale synthetique, non calibree",
        "w_EE": W_EE,
        "duree": DUREE,
        "dt": DT,
        "fenetre_analyse": list(FENETRE_ANALYSE),
        "graines": GRAINES,
        "amplitudes": AMPLITUDES,
        "perturbation": "Gaussienne centree, amplitude controlee",
        "amplitude_E_moyenne": float(amplitudes_E.mean()),
        "amplitude_E_ecart_type": float(amplitudes_E.std()),
        "amplitude_I_moyenne": float(amplitudes_I.mean()),
        "amplitude_I_ecart_type": float(amplitudes_I.std()),
        "periode_phase_E_moyenne": float(periodes_E.mean()),
        "periode_phase_E_ecart_type": float(periodes_E.std()),
        "periode_phase_I_moyenne": float(periodes_I.mean()),
        "periode_phase_I_ecart_type": float(periodes_I.std()),
        "phase_moyenne_deg": float(np.degrees(phase_moyenne)),
        "concentration_phase_globale": concentration_globale,
        "limites": [
            "Robustesse locale autour de l'etat uniforme 0.5.",
            "Vingt trajectoires : cinq graines par amplitude.",
            "La concentration globale n'est pas un intervalle de confiance.",
            "La periode par phase depend du filtrage et de la fenetre.",
        ],
        "resultats": resultats,
    }

    with (DOSSIER / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, allow_nan=False)

    with (DOSSIER / "resultats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "amplitude",
                "graine",
                "amplitude_E",
                "amplitude_I",
                "intervalle_moyen_E",
                "intervalle_moyen_I",
                "periode_pics_E",
                "periode_pics_I",
                "periode_phase_E",
                "periode_phase_I",
                "phase_relative_deg",
                "retard_I_sur_E",
                "concentration_R",
            ]
        )
        for r in resultats:
            writer.writerow(
                [
                    r["amplitude"],
                    r["graine"],
                    r["amplitude_E"],
                    r["amplitude_I"],
                    r["intervalle_moyen_E"],
                    r["intervalle_moyen_I"],
                    r["intervalle_moyen_E"],
                    r["intervalle_moyen_I"],
                    r["periode_phase_E"],
                    r["periode_phase_I"],
                    r["phase_relative_deg"],
                    r["phase_relative_deg"] / 360.0
                    * r["periode_phase_E"],
                    r["concentration_R"],
                ]
            )

    print("\n=== Resume global ===")
    print(
        f"Amplitude E : {amplitudes_E.mean():.6g} "
        f"+/- {amplitudes_E.std():.3g}"
    )
    print(
        f"Amplitude I : {amplitudes_I.mean():.6g} "
        f"+/- {amplitudes_I.std():.3g}"
    )
    print(
        f"Periode phase E : {periodes_E.mean():.6g} "
        f"+/- {periodes_E.std():.3g}"
    )
    print(
        f"Periode phase I : {periodes_I.mean():.6g} "
        f"+/- {periodes_I.std():.3g}"
    )
    print(
        f"Phase moyenne E-I : {np.degrees(phase_moyenne):.2f} deg ; "
        f"concentration globale = {concentration_globale:.3f}"
    )
    print(f"Resultats : {DOSSIER}")
    print(
        "Cette experience teste une robustesse locale ; elle ne caracterise "
        "pas l'ensemble du bassin d'attraction."
    )
    print("Aucune interpretation clinique ou de conscience.")


if __name__ == "__main__":
    main()
PYTHON
