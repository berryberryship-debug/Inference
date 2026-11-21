# Inférence IA en Python

Programme minimal pour charger un modèle de langage, lui poser une question et afficher sa réponse dans le terminal.

Ce projet utilise `Qwen/Qwen2.5-0.5B-Instruct` avec PyTorch et Hugging Face Transformers.

## Fonctionnalités

- Questions et réponses depuis le terminal.
- Consigne de réponse en français.
- Utilisation de CUDA si disponible, sinon du CPU.
- Limite de 256 nouveaux tokens par réponse.
- Chargement du modèle une seule fois par session.

## Structure du projet

```text
inference-ia/
├── README.md
├── requirements.txt
├── .gitignore
└── inference.py
```

## Prérequis

- Python et pip.
- Une connexion Internet pour télécharger les dépendances et les fichiers du modèle.
- Un environnement compatible avec PyTorch.

## Installation

Depuis le dossier du projet, créez un environnement virtuel :

```bash
python -m venv .venv
```

### Activer l’environnement sous Windows

Dans PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

### Activer l’environnement sous Linux ou macOS

```bash
source .venv/bin/activate
```

### Installer les dépendances

Le fichier `requirements.txt` doit contenir :

```text
torch
transformers
```

Installez ensuite les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Utilisation

Placez le code du programme dans `inference.py`, puis lancez :

```bash
python inference.py
```

Saisissez votre question lorsque le programme indique qu’il est prêt :

```text
Modèle prêt. Tape 'quitter' pour terminer.

Toi : Explique la différence entre une pile et une file.
```

Pour terminer la session, saisissez :

```text
quitter
```

Les commandes `exit` et `quit` sont également acceptées.

## Fonctionnement

```text
Question de l’utilisateur
          ↓
Mise en forme des messages
          ↓
Tokenisation
          ↓
Génération par le modèle
          ↓
Décodage des nouveaux tokens
          ↓
Affichage de la réponse
```

Le modèle reste chargé pendant toute la session. Chaque question est traitée indépendamment : cette version ne conserve pas l’historique de conversation.

## Configuration

Les paramètres suivants se trouvent dans `inference.py`.

### Modèle utilisé

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

### Longueur maximale de génération

```python
max_new_tokens=256
```

Cette limite concerne les nouveaux tokens générés, pas un nombre de mots.

### Mode de génération

```python
do_sample=False
```

### Consigne système

```python
"Tu es un assistant utile. Réponds en français."
```

### Périphérique de calcul

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
---

## Mise à jour finale : extension expérimentale du dépôt

*Note ajoutée en dernier pour documenter l'évolution du dépôt au-delà du script initial `inference.py`.*

En complément du programme minimal d'inférence interactive (`inference.py`), le dépôt regroupe l'ensemble des scripts expérimentaux suivants :

- **Instrumentation d'inférence LLM (`trajectoire_inference.py`, `trajectoires.py`) :** extraction pas-à-pas des temps de pré-remplissage (*TTFT*) et de décodage (*ITL* / *TPOT*), mesure de l'empreinte mémoire du *KV Cache* et suivi de la norme $L_2$ et de la dérive des états cachés de la dernière couche.
- **Dynamique neuronale $E/I$ et stabilité (`stabilite_neuronale.py`, `dynamique_neuronale.py`, `experience_longue.py`, `simulation_tres_longue.py`, `robustesse_regime.py`, `experience_repetitions.py`) :** intégration numérique par schéma de Runge-Kutta d'ordre 4 (RK4) sur graphe spatial 3D et étude des régimes oscillatoires et de leur stabilité locale.
- **Analyse spectrale et phase relative (`analyse_derives_longue.py`, `analyse_oscillations.py`, `analyse_phase_relative.py`, `verification_phase_longue.py`) :** filtrage passe-bande, extraction du signal analytique par transformée de Hilbert et suivi de la dérive de phase relative $E-I$.
- **Champs d'orientation (`experience_orientation.py`, `experience_orientation_localisee.py`, `experience_orientation_multisources.py`, `experience_orientation_sensibilite.py`, `experience_orientation_sensibilite_inverse.py`, `comparer_champ_orientation.py`) :** étude de la propagation directionnelle d'impulsions localisées sous couplage spatial anisotrope.
- **Géométrie futur-passé (`prototype_futur_passe.py`, `joindre_futur_passe_metric.py`) :** projection MDS régularisée d'ensembles de branches futures, calcul des volumes de covariance, bootstrap par blocs, tests de permutation et export vérifié par empreintes SHA-256.

**Dépendances supplémentaires (`requirements.txt`) :** `numpy`, `scipy`, `matplotlib` (en plus de `torch` et `transformers`).
