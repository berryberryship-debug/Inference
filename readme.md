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
