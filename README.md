# ID By Rivoli - Audio Separator

Application web de séparation audio et création d'éditions DJ.

## Fonctionnalités

- **Séparation vocale/instrumentale** via Demucs (IA)
- **Génération d'éditions DJ** : Clap In, Acap In, Extended, Short, Slam...
- **Export MP3 & WAV** avec métadonnées complètes
- **Téléchargement ZIP** de tous les fichiers traités
- **Envoi automatique** des métadonnées vers l'API ID By Rivoli

## Installation sur RunPod

### 1. Cloner le repository

```bash
git clone https://github.com/EytanTeachr/IDByRivoli-separate-audio.git
cd IDByRivoli-separate-audio
```

### 2. Installer les dépendances système

```bash
apt-get update && apt-get install -y ffmpeg
```

### 3. Installer les dépendances Python

```bash
pip install --ignore-installed -r requirements.txt
```

### 4. Lancer l'application

```bash
python app.py
```

L'application sera accessible sur le port **8888** :
`https://[votre-pod-id]-8888.proxy.runpod.net/`

## Commandes utiles

### Mettre à jour vers la dernière version

```bash
cd IDByRivoli-separate-audio
git pull
python app.py
```

### Si le port 8888 est déjà utilisé

```bash
# Trouver et tuer le processus qui utilise le port
lsof -i :8888 | awk 'NR>1 {print $2}' | xargs kill -9

# Ou relancer avec un port différent (modifier app.py)
# Changer la ligne : app.run(host='0.0.0.0', port=8888, debug=True)
# Par exemple : app.run(host='0.0.0.0', port=8889, debug=True)
```

### Lancer en arrière-plan

```bash
nohup python app.py > app.log 2>&1 &
```

## Structure des fichiers

```
IDByRivoli-separate-audio/
├── app.py                 # Application Flask principale
├── audio_processor.py     # Logique de traitement audio
├── templates/
│   └── index.html         # Interface web
├── static/
│   ├── covers/            # Pochettes
│   └── fonts/             # Police ClashGrotesk
├── uploads/               # Fichiers uploadés
├── output/                # Fichiers Demucs (vocals/instrumental)
├── processed/             # Fichiers finaux (éditions)
└── assets/                # Ressources (clap sample, covers)
```

## Genres et éditions

- **House, Electro House, Dance** : Export uniquement en version "Main" (MP3 + WAV)
- **Autres genres** : Suite complète d'éditions DJ (Clap In, Acap In, Extended, etc.)

## Configuration

Les variables d'environnement suivantes peuvent être configurées :

- `PUBLIC_URL` : URL publique du pod pour les liens de téléchargement
- `API_KEY` : Clé d'authentification pour l'API ID By Rivoli

## Requirements

- Python 3.8+
- FFmpeg
- ~4GB RAM minimum (pour Demucs)
- GPU recommandé pour un traitement plus rapide
