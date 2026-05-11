# Mission : Déployez et monitorez votre modèle de scoring

## Context

Vous êtes Data Scientist dans l'entreprise **"Prêt à Dépenser"**. Après avoir développé et versionné un modèle de scoring (Projet MLOps Part 1/2), vous recevez un message Slack de **Chloé Dubois**, la Lead Data Scientist :

> Salut ! Excellents résultats sur la dernière version du modèle de scoring ! Le département 'Crédit Express' est très impatient de l'utiliser pour traiter les nouvelles demandes en quasi temps réel. Il nous faut absolument une API fonctionnelle et déployable (Docker Ready!) d'ici la fin de la semaine prochaine. Peux-tu prioriser ça ? On a aussi besoin d'un dashboard ou rapport de suivi pour vérifier que tout se passe bien une fois en prod (distribution des scores, temps de réponse, ce genre de choses). Tiens-moi au courant de ton plan d'action ! Merci !

**Mission** : Piloter la mise en production effective du modèle de scoring. Cela inclut :
- Création d'une API robuste
- Conteneurisation pour un déploiement fluide
- Mise en place d'un monitoring proactif
- Garantir la performance et la fiabilité du modèle dans le temps

---

## Livrables attendus

1. **Historique des versions**
   - Retracer la construction du projet via l'historique Git
   - Repository public sur GitHub contenant tous les commits

2. **Scripts et API**
   - API fonctionnelle (FastAPI ou Gradio) pour prédiction en temps réel
   - Entrée : données client
   - Sortie : score de prédiction
   - Tests unitaires automatisés

3. **Conteneurisation**
   - Dockerfile pour la conteneurisation du code

4. **Analyse du Data Drift**
   - Détection automatique des anomalies
   - Tableau de bord ou rapport de monitoring
   - Métriques clés : distribution des scores, latence API, temps d'inférence
   - Screenshots de la solution de stockage des données de production

5. **Pipeline CI/CD**
   - Fichier YAML pour l'automatisation
   - Automatisation de la mise en production et des tests
   - Déclenché à minima sur push vers la branche principale

6. **Documentation**
   - README expliquant comment lancer l'API
   - Guide d'interprétation du monitoring

---

## Ressources pédagogiques

- **Outils suggérés** : Streamlit et Gradio
- Réutiliser les artefacts du projet MLOps Part 1/2
- Adapter si nécessaire et construire l'environnement de déploiement
- Vous êtes libre d'utiliser d'autres outils si vous justifiez vos choix techniques

---

## Étapes

### Étape 1 : Contrôle de version et dépôt

**Description** : Initialisez un dépôt Git pour votre projet avec une structure claire.

**Prérequis** :
- Git installé
- Compte sur GitHub, GitLab ou Bitbucket

**Structure attendue** :
```
project/
├── src/                    # Code source principal
├── tests/                  # Tests unitaires
├── notebooks/              # Notebooks d'analyse
├── models/                 # Artefacts du modèle
├── Dockerfile              # Configuration Docker
├── pyproject.toml          # Dependencies
├── .gitignore              # Fichiers à ignorer
├── README.md               # Documentation
└── .github/workflows/      # Pipelines CI/CD
```

**Résultats attendus** :
- ✅ Dépôt Git public avec code structuré
- ✅ Historique de commits clair et pertinent
- ✅ Fichier `.gitignore` configuré

**Recommandations** :
- Utiliser des messages de commit explicites
- Adopter une stratégie de branche si nécessaire
- Ne jamais committer de données sensibles

**Outils** :
- Git
- GitHub / GitLab / Bitbucket

**Ressources** :
- [Documentation Git](https://git-scm.com/doc)
- [Quickstart GitHub](https://docs.github.com/en/get-started)
- [Cours - Gérez du code avec Git et GitHub](https://openclassrooms.com)

---

### Étape 2 : Déploiement API et CI/CD

**Description** : Développez une API pour exposer votre modèle, conteneurisez-la avec Docker et créez un pipeline CI/CD automatisé.

**Prérequis** :
- Code versionné sur une plateforme supportant CI/CD
- Framework d'API choisi (FastAPI / Gradio)
- Docker installé

**Fonctionnalités de l'API** :
- Recevoir des données d'entrée (Pydantic validation)
- Retourner une prédiction
- Gestion des erreurs robuste
- Documentation Swagger/OpenAPI

**Pipeline CI/CD automatisé** :
1. ✅ Exécuter les tests (unitaires, intégration)
2. ✅ Construire l'image Docker si tests concluants
3. ✅ Déployer l'image conteneurisée
4. ✅ Tests automatisés intégrés

**Résultats attendus** :
- ✅ Code source fonctionnel pour l'API
- ✅ Dockerfile créé et opérationnel
- ✅ Pipeline CI/CD fonctionnel et visible (GitHub Actions, etc.)
- ✅ Tests automatisés avec bonne couverture
- ✅ Secrets gérés de manière sécurisée

**Recommandations** :
- Commencer par une API simple et un pipeline basique, puis itérer
- Séparer les étapes : build → test → déploiement
- Utiliser des secrets pour les credentials
- Considérer [Hugging Face Spaces](https://huggingface.co/spaces) pour un déploiement simple

**Points de vigilance critiques** :

⚠️ **Chargement du modèle** :
- ❌ Ne JAMAIS charger le modèle à chaque requête
- ✅ Charger le modèle UNE SEULE FOIS au démarrage de l'API
- ✅ Réutiliser l'instance pour toutes les requêtes

Bénéfices :
- Réduire le temps de réponse
- Éviter une surcharge mémoire
- Améliorer la scalabilité

⚠️ **Validation des entrées** :
- Données manquantes pour champs obligatoires
- Valeurs hors plages attendues (ex: âge négatif, revenu = 0)
- Types de données incorrects (texte vs nombre)

⚠️ **Sécurité** :
- Valider toutes les entrées utilisateur
- Gérer les secrets de manière sécurisée
- Vérifier les ressources disponibles en déploiement

**Outils** :
- FastAPI / Gradio
- Docker
- Pytest
- Postman / curl
- GitHub Actions / GitLab CI / Jenkins
- Plateformes : Hugging Face, Heroku, Google Cloud Run

**Ressources** :
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Gradio](https://gradio.app/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- Tutoriels pytest

---

### Étape 3 : Stockage et analyse des données de production

**Description** : Stockez les données générées par votre API et analysez-les pour détecter le data drift et les anomalies.

**Données à logger (minimum)** :
- Appels API (timestamp, utilisateur)
- Inputs du modèle
- Outputs du modèle
- Temps d'exécution
- Taux d'erreur

**Prérequis** :
- API déployée via CI/CD
- Données clés identifiées et loggées

**Résultats attendus** :
- ✅ Solution de stockage décrite et/ou implémentée
- ✅ Script ou notebook d'analyse automatique
- ✅ Détection de data drift
- ✅ Détection d'anomalies opérationnelles
- ✅ Présentation des résultats et vigilances

**Recommandations** :
- Configurer le logging structuré (JSON)
- Utiliser des bibliothèques dédiées : [Evidently AI](https://www.evidentlyai.com/), [NannyML](https://www.nannyml.com/)
- Visualiser les résultats (dashboard Streamlit/Dash)
- Stocker en local si pas de service cloud disponible

**Points de vigilance** :
- Gérer les contraintes de stockage et de coût
- Assurer la conformité RGPD si nécessaire
- La détection de drift nécessite une référence (données d'entraînement)

**Outils** :
- Logging Python (structlog, loguru)
- Bases de données : PostgreSQL, Elasticsearch
- Drift detection : Evidently AI, NannyML
- Visualisation : Grafana, Kibana, Streamlit, Dash

**Ressources** :
- [Article sur le monitoring ML en Python](https://docs.evidentlyai.com/)
- [Documentation Evidently](https://www.evidentlyai.com/documentation)

---

### Étape 4 : Optimisation des performances du modèle

**Description** : Analysez et optimisez les performances réelles du modèle en production.

**Prérequis** :
- API déployée
- Système de monitoring/logging en place

**Analyse requise** :
1. Identifier les goulots d'étranglement via le monitoring
2. Profiler le code (CPU, mémoire)
3. Tester des stratégies d'optimisation :
   - Quantification du modèle
   - Optimisation du code
   - Configuration hardware

**Résultats attendus** :
- ✅ Rapport détaillé des tests d'optimisation
- ✅ Résultats mesurables et goulots identifiés
- ✅ Version optimisée déployée via CI/CD
- ✅ Justification de la configuration finale
- ✅ Démonstration de l'amélioration (latence, inférence)

**Recommandations** :
- Baser les optimisations sur les données de monitoring réelles
- Documenter l'impact sur performance ET précision
- Avant/Après : comparer les métriques

**Points de vigilance** :
- ❌ Les optimisations ne doivent pas réduire la précision du modèle
- ❌ Vérifier l'absence de biais introduits
- ✅ Valider la compatibilité en production

**Outils** :
- cProfile, line_profiler (profiling)
- ONNX Runtime (optimisation)
- Scikit-learn, TensorFlow, PyTorch (quantification)

**Ressources** :
- [Documentation cProfile](https://docs.python.org/3/library/profile.html)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## Résumé de la mission

| Étape | Livrables | Priorité |
|-------|-----------|----------|
| 1 | Repository Git structuré | 🔴 Critique |
| 2 | API + Docker + CI/CD | 🔴 Critique |
| 3 | Stockage + Monitoring + Drift | 🟡 Haute |
| 4 | Optimisation + Rapports | 🟡 Haute |

**Échéance** : Fin de la semaine prochaine ⏰
