# README — `fine_turning_bambara_français_2.ipynb`

## Rôle de cette phase

Ce notebook finalise le fine-tuning avec une logique de **boost chirurgical** à partir du modèle Phase 1.  
C'est la couche qui rapproche le système d'un niveau démontrable pour usage API et présentation recruteur.

## Point de départ et objectif

- Modèle source: `Fatoumataa/mt5-bambara-resumer-final`
- Modèle cible: `Fatoumataa/mt5-bambara-resumer-boost1`
- Objectif: améliorer la qualité de résumé bilingue tout en conservant la stabilité d'entraînement.

## Stratégie d'entraînement

Configuration explicitement documentée:

- `learning_rate=2e-5`
- `num_train_epochs=8`
- `per_device_train_batch_size=2`
- `per_device_eval_batch_size=2`
- `eval_strategy="epoch"`
- `save_strategy="epoch"`
- `hub_strategy="checkpoint"` (push des checkpoints à chaque epoch)
- `load_best_model_at_end=True`
- `metric_for_best_model="rougeL"`

Métriques:

- Calcul ROUGE avec pipeline sécurisé:
  - correction des labels `-100`,
  - filtrage des ids hors vocab avant décodage.

## Gestion de continuité expérimentale

Le notebook inclut une **reprise forcée depuis checkpoint**:

- message affiché: `✅ REPRISE FORCÉE DEPUIS : ./mt5-bambara-resumer-boost1/last-checkpoint`

Cette pratique est essentielle pour garantir:

- continuité des runs longs,
- meilleure traçabilité,
- réduction du coût de ré-entraînement.

## Résultats et preuves qualitatives

Le notebook contient des tests détaillés sur exemples FR/BM, avec comparaison `ATTENDU (HUMAIN)` vs `GÉNÉRÉ (IA)`.

Exemple observé (FR, agriculture à forte valeur ajoutée):

- le résumé généré récupère bien les axes principaux:
  - valeur stratégique/économique,
  - risque de fraude/contrefaçon.

Autres validations visibles:

- batterie de 8 exemples,
- tests sur 10 exemples inédits,
- tests sur 10 exemples de niveau élevé.

Ce protocole montre une volonté de vérifier la généralisation au-delà du set de training immédiat.

## Artefact final publié

Push validé sur Hugging Face:

- Repo: `Fatoumataa/mt5-bambara-resumer-boost1`
- Commit:
  - message: **"Phase 1 Boost - Final Stable Version"**
  - hash: `1bb7b93eb35950babe031720987a13517b5c5572`

## Pourquoi cette phase est crédible en entretien

- Elle ne se limite pas à "entraîner plus longtemps"; elle applique une stratégie contrôlée (checkpointing, best model, reprise).
- Elle documente des sorties réelles et comparées à une référence humaine.
- Elle aboutit à un artefact versionné et publiquement traçable (Hub + commit).

## Limites à expliciter

- Les preuves chiffrées consolidées (tableau ROUGE final complet) gagneraient à être exportées dans un rapport séparé.
- L'évaluation est déjà riche qualitativement, mais une batterie quantitative standardisée supplémentaire renforcerait encore la défense académique.

## Cible FastAPI (modèle de production)

Ce checkpoint est le candidat naturel pour:

- `POST /summarize`
  - entrées: `question`, `contexte`, `langue`
  - sortie: résumé final

Avec orchestration future:

- récupération contextuelle (RAG) via la couche sémantique,
- génération finale via `mt5-bambara-resumer-boost1`.

