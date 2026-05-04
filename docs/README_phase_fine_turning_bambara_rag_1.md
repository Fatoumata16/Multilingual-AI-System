# README — `phase_fine_turning_bambara_rag_1.ipynb`

## Position dans le pipeline

Cette phase correspond au **vrai démarrage du fine-tuning supervisé** pour la tâche de résumé.  
Le modèle de départ est le checkpoint Phase 0: `Fatoumataa/mt5-bambara-phase0-pro`.

## Objectif

Transformer un mT5 préparé sur corpus bilingue en un modèle capable de résumer des entrées structurées en français et en bambara.

## Configuration d'entraînement (explicite dans le notebook)

- Modèle source: `Fatoumataa/mt5-bambara-phase0-pro`
- Sortie visée: `Fatoumataa/mt5-bambara-resumer-final`
- Reproductibilité: seed fixé
- Split dataset: train/test `0.9/0.1`
- Génération encadrée (`GenerationConfig`) avec gestion explicite des tokens de début/fin/padding
- Métrique principale: `ROUGE` via `evaluate`
- Sélection du meilleur modèle: `metric_for_best_model="rougeL"`

Hyperparamètres clés:

- `learning_rate=1e-5` (abaissement pour réduire l'instabilité)
- `num_train_epochs=8`
- `per_device_train_batch_size=2`
- `per_device_eval_batch_size=2`
- `eval_strategy="epoch"`
- `save_strategy="epoch"`
- `load_best_model_at_end=True`

## Stratégie de robustesse implémentée

- Pipeline de métriques "sécurisé":
  - nettoyage des labels `-100`,
  - clipping des ids invalides hors vocabulaire avant décodage.
- Gestion robuste des checkpoints:
  - détection/reprise automatique,
  - logique de "reprise exacte" documentée.

Cette approche est scientifiquement saine car elle limite les crashs silencieux et rend les runs traçables.

## Observations expérimentales importantes

Le notebook montre des tentatives successives avec:

- lancement initial,
- puis reprise après incident d'entraînement (`trainer.train(...)` avec traceback visible),
- puis pipeline de reprise renforcée.

Ce point est crucial en contexte réel: la qualité finale dépend autant de la stratégie de reprise que des hyperparamètres.

## Artefact ciblé

- Repo final visé: `Fatoumataa/mt5-bambara-resumer-final`
- Message de push utilisé: **"Phase 1 finale - Résumé bambara stabilisé"**
- Message de reprise également documenté: **"Phase 1 - Reprise réussie vers Epoch 8"**

## Valeur ajoutée de la Phase 1

- Passage d'un modèle "préparé linguistiquement" à un modèle "orienté tâche résumé".
- Stabilisation de l'entraînement (LR réduit + checkpoints + best model by ROUGE-L).
- Mise en place d'une base crédible pour le boost final de performance (phase suivante).

## Limites explicites

- Le notebook ne consolide pas dans une table unique tous les scores ROUGE finaux par epoch.
- Les incidents d'exécution montrent que l'environnement d'entraînement reste un facteur de variance.

Ces limites sont normales dans une phase R&D; la valeur est la rigueur de reprise et la traçabilité.

## Intégration FastAPI (couche modèle intermédiaire)

Ce checkpoint est adapté pour:

- endpoint interne `POST /summarize-v1`
- A/B testing contre le modèle boosté de phase 2
- fallback opérationnel si le modèle final est indisponible

