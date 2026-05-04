# README — `main.py` (Orchestration FastAPI)

## Rôle de `main.py`

`main.py` est la couche d'orchestration qui connecte:

- le **retriever sémantique** (`SentenceTransformer` + FAISS),
- le **générateur mT5** (`mt5-bambara-resumer-boost1`),
- l'API FastAPI exposée aux clients.

Ce fichier transforme les modèles entraînés en service concret de question-réponse bilingue.

## Modèles et artefacts chargés

- Retriever: `Fatoumataa/embedding_billingue_francais_bambara_version3`
- Générateur: `Fatoumataa/mt5-bambara-resumer-boost1`
- Index local:
  - `embeddings.npy`
  - `ids_map.npy`
  - `parent_ids.npy`
  - `langs.npy`
  - `parents_map.json`

Le moteur FAISS utilise un produit interne (`IndexFlatIP`) avec normalisation L2 pour un scoring cosinus efficace.

## Logique d'inférence

1. Normaliser la question.
2. Encoder la question et interroger FAISS.
3. Appliquer un **keyword boosting** orienté parent (`+0.25`) lorsque des mots-clés spécifiques sont détectés.
4. Regrouper les enfants par document parent, conserver le meilleur score parent.
5. Prendre le meilleur contexte (ou fusionner plusieurs contextes si le second score est proche).
6. Construire l'entrée mT5:
   - `summarize_bm: question: ... contexte: ...`
   - `summarize_fr: question: ... contexte: ...`
7. Générer la réponse dans la langue cible.

## Stratégies de robustesse intégrées

- Validation des langues source/cible (`bm` ou `fr`) avec fallback.
- Seuil de confiance (`CONF_THRESHOLD=0.40`) pour éviter les réponses hasardeuses.
- Message de reformulation en cas de confiance faible.
- Endpoint de debug retrieval pour analyser scoring et effet du boosting.

## Paramètres de génération mT5

- `MAX_NEW_TOKENS=70`
- `MIN_NEW_TOKENS=15`
- `NUM_BEAMS=6`
- `REPETITION_PENALTY=1.5`
- `LENGTH_PENALTY=0.9`
- `no_repeat_ngram_size=3`
- `do_sample=False`

Ces choix favorisent des réponses plus stables, moins répétitives et mieux contrôlées.

## Endpoints disponibles

- `POST /ask`
  - Entrée: `question`, `langue_source`, `langue_cible`
  - Sortie: réponse, langue, source documentaire, confiance
- `GET /ask`
  - version compatible historique
- `GET /ask/cross`
  - version cross-lingue simplifiée
- `GET /debug/retrieval`
  - inspection des scores par parent/document

## Pourquoi cette orchestration est pertinente

- Elle matérialise un pipeline RAG complet dans une API simple à déployer.
- Elle exploite correctement le retrieval top-k avant génération.
- Elle supporte le cross-lingual (question BM -> réponse FR et inversement).
- Elle rend le système traçable (source + confiance), point essentiel en contexte éducatif.

## Lien avec les performances du projet

Cette API bénéficie directement:

- des performances retrieval (excellents rappels top-k),
- des gains de résumé mT5 (jusqu'à `RougeL ~ 0.372` après boost),
- de la stratégie de développement par phases adaptée aux contraintes GPU.

En résumé, `main.py` est la preuve d'industrialisation: les résultats de recherche sont transformés en service utilisable.

