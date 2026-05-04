# README — `phase0_rag.ipynb`

## Rôle de cette phase

`phase0_rag.ipynb` prépare la base linguistique et statistique du projet avant le fine-tuning de spécialisation.  
Il s'agit d'une phase de **pré-entraînement/alignement de domaine** sur corpus mixte bambara + français pour stabiliser mT5.

## Objectifs techniques

- Nettoyer les corpus sources BM/FR de manière reproductible.
- Contrôler la tokenisation bambara avec le tokenizer `google/mt5-small`.
- Construire un dataset final équilibré et mélangé pour éviter les biais d'ordre.
- Lancer un entraînement T5 span-corruption rigoureux avec collator dédié.

## Pipeline de données (documenté dans le notebook)

### 1) Ingestion et nettoyage

- Extraction des données compressées.
- Nettoyage ligne par ligne (normalisation + filtrage).
- Traitement des doublons avec règle explicite:
  - **pas de `set()` global** (qui détruit le signal fréquentiel),
  - plafonnement à **5 occurrences max** par segment pour préserver la statistique.

### 2) Contrôle tokenisation Bambara

- Tokenizer utilisé: `google/mt5-small`.
- Mesure sur échantillon:
  - **nombre moyen de tokens par phrase = 32.49**.
- Impact: confirme des séquences potentiellement longues et justifie le chunking choisi ensuite.

### 3) Fusion bilingue Phase 0

Résultat affiché dans le notebook:

- **Total phrases Bambara: 74 077**
- **Total phrases Français: 74 949**
- **Taille totale dataset Phase 0: 149 026**
- Fichier final: `dataset_final_phase0.txt`

Le mélange est aléatoire avec `seed` pour la reproductibilité.

## Stratégie d'entraînement mT5 (Phase 0)

- Modèle de base: `google/mt5-small`
- Collator custom de span corruption (sentinelles `<extra_id_x>`) dans une logique T5 stricte.
- Chunking: séquences de **256 tokens**.
- Split train/test: **90/10**.
- Hyperparamètres visibles:
  - `num_train_epochs=3`
  - `learning_rate=1e-4`
  - `gradient_accumulation_steps=2`
  - `load_best_model_at_end=True`

## Artefact produit

Le modèle Phase 0 est poussé sur Hugging Face:

- Repo: `Fatoumataa/mt5-bambara-phase0-pro`
- Commit de référence:
  - message: **"Phase 0: T5 Rigor Final (NFC, 5e-5, Correct Spans)"**
  - hash: `36d06ce4104de57761920aa607c6712e2aa8a51a`

## Pourquoi cette phase est pertinente

- Elle crée une base bilingue robuste avant la tâche finale de résumé.
- Elle limite les risques classiques:
  - bruit textuel,
  - effondrement de diversité dû au dédoublonnage agressif,
  - instabilité due à un pipeline non reproductible.
- Elle prépare directement la reprise en Phase 1 (fine-tuning orienté résumé).

## Limites et points de vigilance

- Cette phase optimise surtout la représentation et la stabilité, pas encore la qualité finale de résumé métier.
- Les métriques textuelles finales (ROUGE détaillé) sont traitées surtout dans les phases de fine-tuning suivantes.

## Projection API FastAPI

Cette phase alimente l'endpoint de génération via un modèle de base déjà adapté au domaine:

- `POST /summarize` (pré-version)
  - entrée: texte + langue
  - sortie: résumé brut

En production, ce modèle Phase 0 sert de checkpoint de départ, pas de modèle final utilisateur.

