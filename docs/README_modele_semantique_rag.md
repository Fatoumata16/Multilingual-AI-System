# README — `modele_semantique_rag.ipynb`

## Objectif scientifique du notebook

Ce notebook construit et valide la couche **sémantique** du pipeline RAG bilingue (français-bambara) avant l'entraînement génératif.  
L'enjeu est de mesurer la capacité d'un encodeur à rapprocher une question de sa réponse pertinente et à éloigner une réponse incorrecte.

## Position dans l'architecture globale

Ce bloc intervient en amont du modèle de résumé:

1. Encodage des requêtes et passages (`SentenceTransformer`).
2. Calcul de similarité cosinus.
3. Sélection du socle d'embeddings le plus exploitable pour la suite RAG + génération.

Il sert de base pour la partie retrieval qui sera orchestrée ensuite dans FastAPI.

## Méthodologie mise en oeuvre

- **Familles de modèles testées**
  - `sentence-transformers/LaBSE`
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - variantes supplémentaires testées dans le notebook: `E5-small`, `Jina-v2`, `BGE-M3`, `Davlan/afro-xlmr-large`

- **Protocole d'évaluation utilisé**
  - Construction de triplets simples: *(question, bonne réponse, mauvaise réponse)*.
  - Calcul de cosinus entre l'embedding de la question et chacune des réponses.
  - Critère local: `sim(question, bonne_reponse) > sim(question, mauvaise_reponse)`.

- **Scénarios couverts**
  - Test direct sur texte.
  - Test via traduction intermédiaire vers le français pour les segments bambara.
  - Test monolingue de robustesse.
  - Début d'intégration vers un schéma de fine-tuning retrieval (early stopping visible dans le notebook).

## Résultats observés (issus des sorties du notebook)

### Exemples de scores affichés

- Cas initial (`LaBSE`):
  - bonne réponse: **0.0864**
  - mauvaise réponse: **0.1550**
  - interprétation: inversion du rang (mauvaise séparation).

- Comparatif `LaBSE` vs `MiniLM-multilingual`:
  - `LaBSE`: bonne **0.1936** / mauvaise **0.2411**
  - `MiniLM`: bonne **0.3674** / mauvaise **0.4519**
  - interprétation: les deux modèles échouent sur ce cas malgré un niveau absolu de similarité différent.

- Cas où séparation correcte apparaît:
  - bonne **0.3427** / mauvaise **0.2698**
  - bonne **0.4460** / mauvaise **0.3957**
  - bonne **0.6825** / mauvaise **0.6429**

- Cas de faible marge / ambiguité:
  - **0.8564** vs **0.8636**
  - **0.8456** vs **0.8559**
  - **0.7686** vs **0.8800**

## Interprétation scientifique

- Le notebook montre que la performance retrieval est **très dépendante du cas** et du modèle.
- Les scores élevés ne garantissent pas un bon classement: la métrique utile est la **marge relative** entre bonne et mauvaise réponse, pas la valeur absolue.
- Les tests mettent en évidence un vrai sujet de robustesse sur le bambara et les formulations proches, ce qui justifie:
  - un fine-tuning ciblé retrieval,
  - une validation sur jeu de test plus large avec métriques de ranking (MRR/Recall@k/nDCG).

## Décisions de conception justifiées

- Choix de partir sur des encodeurs multilingues: cohérent avec un corpus mixte FR/BM.
- Multiplication des tests de modèles: stratégie pertinente vu la variabilité observée.
- Passage progressif du test qualitatif vers une structure entraînable avec early stopping: bonne trajectoire pour fiabiliser la couche retrieval.

## Limites actuelles (à assumer en entretien)

- Évaluation majoritairement en exemples ciblés, pas encore en benchmark massif stabilisé.
- Absence dans ce notebook d'un tableau final consolidé de métriques ranking globales.
- Certaines familles de modèles testées (ex. Jina) ne sont pas optimisées nativement pour le couple linguistique ciblé.

## Ce que ce notebook apporte au produit final

- Une base expérimentale claire pour choisir la brique sémantique du RAG.
- Des preuves que le problème n'est pas trivial et qu'une calibration sérieuse est nécessaire.
- Une transition cohérente vers les phases de fine-tuning génératif mT5.

## Intégration FastAPI (cible)

Ce notebook prépare la couche qui alimentera un endpoint de type:

- `POST /retrieve`
  - Entrée: question (FR ou BM)
  - Sortie: top-k passages + scores cosinus

Recommandation de production:

- fixer un encodeur validé,
- indexer le corpus (FAISS ou équivalent),
- journaliser les scores et marges pour monitoring.

