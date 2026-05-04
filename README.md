# Chatbot Educatif Bilingue (Français-Bambara)

Ce projet construit un assistant éducatif bilingue pour une langue peu dotée (bambara) en combinant:

- un **retrieval sémantique** (compréhension et recherche de contexte),
- un **générateur mT5 fine-tuné** (synthèse/résumé),
- une **orchestration FastAPI** (pipeline exploitable en API).

L'ambition est double: produire un système utile en production et démontrer une démarche de recherche appliquée et rigoureuse.

## Pourquoi cette approche est pertinente

Le bambara souffre de trois contraintes majeures:

- ressources textuelles limitées,
- faible disponibilité de modèles spécialisés,
- budget GPU contraint pour des entraînements longs.

La stratégie retenue répond directement à ces contraintes:

- découpage en plusieurs notebooks spécialisés (itérations indépendantes, reprises plus simples),
- progression par couches (sémantique -> adaptation mT5 -> boost final),
- checkpointing/push réguliers pour fiabilité expérimentale.

## Résultats clés (ce qui prouve que la stratégie fonctionne)

## 1) Modèle sémantique — Phase 1 (Monolingual Fine-tuning)

`1082` échantillons, `6` epochs.

| Step | Train Loss | Val Loss | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Precision@3 | Recall@10 | nDCG@10 | MRR@10 | MAP@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0730 | 0.2527 | 0.835 | 0.950 | 0.965 | 0.985 | 0.3167 | 0.985 | 0.9190 | 0.8971 | 0.8977 |
| 200 | 0.0407 | 0.1848 | 0.825 | 0.960 | 0.980 | 0.985 | 0.3200 | 0.985 | 0.9164 | 0.8931 | 0.8941 |
| 300 | 0.0310 | 0.1943 | 0.825 | 0.955 | 0.970 | 0.980 | 0.3183 | 0.980 | 0.9133 | 0.8908 | 0.8918 |
| 400 | 0.0131 | 0.1814 | 0.840 | 0.965 | 0.970 | 0.980 | 0.3217 | 0.980 | 0.9196 | 0.8992 | 0.9003 |

Lecture: la phase monolingue atteint une excellente qualité de ranking (`MRR@10 ~ 0.90`, `MAP@100 ~ 0.90`) avec une bonne précision top-1.

## 2) Modèle sémantique — Phase 2 (Cross-lingual Adaptation)

`1125` échantillons, `6` epochs.

| Step | Train Loss | Val Loss | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Precision@3 | Recall@10 | nDCG@10 | MRR@10 | MAP@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.3230 | 0.3195 | 0.585 | 0.895 | 0.965 | 0.975 | 0.2983 | 0.975 | 0.7958 | 0.7357 | 0.7374 |
| 200 | 0.0799 | 0.2243 | 0.695 | 0.920 | 0.965 | 0.980 | 0.3067 | 0.980 | 0.8515 | 0.8085 | 0.8100 |
| 300 | 0.0544 | 0.1854 | 0.720 | 0.935 | 0.980 | 0.995 | 0.3117 | 0.995 | 0.8702 | 0.8285 | 0.8289 |
| 400 | 0.0439 | 0.1874 | 0.735 | 0.935 | 0.975 | 0.995 | 0.3117 | 0.995 | 0.8755 | 0.8357 | 0.8360 |

Lecture: l'adaptation cross-lingue dégrade naturellement le top-1 par rapport au monolingue, mais maintient un excellent rappel à `k=10` (`0.995`), ce qui est idéal pour une architecture RAG (on récupère très souvent le bon contexte dans le top-k).

## 3) mT5 — Fine-tuning résumé (8 epochs)

| Epoch | Train Loss | Val Loss | Rouge1 | Rouge2 | RougeL | RougeLsum |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 35.3040 | 3.8356 | 0.4562 | 0.2151 | 0.3057 | 0.3056 |
| 2 | 32.9291 | 3.6969 | 0.4756 | 0.2304 | 0.3237 | 0.3239 |
| 3 | 31.6170 | 3.5895 | 0.4849 | 0.2372 | 0.3314 | 0.3314 |
| 4 | 30.4512 | 3.4988 | 0.4863 | 0.2388 | 0.3343 | 0.3344 |
| 5 | 29.9236 | 3.4431 | 0.4915 | 0.2443 | 0.3402 | 0.3403 |
| 6 | 29.3764 | 3.3960 | 0.5022 | 0.2513 | 0.3459 | 0.3459 |
| 7 | 29.0224 | 3.3521 | 0.5072 | 0.2553 | 0.3517 | 0.3517 |
| 8 | 28.6897 | 3.3116 | 0.5090 | 0.2569 | 0.3540 | 0.3542 |

Lecture: progression régulière des scores ROUGE et baisse de la validation loss, indiquant une amélioration stable de la qualité de résumé.

## 4) mT5 — Boost 1 (8 epochs supplémentaires)

Derniers résultats observés:

| Epoch | Train Loss | Val Loss | Rouge1 | Rouge2 | RougeL | RougeLsum |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 42.8678 | 2.4828 | 0.5279 | 0.2721 | 0.3723 | 0.3720 |
| 7 | 42.8899 | 2.4752 | 0.5265 | 0.2721 | 0.3719 | 0.3718 |
| 8 | 42.9689 | 2.4747 | 0.5265 | 0.2725 | 0.3724 | 0.3723 |

**Résultat majeur**: `RougeL = 0.3724` sur une langue peu dotée avec peu de données est un niveau très solide et valide la stratégie d'entraînement par étapes.

## Message principal pour recruteur

Ce projet ne se limite pas à "faire tourner un modèle":

- il démontre une stratégie expérimentale adaptée à un contexte réel de contraintes GPU,
- il articule retrieval + génération dans une architecture RAG cohérente,
- il aboutit à un gain mesurable jusqu'à `RougeL ~ 0.372`,
- il est industrialisable via FastAPI (voir `docs/README_main_fastapi.md`).

## Architecture globale (vue simple)

1. Question utilisateur (FR ou BM)
2. Embedding query + recherche FAISS (retrieval sémantique)
3. Sélection/fusion de contexte
4. Prompt structuré `summarize_fr` ou `summarize_bm`
5. Génération mT5 fine-tunée
6. Réponse bilingue + source + score de confiance

## Détails par composant

- `docs/README_modele_semantique_rag.md`: couche embeddings/retrieval.
- `docs/README_phase0_rag.md`: préparation des données + phase 0 mT5.
- `docs/README_phase_fine_turning_bambara_rag_1.md`: phase 1 résumé.
- `docs/README_fine_turning_bambara_français_2.md`: boost final.
- `docs/README_main_fastapi.md`: orchestration API complète.

## Structure recommandée du dépôt

```text
.
├── README.md
├── api/
│   └── main.py
├── notebooks/
│   ├── modele_semantique_rag.ipynb
│   ├── phase0_rag.ipynb
│   ├── phase_fine_turning_bambara_rag_1.ipynb
│   └── fine_turning_bambara_français_2.ipynb
└── docs/
    ├── README_main_fastapi.md
    ├── README_modele_semantique_rag.md
    ├── README_phase0_rag.md
    ├── README_phase_fine_turning_bambara_rag_1.md
    └── README_fine_turning_bambara_français_2.md
```

## Objectif d'impact

Fournir un chatbot éducatif bilingue fiable pour améliorer l'accès au savoir en contexte multilingue africain, tout en proposant une base technique réutilisable pour d'autres langues peu dotées.
