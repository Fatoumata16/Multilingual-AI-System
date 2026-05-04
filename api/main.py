

# api_hybrid_optimized.py - Version avec retrieval amélioré (boost sur les enfants) + Cross-lingual
import os
import json
import faiss
import torch
import numpy as np
import unicodedata
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional

# ======================
# CONFIGURATION
# ======================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

EMBEDDINGS_PATH = "embeddings.npy"
IDS_PATH = "ids_map.npy"
PARENT_IDS_PATH = "parent_ids.npy"
LANGS_PATH = "langs.npy"
PARENTS_MAP_PATH = "parents_map.json"

RETRIEVER_ID = "Fatoumataa/embedding_billingue_francais_bambara_version3"
GENERATOR_ID = "Fatoumataa/mt5-bambara-resumer-boost1"

TOP_CHUNKS_TO_RETRIEVE = 10
TOP_CHUNKS_TO_MERGE = 2
CONF_THRESHOLD = 0.40

# 🔥 PARAMÈTRES MT5 (inchangés)
MAX_NEW_TOKENS = 70
MIN_NEW_TOKENS = 15
NUM_BEAMS = 6
REPETITION_PENALTY = 1.5
LENGTH_PENALTY = 0.9

# ======================
# 🔥 DICTIONNAIRE MOTS-CLÉS → PARENT_ID (inchangé)
# ======================
KEYWORD_TO_SOURCE = {
    # ===== ARTISANAT (MALI_ART_001) =====
    "numuw": "MALI_ART_001",
    "nɛgɛsotigiw": "MALI_ART_001",
    "forgerons": "MALI_ART_001",
    "forgeron": "MALI_ART_001",
    "bololabaara": "MALI_ART_001",
    
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ======================
# CHARGEMENT
# ======================
print("Chargement des modèles...")
retriever = SentenceTransformer(RETRIEVER_ID, device=str(device))
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_ID, use_fast=False)
generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_ID).to(device)
generator.eval()

embeddings = np.load(EMBEDDINGS_PATH)
ids = np.load(IDS_PATH, allow_pickle=True)
parent_ids = np.load(PARENT_IDS_PATH, allow_pickle=True)
langs = np.load(LANGS_PATH, allow_pickle=True)

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

with open(PARENTS_MAP_PATH, "r", encoding="utf-8") as f:
    parents_map = json.load(f)

print(f"✅ {len(parents_map)} documents chargés")
print(f"📚 {len(ids)} enfants dans l'index")

app = FastAPI()

# ======================
# 🔥 NOUVEAU : Request model avec cross-lingual
# ======================
class AskRequest(BaseModel):
    question: str
    langue_source: str = "bm"   # Langue de la question (bm ou fr)
    langue_cible: str = "fr"    # Langue de la réponse (bm ou fr)

# ======================
# NORMALISATION (SANS ESPACE AUTOUR DES POINTS)
# ======================
def normalize_text(text: str, lang: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace('–', '-').replace('—', '-')
    text = re.sub(r"^\d+\s+", "", text)
    text = text.strip(" -–—")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_for_mt5(context: str, question: str, lang: str) -> str:
    context_norm = normalize_text(context, lang)
    question_norm = normalize_text(question, lang)
    prefix = "summarize_bm: " if lang == "bm" else "summarize_fr: "
    return f"{prefix}question: {question_norm} contexte: {context_norm}"

# ======================
# 🔥 RETRIEVAL (inchangé)
# ======================
def retrieve_chunks(query: str, lang: str, top_k: int = TOP_CHUNKS_TO_MERGE) -> List[Dict]:
    """
    Récupère les chunks avec boost sur les parents ciblés par mots-clés.
    Le boost s'applique à TOUS les enfants de ce parent.
    """
    # 1. Détection des parents cibles via mots-clés
    target_parents = set()
    query_lower = query.lower()
    
    for keyword, parent_id in KEYWORD_TO_SOURCE.items():
        if keyword.lower() in query_lower:
            target_parents.add(parent_id)
            print(f"🔑 Mot-clé trouvé: '{keyword}' → parent: {parent_id}")
    
    # 2. Recherche sémantique dans les enfants
    q_emb = retriever.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    scores, idxs = index.search(q_emb, TOP_CHUNKS_TO_RETRIEVE * 3)
    
    # 3. Collecte des enfants avec boost sur leurs parents
    children_scores = {}  # child_id -> (score, parent_id)
    
    for i, idx in enumerate(idxs[0]):
        if i >= len(langs) or langs[idx] != lang:
            continue
        
        child_id = ids[idx]
        parent_id_val = parent_ids[idx]
        score = float(scores[0][i])
        
        # 🔥 BOOST : si le parent de cet enfant est dans target_parents
        if parent_id_val in target_parents:
            score += 0.25  # Boost de 0.25
            print(f"🚀 Boost enfant {child_id} (parent: {parent_id_val}) → score: {score:.3f}")
        
        # Garder le meilleur score pour chaque enfant
        if child_id not in children_scores or score > children_scores[child_id][0]:
            children_scores[child_id] = (score, parent_id_val)
    
    # 4. Regroupement par parent (prendre le meilleur score parmi ses enfants)
    parent_results = {}
    for child_id, (score, parent_id_val) in children_scores.items():
        if parent_id_val not in parent_results or score > parent_results[parent_id_val]["score"]:
            parent_results[parent_id_val] = {
                "parent_id": parent_id_val,
                "score": score,
                "context": parents_map[parent_id_val][lang]
            }
    
    chunks = list(parent_results.values())
    chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Debug : afficher les top chunks
    print(f"\n📊 Top parents après boosting:")
    for i, c in enumerate(chunks[:3]):
        print(f"   {i+1}. {c['parent_id']} - score: {c['score']:.3f}")
    
    return chunks[:top_k]

# ======================
# GÉNÉRATION (INCHANGÉE)
# ======================
def generate_answer(context: str, question: str, lang: str) -> str:
    input_text = preprocess_for_mt5(context, question, lang)
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = generator.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=3,
            length_penalty=LENGTH_PENALTY,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

# ======================
# 🔥 API MODIFIÉE POUR LE CROSS-LINGUAL
# ======================

@app.post("/ask")
async def ask(request: AskRequest):
    # Langues
    source_lang = request.langue_source  # bm ou fr (langue de la question)
    target_lang = request.langue_cible   # bm ou fr (langue de la réponse)
    
    # Validation
    if source_lang not in ["bm", "fr"]:
        source_lang = "bm"
    if target_lang not in ["bm", "fr"]:
        target_lang = "fr"
    
    # 🔥 Pour la recherche, on utilise la langue source
    # (le retrieveur cross-lingual comprend les deux langues)
    search_lang = source_lang
    
    # 1. Retrieval (recherche dans la langue source)
    chunks = retrieve_chunks(request.question, search_lang, TOP_CHUNKS_TO_MERGE)
    
    if not chunks:
        return {
            "reponse": "N tɛ dɔnniya sɔrɔ." if target_lang == "bm" else "Aucune information trouvée.",
            "question": request.question,
            "question_langue": source_lang,
            "reponse_langue": target_lang
        }
    
    best_chunk = chunks[0]
    
    if best_chunk["score"] < CONF_THRESHOLD:
        return {
            "reponse": "N bɛɛtɛ, i bɛ se ka ladilan fana?" if target_lang == "bm" else "Je ne suis pas sûr, peux-tu reformuler ?",
            "question": request.question,
            "question_langue": source_lang,
            "reponse_langue": target_lang,
            "source": best_chunk["parent_id"],
            "confiance": round(best_chunk["score"], 3)
        }
    
    # Fusion intelligente (seulement si le score du 2ème est proche)
    if len(chunks) > 1 and (chunks[1]["score"] > best_chunk["score"] * 0.8):
        merged_context = " ".join([c["context"] for c in chunks])
    else:
        merged_context = best_chunk["context"]
    
    # 🔥 CLÉ : Récupérer le contexte dans la LANGUE CIBLE
    context = parents_map[best_chunk["parent_id"]][target_lang]
    
    # 3. Génération dans la langue cible
    answer = generate_answer(context, request.question, target_lang)
    
    return {
        "question": request.question,
        "question_langue": source_lang,
        "reponse": answer,
        "reponse_langue": target_lang,
        "source": best_chunk["parent_id"],
        "confiance": round(best_chunk["score"], 3)
    }


# ======================
# 🔥 ENDPOINT COMPATIBILITÉ ANCIENNE VERSION
# ======================
@app.get("/ask")
async def ask_get(question: str, lang: str = "fr"):
    """Version GET pour compatibilité (utilise ancien format)"""
    request = AskRequest(question=question, langue_source=lang, langue_cible=lang)
    return await ask(request)


@app.get("/ask/cross")
async def ask_cross(question: str, source: str = "bm", target: str = "fr"):
    """Version GET simplifiée pour le cross-lingual"""
    request = AskRequest(question=question, langue_source=source, langue_cible=target)
    return await ask(request)


@app.get("/debug/retrieval")
async def debug_retrieval(question: str, lang: str = "bm"):
    """Debug pour voir le scoring des chunks"""
    lang_tag = "bm" if lang.lower().startswith("b") else "fr"
    
    # Détection des parents cibles
    target_parents = set()
    query_lower = question.lower()
    for keyword, parent_id in KEYWORD_TO_SOURCE.items():
        if keyword.lower() in query_lower:
            target_parents.add(parent_id)
    
    # Recherche sémantique
    q_emb = retriever.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, 20)
    
    # Collecte par parent
    parent_scores = {}
    for i, idx in enumerate(idxs[0]):
        if i >= len(langs) or langs[idx] != lang_tag:
            continue
        pid = parent_ids[idx]
        score = float(scores[0][i])
        if pid not in parent_scores or score > parent_scores[pid]["original"]:
            parent_scores[pid] = {
                "original": score,
                "boosted": score + (0.25 if pid in target_parents else 0)
            }
    
    results = []
    for pid, scores in parent_scores.items():
        results.append({
            "id": pid,
            "score_original": round(scores["original"], 3),
            "score_boosted": round(scores["boosted"], 3),
            "boost_applique": pid in target_parents
        })
    
    results.sort(key=lambda x: x["score_boosted"], reverse=True)
    
    return {
        "question": question,
        "parents_cibles": list(target_parents),
        "top_parents": results[:10]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎓 BAMBARA QA - RETRIEVAL AMÉLIORÉ + CROSS-LINGUAL")
    print("="*60)
    print(f"✅ Keyword boosting: {len(KEYWORD_TO_SOURCE)} associations")
    print(f"✅ Boost sur les enfants des parents cibles (+0.25)")
    print(f"✅ Fusion intelligente (seuil: 80%)")
    print(f"✅ CROSS-LINGUAL: Question BM → Réponse FR / Question FR → Réponse BM")
    print(f"✅ max_new_tokens={MAX_NEW_TOKENS} (inchangé)")
    print("\n🌐 http://localhost:8000")
    print("📖 /debug/retrieval?question=... pour analyser")
    print("\n🔤 Exemples cross-lingual:")
    print("   POST /ask -d '{\"question\": \"...\", \"langue_source\": \"bm\", \"langue_cible\": \"fr\"}'")
    print("   GET /ask/cross?question=...&source=bm&target=fr")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)














































